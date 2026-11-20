import csv
import gpytorch
import gymnasium as gym
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import re
import torch
import warnings

from collections import deque
from gymnasium.utils.env_checker import check_env
from scipy.stats import entropy
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.metrics import mean_squared_error, r2_score, pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import cross_val_score, cross_val_predict
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.monitor import load_results as _lr
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from typing import Optional

class ARDExactGPModel_Dense(gpytorch.models.ExactGP):
    """Gaussian Process Architecture for Dense Datasets (i.e Composition datasets)

        Params:
            - train_x: train features
            - train_y: train labels
            - likelihood: likelihood of Gaussian Process  
        """
    def __init__(self, 
                 train_x: torch.Tensor, 
                 train_y: torch.Tensor, 
                 likelihood: gpytorch.likelihoods):
        super().__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ConstantMean()

        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                ard_num_dims=train_x.shape[1]
            )
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x) 

class GPRegressor(BaseEstimator, RegressorMixin):
    """Scikit-learn wrapper around ARDExactGPModel_Dense."""

    def __init__(self, lr=0.1, training_iter=100, device='cpu', early_stop_tol:float = 1e-3, patience: int = 5, warm_start: bool = True, jitter: float = 1e-4):
        self.lr = lr 
        self.training_iter = training_iter
        self.device = device
        self.early_stop_tol = early_stop_tol
        self.warm_start = warm_start
        self.patience = patience
        self.jitter = jitter
        self._warm_state = None

    def fit(self, X, y):
        # Convert to tensors (sklearn passes numpy arrays)
        train_x = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=self.device)
        train_y = torch.as_tensor(np.asarray(y), dtype=torch.float32, device=self.device).ravel()

        # Fresh likelihood + model for THIS fold
        self.likelihood_ = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        self.model_ = ARDExactGPModel_Dense(train_x, train_y, self.likelihood_).to(self.device)
        
        # Warm-start from previously learned hyperparameters when shapes match
        if self.warm_start and self._warm_state is not None:
            try:
                self.model_.load_state_dict(self._warm_state["model"], strict=False)
                self.likelihood_.load_state_dict(self._warm_state["lik"], strict=False)
            except Exception:
                pass  # shape mismatch (e.g. ard dims changed) -> fresh start

        self.model_.train()
        self.likelihood_.train()

        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood_, self.model_)
        
        best_loss = float("inf")
        no_improve = 0

        # Added numerical jitter + fast settings reduce Cholesky failures/time
        with gpytorch.settings.cholesky_jitter(self.jitter), \
             gpytorch.settings.max_cholesky_size(2000):
            for _ in range(self.training_iter):
                optimizer.zero_grad()
                output = self.model_(train_x)
                loss = -mll(output, train_y)
                loss.backward()
                optimizer.step()

                cur = loss.item()
                if best_loss - cur > self.early_stop_tol:
                    best_loss = cur
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= self.patience:
                        break  # MLL plateaued -> stop early

        if self.warm_start:
            self._warm_state = {
                "model": {k: v.detach().clone() for k, v in self.model_.state_dict().items()},
                "lik": {k: v.detach().clone() for k, v in self.likelihood_.state_dict().items()},
            }
        return self
    
    def predict(self, X):
        test_x = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=self.device)

        self.model_.eval()
        self.likelihood_.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.max_cholesky_size(2000):
            preds = self.likelihood_(self.model_(test_x))
            mean = preds.mean

        return mean.cpu().numpy()

class CRDADataGenerator(gym.Env):
    metadata = {"render_modes":[]}

    def __init__(self,
                 X: np.ndarray,
                 y: np.ndarray,
                 model: BaseEstimator,
                 residual: np.ndarray,
                 MSE: float,
                 div_low: float,
                 div_high: float,
                 reward_clip:float=1.0,
                 verbose: bool=False,
                 cv:int=10,
                 history_length:int=5,
                 similarity_threshold:float=0.95,
                 bins:int=10
                 ):
        super().__init__()

        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32)

        self.residual = np.asarray(residual, dtype=np.float32)

        self.dataset_length = self.X.shape[0]
        self.bins = bins
        self.model = model
        self.MSE = MSE
        self.entropy = self._avg_entropy_cal(self.X, self.bins)

        self.current_step = 0
        self.div_low = div_low
        self.div_high = div_high

        self.history_length = history_length
        self.n_features = self.X.shape[-1]

        self.reward_clip = reward_clip
        self.verbose = verbose
        self.cv = cv
        self.similarity_threshold = similarity_threshold

        # For Reset
        self._datapoints_init = self.X.copy()
        self._targets_init = self.y.copy()
        self._residual_init = self.residual.copy()
        self._MSE_init = self.MSE
        self._entropy_init = self.entropy
        self._init_params = self.model.get_params()

        # Action space
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_features,), dtype=np.float32
        )

        # Observation space
        self.observation_space = gym.spaces.Dict({
            "mse": gym.spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
            "n_points": gym.spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
            "entropy": gym.spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
            "diversity": gym.spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
            "corr_res": gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
            "corr_feat_w_y": gym.spaces.Box(low=-1.0, high=1.0, shape=(self.n_features,), dtype=np.float32),
            "action_history": gym.spaces.Box(low=-1.0, high=1.0, shape=(self.history_length, self.n_features), dtype=np.float32),
            "feat_mean": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_features,), dtype=np.float32),
            "target_mean": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            "feat_std": gym.spaces.Box(low=0.0, high=np.inf, shape=(self.n_features,), dtype=np.float32),
            "target_std": gym.spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
            "feat_min": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_features,), dtype=np.float32),
            "target_min": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            "feat_max": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_features,), dtype=np.float32),
            "target_max": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            "feat_median": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_features,), dtype=np.float32),
            "target_median": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            "feat_95th_percentile": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_features,), dtype=np.float32),
            "target_95th_percentile": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            "feat_5th_percentile": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self .n_features,), dtype=np.float32),
            "target_5th_percentile": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            "feat_skewness": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_features,), dtype=np.float32),
            "target_skewness": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            "feat_kurtosis": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_features,), dtype=np.float32),
            "target_kurtosis": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            })

        self.state = None

    def _avg_entropy_cal(self, X: np.ndarray, bins: int) -> float:
        """Calculate the average entropy of the features in X."""
        entropies = []
        for i in range(X.shape[1]):
            hist, _ = np.histogram(X[:, i], bins=bins, density=True)
            hist = hist[hist > 0]  # Remove zero entries to avoid log(0)
            entropies.append(entropy(hist))
        return np.mean(entropies)

    def _get_observation(self, MSE: float, X: np.ndarray, y: np.ndarray, residual: np.ndarray, entropy: float, action_history: deque) -> dict:
            """Construct the observation dictionary."""
            obs = {
                "mse": np.array([MSE], dtype=np.float32),
                "n_points": np.array([X.shape[0]], dtype=np.float32),
                "entropy": np.array([entropy], dtype=np.float32),
                "diversity": np.array([self._calculate_diversity(X)], dtype=np.float32),
                "corr_res": np.array([np.corrcoef(residual, y)[0, 1]], dtype=np.float32) if len(residual) > 1 else np.array([0.0], dtype=np.float32),
                "corr_feat_w_y": np.array([np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])], dtype=np.float32),
                "action_history": np.array(action_history, dtype=np.float32),
                "feat_mean": np.array(np.mean(X, axis=0), dtype=np.float32),
                "target_mean": np.array([np.mean(y)], dtype=np.float32),
                "feat_std": np.array(np.std(X, axis=0), dtype=np.float32),
                "target_std": np.array([np.std(y)], dtype=np.float32),
                "feat_min": np.array(np.min(X, axis=0), dtype=np.float32),
                "target_min": np.array([np.min(y)], dtype=np.float32),
                "feat_max": np.array(np.max(X, axis=0), dtype=np.float32),
                "target_max": np.array(np.max(y), dtype=np.float32).reshape(1, ),
                "feat_median": np.array(np.median(X, axis=0), dtype=np.float32),
                "target_median": np.array(np.median(y), dtype=np.float32).reshape(1, ),
                "feat_95th_percentile": np.array(np.percentile(X, 95, axis=0), dtype=np.float32),
                "target_95th_percentile": np.array(np.percentile(y, 95), dtype=np.float32).reshape(1, ),
                "feat_5th_percentile": np.array(np.percentile(X, 5, axis=0), dtype=np.float32),
                "target_5th_percentile": np.array(np.percentile(y, 5), dtype=np.float32).reshape(1, ),
                "feat_skewness": np.array([np.mean(((X[:, i] - np.mean(X[:, i])) / np.std(X[:, i]))**3) for i in range(X.shape[1])], dtype=np.float32),
                "target_skewness": np.array([np.mean(((y - np.mean(y)) / np.std(y))**3)], dtype=np.float32),
                "feat_kurtosis": np.array([np.mean(((X[:, i] - np.mean(X[:, i])) / np.std(X[:, i]))**4) - 3 for i in range(X.shape[1])], dtype=np.float32),
                "target_kurtosis": np.array([np.mean(((y - np.mean(y)) / np.std(y))**4) - 3], dtype=np.float32),
            }
            return obs

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed, options=options)

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)

        self.X = self._datapoints_init.copy()
        self.y = self._targets_init.copy()
        self.residual = self._residual_init.copy()
        self.MSE = self._MSE_init
        self.entropy = self._entropy_init

        self.current_step = 0
        self.action_history = deque(maxlen=self.history_length)

        # Initialize action history with zeros
        for _ in range(self.history_length):
            self.action_history.append(np.zeros(self.n_features, dtype=np.float32))

        self.model.set_params(**self._init_params)

        # Update state
        obs = self._get_observation(self.MSE, self.X, self.y, self.residual, self.entropy, self.action_history)

        self.state = self._get_state(self.X, self.y)

        return obs, {}

    def _get_state(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Construct the state dictionary."""
        state = {
            "feat_mean": np.mean(X, axis=0).astype(np.float32),
            "target_mean": np.mean(y).astype(np.float32),
            "feat_std": np.std(X, axis=0).astype(np.float32),
            "target_std": np.std(y).astype(np.float32),
            "feat_min": np.min(X, axis=0).astype(np.float32),
            "target_min": np.min(y).astype(np.float32),
            "feat_max": np.max(X, axis=0).astype(np.float32),
            "target_max": np.max(y).astype(np.float32),
            "feat_median": np.median(X, axis=0).astype(np.float32),
            "target_median": np.median(y).astype(np.float32),
            "feat_95th_percentile": np.percentile(X, 95, axis=0).astype(np.float32),
            "target_95th_percentile": np.percentile(y, 95).astype(np.float32),
            "feat_5th_percentile": np.percentile(X, 5, axis=0).astype(np.float32),
            "target_5th_percentile": np.percentile(y, 5).astype(np.float32),
            "feat_skewness": np.array([np.mean(((X[:, i] - np.mean(X[:, i])) / np.std(X[:, i]))**3) for i in range(X.shape[1])], dtype=np.float32),
            "target_skewness": np.array([np.mean(((y - np.mean(y)) / np.std(y))**3)], dtype=np.float32),
            "feat_kurtosis": np.array([np.mean(((X[:, i] - np.mean(X[:, i])) / np.std(X[:, i]))**4) - 3 for i in range(X.shape[1])], dtype=np.float32),
            "target_kurtosis": np.array([np.mean(((y - np.mean(y)) / np.std(y))**4) - 3], dtype=np.float32),
        }
        return state

    def _calculate_diversity(self, X: np.ndarray) -> float:
        """Calculate the diversity of the dataset based on pairwise distances."""
        if X.shape[0] < 2:
            return 0.0
        pairwise_dist = pairwise_distances(X)
        diversity = np.mean(pairwise_dist)
        return diversity

    def step(self, action: np.ndarray):
        raw_action = np.clip(action, -1.0, 1.0)

        candidate = self._get_candidate_point(raw_action)

        self.action_history.append(raw_action)

        y_label = self.model.predict(candidate.reshape(1, -1)).item()
        similarity = cosine_similarity(candidate.reshape(1, -1), self.X).max()

        if similarity > self.similarity_threshold:
            reward = -self.reward_clip
            info = {
                "accepted": False,
                "duplicate": True,
                "invalid_action": True,
                "novelty_reward": 0.0,
                "entropy_reward": 0.0,
                "similarity": float(similarity),
                "entropy": float(self.entropy),
                "MSE": float(self.MSE),
                "n_points": int(self.X.shape[0]),
                "reason": "Similarity threshold exceeded."
                }
            obs = self._get_observation(self.MSE, self.X, self.y, self.residual, self.entropy, self.action_history)
            truncated = (len(self.X) >= int(1.2 * self.dataset_length)
                         or self.current_step >= 500)
            return obs, reward, False, truncated, info

        nn_idx = int(np.argmax(similarity))
        y_label += self.residual[nn_idx]

        X_new = np.vstack([self.X, candidate])
        y_new = np.append(self.y, y_label)

        eval_model = clone(self.model)
        y_pred = cross_val_predict(eval_model, X_new, y_new, cv=self.cv, n_jobs=1)
        MSE_new = mean_squared_error(y_new, y_pred)
        residual_new = y_new - y_pred
        entropy_mean = self._avg_entropy_cal(X_new, self.bins)
        
        accepted = False
        novelty_reward = 0.0
        entropy_reward = 0.0

        if entropy_mean > self.entropy:
            accepted = True
            entropy_gain  = entropy_mean - self.entropy
            
            self.entropy = entropy_mean
            self.model = eval_model
            self.X = X_new
            self.y = y_new
            self.residual = residual_new
            self.MSE = MSE_new
            
            log_r2 = self._log_r_score_cal(y_new, y_pred)
            novelty_reward = float(log_r2)
            entropy_reward = float(entropy_gain)
            reward = novelty_reward + entropy_reward
        else:
            entropy_reward = (entropy_mean - self.entropy)
            reward = entropy_reward

        truncated = (len(self.X) >= int(1.2 * self.dataset_length) or self.current_step >= 500)
        terminated = False

        obs = self._get_observation(self.MSE, self.X, self.y, self.residual, self.entropy, self.action_history)

        info = {
            "accepted": accepted,
            "duplicate": False,
            "invalid_action": False,
            "novelty_reward": novelty_reward,
            "entropy_reward": entropy_reward,
            "reward": float(reward),
            "similarity": float(similarity),
            "entropy": float(self.entropy),
            "MSE": float(self.MSE),
            "n_points": int(self.X.shape[0])
        }
        
        if self.verbose:
            print(f"Accepted: {info['accepted']} | Novelty Reward: {info['novelty_reward']} | Entropy Reward: {info['entropy_reward']} | Reward: {info['reward']} | Entropy: {info['entropy']} | Entropy Mean: {entropy_mean}")
        
        return obs, float(reward), terminated, truncated, info

    def _get_candidate_point(self, action: np.ndarray) -> np.ndarray:
        """Generate a candidate point based on the action."""
        # Scale action to the feature space
        self.current_step += 1
        scaled_action = (action + 1) / 2  # Scale from [-1, 1] to [0, 1]
        weights = scaled_action / np.sum(scaled_action)  # Normalize to sum to 1

        return weights.astype(np.float32)

    def _log_r_score_cal(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate the log of the R^2 score."""
        r2 = r2_score(y_true, y_pred)
        return np.log(1/np.maximum(1.0 - r2, 1e-6))  # Add a small value to avoid log(0)

class RLTrainingLogger(BaseCallback):

    def __init__(self, verbose=0):
        super().__init__(verbose)

        # Episode-level
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_accepted = []      # accepted count per episode
        self.episode_duplicates = []    # duplicate count per episode
        self.episode_final_mse = []
        self.episode_final_entropy = []
        self.episode_final_npoints = []

        # Step-level
        self.accepted = []
        self.duplicates = []
        self.novelty_rewards = []
        self.entropy_rewards = []
        self.step_rewards = []
        self.mse = []
        self.entropy = []
        self.n_points = []

        # Running accumulators
        self.current_reward = 0.0
        self.current_length = 0
        self.current_accepted = 0
        self.current_duplicates = 0

    def _on_step(self):
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]
        infos = self.locals["infos"]

        for reward, done, info in zip(rewards, dones, infos):
            self.current_reward += float(reward)
            self.current_length += 1

            acc = int(info.get("accepted", False))
            dup = int(info.get("duplicate", False))

            self.current_accepted += acc
            self.current_duplicates += dup

            # step-level logs
            self.accepted.append(acc)
            self.duplicates.append(dup)
            self.novelty_rewards.append(float(info.get("novelty_reward", 0.0)))
            self.entropy_rewards.append(float(info.get("entropy_reward", 0.0)))
            self.step_rewards.append(float(reward))
            self.mse.append(float(info.get("MSE", np.nan)))
            self.entropy.append(float(info.get("entropy", np.nan)))
            self.n_points.append(int(info.get("n_points", 0)))

            # live TensorBoard scalars (optional but handy)
            self.logger.record("rollout/accepted", acc)
            self.logger.record("rollout/duplicate", dup)
            self.logger.record("rollout/mse", info.get("MSE", np.nan))
            self.logger.record("rollout/entropy", info.get("entropy", np.nan))

            if done:
                self.episode_rewards.append(self.current_reward)
                self.episode_lengths.append(self.current_length)
                self.episode_accepted.append(self.current_accepted)
                self.episode_duplicates.append(self.current_duplicates)
                self.episode_final_mse.append(float(info.get("MSE", np.nan)))
                self.episode_final_entropy.append(float(info.get("entropy", np.nan)))
                self.episode_final_npoints.append(int(info.get("n_points", 0)))

                self.current_reward = 0.0
                self.current_length = 0
                self.current_accepted = 0
                self.current_duplicates = 0

        return True

    def _on_training_end(self):
        pd.DataFrame({
            "episode": np.arange(1, len(self.episode_rewards) + 1),
            "episode_reward": self.episode_rewards,
            "episode_length": self.episode_lengths,
            "accepted_count": self.episode_accepted,
            "duplicate_count": self.episode_duplicates,
            "final_mse": self.episode_final_mse,
            "final_entropy": self.episode_final_entropy,
            "final_n_points": self.episode_final_npoints,
        }).to_csv("ppo_learning_curve_2.csv", index=False)

        pd.DataFrame({
            "timestep": np.arange(1, len(self.accepted) + 1),
            "accepted": self.accepted,
            "duplicate": self.duplicates,
            "novelty_reward": self.novelty_rewards,
            "entropy_reward": self.entropy_rewards,
            "step_reward": self.step_rewards,
            "mse": self.mse,
            "entropy": self.entropy,
            "n_points": self.n_points,
        }).to_csv("ppo_generation_metrics_2.csv", index=False)

def make_env(X, y, residual, MSE, div_low, div_high, rank, seed=42, log_dir="./logs/"):
    """Factory for a single CRDADataGenerator wrapped in Monitor.
    Rebuilds + refits the base model inside the worker for clean pickling."""
    def _init():
        
        torch.set_num_threads(1)  # critical: 1 thread per worker process

        # Rebuild and refit the base model locally in this worker
        local_model = GPRegressor()
        local_model.fit(X, y)

        env = CRDADataGenerator(
            X=X, y=y,
            model=local_model,
            residual=residual,
            MSE=MSE,
            div_low=div_low,
            div_high=div_high,
            reward_clip=1.0,
            verbose=False,
            cv=5,
        )
        env.reset(seed=seed + rank)   # distinct seed per worker
        env = Monitor(env, filename=os.path.join(log_dir, f"monitor_env{rank}"))
        return env
    return _init


def set_seed(seed: int = 42):
    """Set the random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.deterministic = False


def main():
    warnings.filterwarnings('ignore')
    
    with open(file="steels_yield.json") as file:
        data = json.load(file)

    my_data = {}
    for i in range(len(data['columns'])):
        my_data[data['columns'][i]] = []
        for j in range(len(data['data'])):
            my_data[data['columns'][i]].append(data['data'][j][i])

    my_data['split_composition'] = []
    for composition in my_data['composition']:
        matches = re.findall(r'([A-Z][a-z]?)(\d*\.?\d+)', composition)
        my_data['split_composition'].append({element: float(value) for element, value in matches})

    key_count = [key.keys() for key in my_data['split_composition']]
    keys = list(max(key_count))

    composition_dict = {}

    for key in keys:
        composition_dict[key] = []
        for comp in my_data['split_composition']:
            if key in list(comp.keys()):
                composition_dict[key].append(comp[key])
            else:
                composition_dict[key].append(0)

    df = pd.DataFrame(data=composition_dict)
    df_targets = pd.DataFrame(my_data['yield strength'], columns=["yield_strength"])
    df_final = pd.concat([df, df_targets], axis=1)

    set_seed(42)

    X = np.array(df_final.drop(columns="yield_strength"))
    y = np.log(np.array(df_final['yield_strength']))

    model = GPRegressor()
    model.fit(X, y)

    residual = y - model.predict(X)

    residual_mean = residual.mean()
    residual_std = residual.std()
    factor = pd.DataFrame(y).skew().to_numpy().item()
    div_low = factor*(residual_mean - residual_std).item()
    div_high = factor*(residual_mean + residual_std).item()

    mean_scores=-cross_val_score(model, X, y, cv=10, scoring='neg_mean_squared_error', n_jobs=-1).mean()

    n_envs = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
    n_envs = max(1, min(n_envs - 1,256))
    print(f"Launching {n_envs} parallel environments")

    log_dir ="./logs/"
    os.makedirs(log_dir, exist_ok=True)
    
    env_fns = [
        make_env(X, y, residual, mean_scores, div_low, div_high,
                 rank=i, seed=42, log_dir=log_dir)
        for i in range(n_envs)
    ]
    vec_env = SubprocVecEnv(env_fns, start_method="spawn")
    vec_env = VecMonitor(vec_env, filename=os.path.join(log_dir, "vec_monitor"))
    
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(1)
    
    model = PPO(
        policy="MultiInputPolicy",
        env=vec_env,
        learning_rate=3e-4,
        batch_size=64,
        n_steps=2048,
        gamma=0.99,
        verbose=1,
        device='cpu',
        tensorboard_log='./tb_logs/'
    )

    model.learn(
        total_timesteps=10_000_000,
        progress_bar=True,
        callback=RLTrainingLogger(verbose=1)
    )

    model.save("NEW_DATA_GENERATOR_PPO_MODEL_1")
    vec_env.close()

if __name__ == '__main__':
    main()
