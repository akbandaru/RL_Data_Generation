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
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.monitor import load_results as _lr
from stable_baselines3 import PPO
from typing import Optional

warnings.filterwarnings('ignore')

def moving_average(values, window):
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")


def plot_reward(csv_path, window=50, title="Episode Reward"):
    df = pd.read_csv(csv_path, skiprows=1)
    rewards = df["r"].values
    episodes = np.arange(1, len(rewards) + 1)

    smooth = moving_average(rewards, window)
    ep_smooth = episodes[len(episodes) - len(smooth):]

    plt.figure(figsize=(8, 5))
    # plt.plot(episodes, rewards, alpha=0.3, label="Raw")
    plt.plot(ep_smooth, smooth, label=f"Moving avg (window={window})")
    # plt.axhline(113, color="red", linestyle="--", alpha=0.5, label="Step cap (113)")
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.imsave('RL_Data_Gen_LC.png')

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

    def __init__(self, lr=0.1, training_iter=100, device='cpu'):
        self.lr = lr 
        self.training_iter = training_iter
        self.device = device

    def fit(self, X, y):
        # Convert to tensors (sklearn passes numpy arrays)
        train_x = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=self.device)
        train_y = torch.as_tensor(np.asarray(y), dtype=torch.float32, device=self.device).ravel()

        # Fresh likelihood + model for THIS fold
        self.likelihood_ = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        self.model_ = ARDExactGPModel_Dense(train_x, train_y, self.likelihood_).to(self.device)

        self.model_.train()
        self.likelihood_.train()

        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood_, self.model_)

        for _ in range(self.training_iter):
            optimizer.zero_grad()
            output = self.model_(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()

        return self

    def predict(self, X):
        test_x = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=self.device)

        self.model_.eval()
        self.likelihood_.eval()

        with torch.no_grad():
            preds = self.likelihood_(self.model_(test_x))
            mean = preds.mean

        return mean.cpu().numpy()

import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Optional

from collections import deque
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import cross_val_score, cross_val_predict
from scipy.stats import entropy

class CRDADataGenEnvV2(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self,
                 X_train: np.ndarray,
                 y_train: np.ndarray,
                 model: BaseEstimator,
                 residual: np.ndarray,
                 MSE: float,
                 div_low: float,
                 div_high: float,
                 normalize_actions: bool = True,
                 reward_clip: float = 5.0,
                 verbose: bool = False,
                 cv=10,
                 history_len:int=3,
                 similarity_threshold:float=0.95):
        super().__init__()

        # --- commit to float32 NumPy arrays everywhere ---
        self.X = np.asarray(X_train, dtype=np.float32)
        self.y = np.asarray(y_train, dtype=np.float32).ravel()

        self.residual = np.asarray(residual, dtype=np.float32).ravel()

        self.dataset_length = len(self.X)
        self.model = model                       # labeling model (kept fixed)
        self.MSE = float(MSE)
        self.vendiScore = self.vendi_score(X_samples=self.X)
        self.div_low = div_low 
        self.div_high = div_high
        self.current_step = 0

        self.normalize_actions = normalize_actions
        self.reward_clip = reward_clip
        self.verbose = verbose
        self.cv = cv
        self.similarity_threshold = similarity_threshold

        # --- snapshots for reset ---
        self._datapoints_init = self.X.copy()
        self._targets_init = self.y.copy()
        self._residual_init = self.residual.copy()
        self._MSE_init = self.MSE
        self._init_vendiScore = self.vendiScore
        self._init_params = self.model.get_params(deep=True)

        self.history_len = int(history_len)
        self.n_features = self.X.shape[-1]

        self.action_space = gym.spaces.Box(
            low=0.0, high=10.0, shape=(self.n_features ,), dtype=np.float32
        )

        # Observation space
        self.observation_space = gym.spaces.Dict({
                    "mse": gym.spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
                    "n_points": gym.spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
                    "vendi": gym.spaces.Box(low=1.0, high=np.inf, shape=(1,), dtype=np.float32),
                    "diversity": gym.spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
                    "corr_res": gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
                    "corr_feat_w_y": gym.spaces.Box(low=-1.0, high=1.0, shape=(self.n_features,), dtype=np.float32),
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
        

    # ------------------------------------------------------------------ #
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed, options=options)

        if seed is not None:
            np.random.seed(seed)

        self.current_step = 0
        self.X = self._datapoints_init.copy()
        self.y = self._targets_init.copy()
        self.residual = self._residual_init.copy()
        self.MSE = self._MSE_init
        self.vendiScore = self._init_vendiScore
        self.state = np.zeros(shape=(self.n_features, ))

        # history buffer: filled with zero-vectors (= "no action yet")

        self.model.set_params(**self._init_params)

        obs = self._get_observation(self.MSE, self.X, self.y, residual=self.residual, vendi=self.vendiScore)
        
        return obs, {}

    # ------------------------------------------------------------------ #
    def step(self, action):
        self.current_step += 1
        action = np.asarray(action, dtype=np.float32).reshape(1, -1)
        action_sum = np.sum(action)

        if action_sum > 0:
            action = action / action_sum
        else:
            action = np.ones_like(action) / len(action)

        self.state = action
        # --- enforce simplex constraint by normalization (preferred) ---
        invalid = False
        if self.normalize_actions:
            s = action.sum()
            if s <= 0:
                invalid = True
            else:
                action = action / s
        else:
            if not (0.9950 <= action.sum() <= 1.0001):
                invalid = True


        # Pseudo-label the proposed point using the fixed labeling model.
        y_label = self.model.predict(action).ravel()
        similarity = cosine_similarity(action, self.X)
        max_similarity = np.max(similarity)

        if max_similarity > self.similarity_threshold:
            invalid = True

        nn_idx = int(np.argmax(similarity))
        y_label = y_label + self.residual[nn_idx]

        # Candidate augmented dataset (not yet committed).
        X_new = np.concatenate((self.X, action), axis=0)
        y_new = np.concatenate((self.y, y_label), axis=0)

        # Evaluate with a fresh evaluation model on held-out data.
        eval_model = clone(self.model)   # or your GPRegressor()
        eval_model.fit(X_new, y_new)
        y_pred_train = eval_model.predict(X_new)
        residual_new = y_new - y_pred_train
        # print(f"Action: {action}")
        # print(f"y_label: {y_label}")
        # print(f"Last element in X_new: {X_new[-1]}")
        # print(f"Last element in y_new: {y_new[-1]}")

        # print(f"Features Equal: {action == X_new[-1]}")
        # print(f"Target Equal: {y_label == y_new[-1]}")

        new_mse = -(cross_val_score(eval_model, X_new, y_new, cv=self.cv, scoring='neg_mean_squared_error', n_jobs=-1).mean())
        y_pred_test = cross_val_predict(eval_model, X_new, y_new, n_jobs=-1)
        # print(f"New MSE: {new_mse:.4f}")
        # entropy_mean = self._avg_entropy_cal(X_new)
        # obs = self._get_obs(new_mse, X_new, residual_new, y_new)
        newVendiScore = self.vendi_score(X_new)
        obs = self._get_observation(new_mse, X_new, y_new, residual_new, newVendiScore)
        info = {}
        info['mse'] = new_mse

        # --- reward ---
        if invalid:
            reward = -self.reward_clip
            orginal_reward = reward
        elif newVendiScore > self.vendiScore:
            # commit the improvement
            self.MSE = new_mse
            self.X = X_new
            self.y = y_new
            self.residual = residual_new
            log_r2 = self._log_r_score_cal(y_new, y_pred_test)
            reward = float(log_r2)
            orginal_reward = reward
            if residual_new[-1] < self.div_low or residual_new[-1] > self.div_high:
                reward += - 0.5
        else:
            reward = -0.01
            orginal_reward = reward
            if residual_new[-1] < self.div_low or residual_new[-1] > self.div_high:
                reward += - 0.005
        # --- termination vs truncation ---
        truncated = (len(self.X) >= int(1.2 * self.dataset_length)
                     or self.current_step > int(0.4 * self.dataset_length))
        terminated = False   # set True here if you define a natural goal state

        if self.verbose and reward > -1:
            print(f"| step {self.current_step} | action_sum {action.sum():.3f} "
                  f"| reward {reward:.4f} | original reward {orginal_reward:.4f} | mse {new_mse:.5f} | saved mse {self.MSE:.4f} | vendi {self.vendiScore:.4f} | new vendi {newVendiScore:.4f} | residual_new: {residual_new[-1]:.4f}")

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    def _get_observation(self, MSE: float, X: np.ndarray, y: np.ndarray, residual: np.ndarray, vendi: float) -> dict:
        """Construct the observation dictionary."""
        obs = {
                    "mse": np.array([MSE], dtype=np.float32),
                    "n_points": np.array([X.shape[0]], dtype=np.float32),
                    "vendi": np.array([vendi], dtype=np.float32),
                    "diversity": np.array([self._calculate_diversity(X)], dtype=np.float32),
                    "corr_res": np.array([np.corrcoef(residual, y)[0, 1]], dtype=np.float32) if len(residual) > 1 else np.array([0.0], dtype=np.float32),
                    "corr_feat_w_y": np.array([np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])], dtype=np.float32),
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

    def _corr(self, X, target):
        corr = pd.DataFrame(X).corrwith(pd.Series(np.asarray(target).ravel()))
        return np.nan_to_num(corr.to_numpy(), nan=0.0).astype(np.float32)
    
    def _calculate_diversity(self, X: np.ndarray) -> float:
            """Calculate the diversity of the dataset based on pairwise distances."""
            if X.shape[0] < 2:
                return 0.0
            pairwise_dist = pairwise_distances(X)
            diversity = np.mean(pairwise_dist)
            return diversity

    def vendi_score(self, X_samples: np.ndarray) -> float:
        n = X_samples.shape[0]

        # Kernel matrix K with k(x, x) = 1 on the diagonal
        K = cosine_similarity(X_samples, X_samples)

        # Eigenvalues of K / n (symmetric PSD -> real, non-negative)
        eigvals = np.linalg.eigvalsh(K / n)

        # Clamp tiny negative values from numerical error, drop zeros (0 log 0 = 0)
        eigvals = eigvals[eigvals > 1e-12]

        entropy = -np.sum(eigvals * np.log(eigvals))
        return np.exp(entropy)

    def _log_r_score_cal(self, y_pred, y_new):
        r2 = r2_score(y_new, y_pred)
        # guard against r2 -> 1 blowing up the log
        return float(np.log(1.0 / max(1.0 - r2, 1e-6)))

class EpisodeMSECallback(BaseCallback):
    """
    Logs the average MSE per episode to a CSV file.
    Columns: episode, avg_mse

    Works with vectorized envs: each sub-env accumulates independently,
    and a row is written whenever a sub-env's episode ends.
    """

    def __init__(self, csv_path: str = "episode_mse.csv", verbose: int = 0):
        super().__init__(verbose)
        self.csv_path = csv_path
        self.episode_idx = 0
        self._mse_sum = None      # per-env running sum
        self._mse_count = None    # per-env step count

    def _on_training_start(self) -> None:
        n_envs = self.training_env.num_envs
        self._mse_sum = np.zeros(n_envs, dtype=np.float64)
        self._mse_count = np.zeros(n_envs, dtype=np.int64)

        # Write header once (overwrite any existing file).
        with open(self.csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "avg_mse"])

    def _on_step(self) -> bool:
        # SB3 puts these in self.locals during the rollout.
        infos = self.locals["infos"]
        dones = self.locals["dones"]

        for i, (info, done) in enumerate(zip(infos, dones)):
            # Accumulate the per-step MSE for this sub-env.
            if "mse" in info:
                self._mse_sum[i] += float(info["mse"])
                self._mse_count[i] += 1

            if done:
                count = max(self._mse_count[i], 1)
                avg_mse = self._mse_sum[i] / count
                self._write_row(self.episode_idx, avg_mse)
                self.episode_idx += 1

                # Reset accumulators for this sub-env's next episode.
                self._mse_sum[i] = 0.0
                self._mse_count[i] = 0

        return True  # returning False would stop training

    def _write_row(self, episode, avg_mse):
        with open(self.csv_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([episode, float(avg_mse)])


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

    key_count = []
    for key in my_data['split_composition']:
        # print(f"Key: {key.keys()}")
        # print(f"Key Length: {len(key.keys())}")
        key_count.append(key.keys())
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

    gym.register(
        id="gymnasium_env/CRDADataGen-v1",
        entry_point=CRDADataGenEnvV2,
        # max_episode_steps=500,  # Prevent infinite episodes
    )

    set_seed(42)

    X = np.array(df_final.drop(columns="yield_strength"))
    y = np.log(np.array(df_final['yield_strength']))

    model = GPRegressor()
    model.fit(X, y)

    residual = y - model.predict(X)

    mean_scores=-cross_val_score(model, X, y, cv=10, scoring='neg_mean_squared_error', n_jobs=-1).mean()

    residual_mean = residual.mean()
    residual_std = residual.std()
    factor = pd.DataFrame(y).skew().to_numpy().item()
    div_low = factor*(residual_mean - residual_std).item()
    div_high = factor*(residual_mean + residual_std).item()

    env = gym.make("gymnasium_env/CRDADataGen-v1",
               X_train=X,
               y_train=y,
               model=model,
               residual=residual,
               MSE=mean_scores,
               div_low=div_low,
               div_high=div_high,
               normalize_actions=False,
               reward_clip=0.1,
               verbose=False)

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(1)
    try:
        check_env(env.unwrapped)
        print("Environment passes all checks!")
    except Exception as e:
        print(f"Environment has issues: {e}")

    log_dir ="./logs/"
    os.makedirs(log_dir, exist_ok=True)

    env = Monitor(env, filename=os.path.join(log_dir, f"monitor_3"))

    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=3e-4,
        batch_size=64,
        n_steps=2048,
        gamma=0.99,
        verbose=1,
        device='cuda'
    )

    mse_callback = EpisodeMSECallback(csv_path="episode_mse_3.csv", verbose=1)

    model.learn(
        total_timesteps=200000,
        callback=mse_callback
    )

    model.save("synthetic_generator_3")

    plot_reward("logs/monitor_3.monitor.csv", window=50)

if __name__ == '__main__':
    main()
