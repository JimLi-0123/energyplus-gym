# -*- coding: utf-8 -*-
"""
Created on Wed Jan  7 07:16:32 2026

@author: khali
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan  7 07:09:02 2026

@author: khali
"""
#%% library part

# Standard library
import os
import pickle
from collections import deque
from pathlib import Path

# Third-party
import numpy as np
import matplotlib.pyplot as plt

# Local project modules
from eplus_gym.envs.env import AmphitheaterEnv
from eplus_gym.envs.energyplus import (
    _resolve_output_dir)
from eplus_gym.agents.q_transformer.dqn_agent import Q_transformer


# -----------------------------------------------------------------------------
# Helper utilities                                                            |
# -----------------------------------------------------------------------------

def _ensure_dir(path: str):
    """Create parent directory if it does not yet exist."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

# -------- training -----------------------------------------------------------

def save_training_metrics(filepath: str, scores, eps_history, steps_array):
    """Persist training arrays to *filepath* (Pickle)."""
    _ensure_dir(filepath)
    with open(filepath, "wb") as f:
        pickle.dump({
            "scores": scores,
            "eps_history": eps_history,
            "steps_array": steps_array,
        }, f)

# 1. Get the absolute path to the main directory 
#    This will be /.../energyplus-gym/    
output_dir = Path(_resolve_output_dir(None))
# 2. recreate gitkeep
output_root = Path(_resolve_output_dir(None))  # your resolver
(output_root / ".gitkeep").touch(exist_ok=True)

checkpoint_dir = _resolve_output_dir(None, default_name="runs/checkpoints")

ALGO_NAME = "CO2-gated-aux-Q-transformer"

CONTROL_PENALTY_CONFIG = {
    "enable_control_penalty": False,
    "w_switch": 0.3,
    "w_du_sat": 0.04,
    "w_du_oac": 0.04,
    "enable_iaq_aware_ventilation": False,
    "iaq_co2_threshold": 850.0,
    "iaq_switch_penalty_scale": 0.5,
    "iaq_oac_penalty_scale": 0.35,
    "iaq_oac_energy_scale": 0.4,
    "iaq_ahu_energy_scale": 0.35,
    "iaq_target_oac_fraction": 0.4,
    "iaq_high_co2_threshold": 950.0,
    "iaq_high_target_oac_fraction": 0.65,
    "w_iaq_ventilation_deficit": 1.0,
    "iaq_co2_severity_threshold": 1000.0,
    "w_iaq_co2_severity": 0.8,
}

#%% Adaptation test (Q-transformer)--------------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    
    
    
    env = AmphitheaterEnv(
        env_config={"output": str(output_dir), "location": "luxembourg", **CONTROL_PENALTY_CONFIG}, new_begin_month=1, new_end_month=12, new_begin_day=1, new_end_day=31,
        train=True)
    env.runner_config.csv = False

    best_score = -np.inf
    load_checkpoint = False
    total_deployment_load_checkpoint = False
    # n_episodes = 50
    n_episodes = 183
    


    agent = Q_transformer(
        gamma=0.99,
        epsilon=1,
        lr=0.001,
        input_dims=(env.observation_space.shape),
        n_actions=env.action_space.n,
        mem_size=250000,
        eps_min=0.05,
        batch_size=64,
        replace=384,
        eps_dec=0.9 * n_episodes,
        chkpt_dir=checkpoint_dir,
        algo=ALGO_NAME,
        env_name='energyplus'
    )

    agent.q_eval.train()
    # if load_checkpoint:
    #     agent.load_models()

    n_steps = 0
    Episode = 0
    scores, eps_history, steps_array = [], [], []
    terminated = True
    
    # A rolling buffer to hold up to 96 recent states
    state_window = deque(maxlen=96)

    for i in range(n_episodes):
        Episode += 1
        done = False

        # If the episode just terminated, reset env
        
        # if terminated:  # in case of generalisation test
        if terminated :
            env.close()
            observation = env.reset()[0]  # or env.reset()
            #state_window.clear()
            state_window.append(observation)

        score = 0

        while not done:
            # If we don't yet have 96 states, fallback to random action (or old approach)
            if len(state_window) < 96 :
                action = np.random.choice(agent.action_space)
            else:
                # Convert the rolling window to a numpy array of shape (96, n_features)
                obs_seq = np.array(state_window, dtype=np.float32)
                # Use the agent's method that expects a 96-step sequence
                action = agent.choose_action_test(obs_seq)

            # Step the environment
            observation_, reward, terminated, truncated, _ = env.step(action)
            score += reward
            done = terminated or truncated

            # Store the transition (single-step) in replay buffer
            # The replay buffer will internally assemble 96-step sequences for training
            
            agent.store_transition(observation, action, reward, observation_, done)
            agent.learn()

            # Move on
            observation = observation_
            state_window.append(observation)
            n_steps += 1

        # End of episode
        agent.decrement_epsilon()
        scores.append(score)
        steps_array.append(n_steps)

        avg_score = np.mean(scores[-100:])
        print('Episode:', Episode,
              'Score:', score,
              'Average score: %.1f' % avg_score,
              'Best score: %.2f' % best_score,
              'Epsilon: %.2f' % agent.epsilon,
              'Steps:', n_steps)

        if avg_score > best_score:
            if not total_deployment_load_checkpoint:
                agent.save_models()
            best_score = avg_score

        eps_history.append(agent.epsilon)

    env.close()
    agent.save_models()
    
    metrics_path = Path(_resolve_output_dir(None, default_name="runs/metrics")) / "generalisation_co2_gated_aux_q_transformer_metrics.pkl"
    save_training_metrics(str(metrics_path), scores, eps_history, steps_array)
    print(f"[Saved metrics] {metrics_path}")
    # -----------------------------------------------------------------
    # Plotting as in your original script
    # -----------------------------------------------------------------
    x = [i+1 for i in range(len(scores))]
    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score', color=color)
    ax1.plot(x, scores, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # Moving Average (Mean of every n_episodes)
    means = [np.mean(scores[i:i+n_episodes])
             for i in range(0, len(scores), n_episodes)]
    x_means = [i+n_episodes for i in range(0, len(scores), n_episodes)]
    ax1.plot(x_means, means, label='Mean Score per chunk',
             color='black', linestyle=':')
    ax1.legend(loc='upper left')


    # Cumulative Average
    cumulative_avg = np.cumsum(scores) / np.arange(1, len(scores)+1)
    ax1.plot(x, cumulative_avg, color='red',
             label='Cumulative Avg', linestyle='--')
    ax1.legend(loc='upper left')

    # Create a second y-axis for epsilon
    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('Epsilon', color=color)
    ax2.plot(x, eps_history, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Scores and Epsilon History Over Episodes')
    fig.tight_layout()

    plot_path = Path(_resolve_output_dir(None, default_name="runs/plots")) / "generalisation_co2_gated_aux_q_transformer_scores_epsilon.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")

    plt.close(fig)
    print(f"[Saved plot] {plot_path}")
    
