# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 16:32:39 2024

@author: kalsayed
"""

import numpy as np
import torch as T
from eplus_gym.agents.q_transformer.deep_q_network import DeepQNetwork
from eplus_gym.agents.q_transformer.replay_memory import ReplayBuffer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from eplus_gym.envs.energyplus import _find_project_root


class Q_transformer(object):
    def __init__(self, gamma, epsilon, lr, input_dims, n_actions,
                 mem_size, eps_min, batch_size, replace, eps_dec,
                 chkpt_dir, algo, env_name,
                 co2_aux_weight=0.1, co2_feature_indices=(14,)):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.algo = algo
        self.env_name = env_name
        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0
        self.co2_aux_weight = co2_aux_weight
        self.co2_feature_indices = tuple(co2_feature_indices)
        self.last_q_loss = None
        self.last_co2_aux_loss = None

        # Exponential epsilon decay
        self.epsilon_decay_episodes = (self.eps_min / max(self.eps_min, self.epsilon)) ** (1 / self.eps_dec)

        # -------------
        # 1) Replay Buffer
        # -------------
        # sac_max_size=1_000_000 (single-step), transformer_max_size=mem_size, e.g.
        self.memory = ReplayBuffer(sac_max_size=1000000,
                                   transformer_max_size=mem_size,
                                   input_shape=input_dims)

        # -------------
        # 2) Q networks (Transformer-based)
        # -------------
        self.q_eval = DeepQNetwork(
            lr=self.lr,
            n_actions=self.n_actions,
            name=self.env_name + '_' + self.algo + '_q_eval',
            input_dims=self.input_dims,
            chkpt_dir=self.chkpt_dir,
            co2_output_dim=len(self.co2_feature_indices),
            co2_feature_indices=self.co2_feature_indices
        )

        self.q_next = DeepQNetwork(
            lr=self.lr,
            n_actions=self.n_actions,
            name=self.env_name + '_' + self.algo + '_q_next',
            input_dims=self.input_dims,
            chkpt_dir=self.chkpt_dir,
            co2_output_dim=len(self.co2_feature_indices),
            co2_feature_indices=self.co2_feature_indices
        )

        # -------------
        # 3) Normalizer
        # -------------
        proj_root = _find_project_root()
        qtx_dir = (proj_root / "src" / "eplus_gym"  / "envs" / "assets" / "normalization" / "Q-transformer.csv")
        self.sample_data = pd.read_csv(qtx_dir).to_numpy()
        self.sample_data = self._match_normalization_features(self.sample_data, self.input_dims[0])
        self.scaler = MinMaxScaler()
        self.scaler.fit(self.sample_data)

    def _match_normalization_features(self, sample_data, target_dim):
        if sample_data.shape[1] == target_dim:
            return sample_data

        raise ValueError(
            f"Normalizer feature count mismatch: CSV has {sample_data.shape[1]} "
            f"features but environment has {target_dim}."
        )

    # -------------------------------------------------------
    # Action selection using single Q(s,a) => loop over actions
    # -------------------------------------------------------
    def choose_action_test(self, recent_96_states):
        """
        Given the last 96 time-steps of state features (shape (96,state_dim)),
        choose an action in discrete space 0..n_actions-1.
        """
        if np.random.random() < self.epsilon:
            # Exploration
            action = np.random.choice(self.action_space)
        else:
            # 1) Ensure the input is an np.array with shape (96, state_dim).
            obs_np = np.array(recent_96_states, dtype=np.float32)

            # 2) Scale with the same scaler used in training.
            #    Flatten to 2D for the scaler, then reshape back.
            obs_flat = obs_np.reshape(-1, obs_np.shape[-1])  # shape (96, state_dim)
            obs_flat = self.scaler.transform(obs_flat)       # scaled shape (96, state_dim)

            # 3) Reshape back and then expand batch dim => (1, 96, state_dim)
            obs_scaled = obs_flat.reshape(96, obs_np.shape[-1])
            S_3d = T.tensor(obs_scaled, dtype=T.float).unsqueeze(0).to(self.q_eval.device)

            # 4) Single forward pass => q_values of shape (1, n_actions)
            q_values = self.q_eval.forward(S_3d)  # shape (1, n_actions)

            # 5) Pick argmax over dimension 1
            action = T.argmax(q_values, dim=1).item()

        return action
     
    

    


    # Same store_transition, but it also populates the single-step buffer
    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    # (Optional) If you still want to sample from single-step buffer somewhere
    def sample_memory(self):
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
        # Normalization
        normalized_state = self.scaler.transform(state)
        normalized_new_state = self.scaler.transform(new_state)

        states = T.tensor(normalized_state).float().to(self.q_eval.device)
        actions = T.tensor(action).long().to(self.q_eval.device)
        rewards = T.tensor(reward).float().to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)
        states_ = T.tensor(normalized_new_state).float().to(self.q_eval.device)

        return states, actions, rewards, states_, dones

    # -------------------------------------------------------
    # The MAIN learn() using the Transformer buffer
    # -------------------------------------------------------
    def learn(self):
        # 1) Skip if not enough sequences
        max_t = min(self.memory.t_mem_cntr, self.memory.transformer_mem_size)
        if max_t < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()
        self.replace_target_network()

        # 2) Sample from transformer buffer
        seq_states, seq_actions, seq_next, seq_rewards, seq_dones = \
            self.memory.sample_transformer_buffer(self.batch_size)
        if seq_states is None:
            return

        B = seq_states.shape[0]
        seq_len = seq_states.shape[1]  # 96

        # Flatten + scale
        flat_S = seq_states.reshape(B*seq_len, -1)
        flat_Sn = seq_next.reshape(B*seq_len, -1)

        flat_S  = self.scaler.transform(flat_S)
        flat_Sn = self.scaler.transform(flat_Sn)
        norm_next_seq = flat_Sn.reshape(B, seq_len, -1)
        co2_targets = norm_next_seq[:, -1, self.co2_feature_indices]

        # Reshape back
        seq_states_t = T.tensor(flat_S.reshape(B, seq_len, -1),
                                dtype=T.float).to(self.q_eval.device)
        seq_next_t   = T.tensor(norm_next_seq,
                                dtype=T.float).to(self.q_eval.device)
        co2_targets_t = T.tensor(co2_targets, dtype=T.float).to(self.q_eval.device)

        # Convert actions, rewards, dones
        # seq_actions shape = (B,1) => we want a 1D LongTensor for gather
        actions = T.tensor(seq_actions, dtype=T.long).view(-1).to(self.q_eval.device)
        rewards = T.tensor(seq_rewards, dtype=T.float).to(self.q_eval.device)
        dones   = T.tensor(seq_dones, dtype=T.bool).to(self.q_eval.device)

        # 3) Current Q-values => shape (B, n_actions)
        q_pred_all, co2_pred = self.q_eval(seq_states_t, return_aux=True)

        # We gather the Q-value for the action that was actually taken
        q_pred = q_pred_all.gather(1, actions.unsqueeze(1)).squeeze(1)  # shape (B,)

        # 4) Next Q-values for double DQN
        # Evaluate next state with q_eval to get best actions
        with T.no_grad():
            q_eval_next_all = self.q_eval(seq_next_t)  # (B, n_actions)
            best_actions = q_eval_next_all.argmax(dim=1)  # shape (B,)

            # Evaluate next state with q_next to get target
            q_next_all = self.q_next(seq_next_t)  # (B, n_actions)

            # 5) Zero out any next Q-values if done
            q_next_all[dones] = 0.0

            # 6) The target
            indices = T.arange(B).long().to(self.q_eval.device)
            q_target = rewards + self.gamma * q_next_all[indices, best_actions]

        # 7) Loss and backprop
        q_loss = self.q_eval.loss(q_pred, q_target).to(self.q_eval.device)
        co2_aux_loss = self.q_eval.loss(co2_pred, co2_targets_t).to(self.q_eval.device)
        loss = q_loss + self.co2_aux_weight * co2_aux_loss
        loss.backward()
        self.q_eval.optimizer.step()

        self.last_q_loss = q_loss.item()
        self.last_co2_aux_loss = co2_aux_loss.item()

        self.learn_step_counter += 1
    
        

    # -------------------------------------------------------
    # Target network soft update
    # -------------------------------------------------------
    def replace_target_network(self):
        # Copy weights every 'replace_target_cnt' steps
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        # Exponential decay
        self.epsilon = max(self.eps_min, self.epsilon * self.epsilon_decay_episodes)

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

