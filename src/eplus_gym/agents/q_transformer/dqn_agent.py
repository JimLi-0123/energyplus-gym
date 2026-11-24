# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 16:32:39 2024

@author: kalsayed
"""

import numpy as np
import torch as T
from eplus_gym.agents.q_transformer.deep_q_network import DeepQNetwork
from eplus_gym.agents.q_transformer.replay_memory import ReplayBuffer
from eplus_gym.agents.q_transformer.gradient_diagnostics import compare_loss_gradients, occupancy_prediction_metrics
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from eplus_gym.envs.energyplus import _find_project_root


class Q_transformer(object):
    def __init__(self, gamma, epsilon, lr, input_dims, n_actions,
                 mem_size, eps_min, batch_size, replace, eps_dec,
                 chkpt_dir, algo, env_name,
                 co2_aux_weight=0.1, co2_feature_indices=(14,),
                 use_gate=True, use_co2_gate=True, use_iat_gate=False,
                 co2_gate_weight=1.0, iat_gate_weight=0.0,
                 iat_feature_index=13,
                 iat_comfort_low=21.0, iat_comfort_high=24.0,
                 use_co2_aux=True, use_iat_aux=False, iat_aux_weight=0.1,
                 use_occ_aux=False, occ_aux_weight=0.1, occ_feature_index=7,
                 record_gradient_diagnostics=False, occ_predict_delta=False):
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
        self.iat_aux_weight = iat_aux_weight
        self.use_co2_aux = bool(use_co2_aux)
        self.use_iat_aux = bool(use_iat_aux)
        self.use_occ_aux = bool(use_occ_aux)
        self.occ_aux_weight = occ_aux_weight
        self.occ_feature_index = int(occ_feature_index)
        self.occ_predict_delta = bool(occ_predict_delta)
        if not 0 <= self.occ_feature_index < input_dims[0]:
            raise ValueError("occ_feature_index is outside the state feature range.")
        if co2_aux_weight < 0 or iat_aux_weight < 0 or occ_aux_weight < 0:
            raise ValueError("Auxiliary loss weights must be nonnegative.")
        self.co2_feature_indices = tuple(co2_feature_indices)
        self.use_gate = bool(use_gate)
        self.use_co2_gate = bool(use_co2_gate)
        self.use_iat_gate = bool(use_iat_gate)
        self.co2_gate_weight = float(co2_gate_weight)
        self.iat_gate_weight = float(iat_gate_weight)
        self.iat_feature_index = int(iat_feature_index)
        self.iat_comfort_low = float(iat_comfort_low)
        self.iat_comfort_high = float(iat_comfort_high)
        self.last_q_loss = None
        self.last_co2_aux_loss = None
        self.last_iat_aux_loss = None
        self.last_occ_aux_loss = None
        self.record_gradient_diagnostics = bool(record_gradient_diagnostics)
        self.last_gradient_diagnostics = None

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
        # 2) Normalizer
        # -------------
        proj_root = _find_project_root()
        qtx_dir = (proj_root / "src" / "eplus_gym"  / "envs" / "assets" / "normalization" / "Q-transformer.csv")
        self.sample_data = pd.read_csv(qtx_dir).to_numpy()
        self.sample_data = self._match_normalization_features(self.sample_data, self.input_dims[0])
        self.scaler = MinMaxScaler()
        self.scaler.fit(self.sample_data)
        self.iat_comfort_low_norm = self._normalize_feature_value(
            self.iat_feature_index,
            self.iat_comfort_low,
        )
        self.iat_comfort_high_norm = self._normalize_feature_value(
            self.iat_feature_index,
            self.iat_comfort_high,
        )

        # -------------
        # 3) Q networks (Transformer-based)
        # -------------
        self.q_eval = DeepQNetwork(
            lr=self.lr,
            n_actions=self.n_actions,
            name=self.env_name + '_' + self.algo + '_q_eval',
            input_dims=self.input_dims,
            chkpt_dir=self.chkpt_dir,
            co2_output_dim=len(self.co2_feature_indices),
            co2_feature_indices=self.co2_feature_indices,
            use_gate=self.use_gate,
            use_co2_gate=self.use_co2_gate,
            use_iat_gate=self.use_iat_gate,
            co2_gate_weight=self.co2_gate_weight,
            iat_gate_weight=self.iat_gate_weight,
            iat_feature_index=self.iat_feature_index,
            iat_comfort_low_norm=self.iat_comfort_low_norm,
            iat_comfort_high_norm=self.iat_comfort_high_norm,
            use_co2_aux=self.use_co2_aux and self.co2_aux_weight > 0,
            use_iat_aux=self.use_iat_aux and self.iat_aux_weight > 0,
            use_occ_aux=self.use_occ_aux and self.occ_aux_weight > 0,
        )

        self.q_next = DeepQNetwork(
            lr=self.lr,
            n_actions=self.n_actions,
            name=self.env_name + '_' + self.algo + '_q_next',
            input_dims=self.input_dims,
            chkpt_dir=self.chkpt_dir,
            co2_output_dim=len(self.co2_feature_indices),
            co2_feature_indices=self.co2_feature_indices,
            use_gate=self.use_gate,
            use_co2_gate=self.use_co2_gate,
            use_iat_gate=self.use_iat_gate,
            co2_gate_weight=self.co2_gate_weight,
            iat_gate_weight=self.iat_gate_weight,
            iat_feature_index=self.iat_feature_index,
            iat_comfort_low_norm=self.iat_comfort_low_norm,
            iat_comfort_high_norm=self.iat_comfort_high_norm,
            use_co2_aux=self.use_co2_aux and self.co2_aux_weight > 0,
            use_iat_aux=self.use_iat_aux and self.iat_aux_weight > 0,
            use_occ_aux=self.use_occ_aux and self.occ_aux_weight > 0,
        )

    def _match_normalization_features(self, sample_data, target_dim):
        if sample_data.shape[1] == target_dim:
            return sample_data

        raise ValueError(
            f"Normalizer feature count mismatch: CSV has {sample_data.shape[1]} "
            f"features but environment has {target_dim}."
        )

    def _normalize_feature_value(self, feature_index, raw_value):
        feature_min = float(self.scaler.data_min_[feature_index])
        feature_max = float(self.scaler.data_max_[feature_index])
        denom = feature_max - feature_min
        if denom == 0.0:
            return 0.0
        return (float(raw_value) - feature_min) / denom

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
        self.last_gradient_diagnostics = None
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
        iat_targets = norm_next_seq[:, -1, self.iat_feature_index:self.iat_feature_index + 1]

        # Reshape back
        seq_states_t = T.tensor(flat_S.reshape(B, seq_len, -1),
                                dtype=T.float).to(self.q_eval.device)
        seq_next_t   = T.tensor(norm_next_seq,
                                dtype=T.float).to(self.q_eval.device)
        co2_targets_t = T.tensor(co2_targets, dtype=T.float).to(self.q_eval.device)
        iat_targets_t = T.tensor(iat_targets, dtype=T.float).to(self.q_eval.device)

        # Convert actions, rewards, dones
        # seq_actions shape = (B,1) => we want a 1D LongTensor for gather
        actions = T.tensor(seq_actions, dtype=T.long).view(-1).to(self.q_eval.device)
        rewards = T.tensor(seq_rewards, dtype=T.float).to(self.q_eval.device)
        dones   = T.tensor(seq_dones, dtype=T.bool).to(self.q_eval.device)

        # 3) Current Q-values => shape (B, n_actions)
        use_co2_loss = self.use_co2_aux and self.co2_aux_weight > 0.0
        use_iat_loss = self.use_iat_aux and self.iat_aux_weight > 0.0
        use_occ_loss = self.use_occ_aux and self.occ_aux_weight > 0.0
        use_aux_loss = use_co2_loss or use_iat_loss or use_occ_loss
        if use_aux_loss:
            q_pred_all, predictions = self.q_eval(seq_states_t, return_aux=True, actions=actions)
        else:
            q_pred_all = self.q_eval(seq_states_t, return_aux=False)

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
        co2_aux_loss = q_loss.new_tensor(0.0)
        iat_aux_loss = q_loss.new_tensor(0.0)
        occ_aux_loss = q_loss.new_tensor(0.0)
        if use_co2_loss:
            co2_aux_loss = self.q_eval.loss(predictions["co2"], co2_targets_t)
        if use_iat_loss:
            iat_aux_loss = self.q_eval.loss(predictions["iat"], iat_targets_t)
        if use_occ_loss:
            occ_feature = slice(self.occ_feature_index, self.occ_feature_index + 1)
            current_occ = seq_states_t[:, -1, occ_feature]
            next_occ = seq_next_t[:, -1, occ_feature]
            occ_target = next_occ - current_occ if self.occ_predict_delta else next_occ
            occ_aux_loss = self.q_eval.loss(predictions["occ"], occ_target)
        weighted_aux_loss = (self.co2_aux_weight * co2_aux_loss
                             + self.iat_aux_weight * iat_aux_loss
                             + self.occ_aux_weight * occ_aux_loss)
        loss = q_loss + weighted_aux_loss
        if getattr(self, 'record_gradient_diagnostics', False) and use_aux_loss:
            model = self.q_eval.transformer
            # Only parameters shared by Q-learning and prediction: not gates/decoder/heads.
            parameters = [p for module in (model.in_embed, model.encoder, model.mem_norm)
                          for p in module.parameters()]
            self.last_gradient_diagnostics = compare_loss_gradients(q_loss, weighted_aux_loss, parameters)
            self.last_gradient_diagnostics['occ_encoder_grad_norm'] = (
                self.last_gradient_diagnostics['weighted_aux_encoder_grad_norm'] / self.occ_aux_weight
                if use_occ_loss and not use_co2_loss and not use_iat_loss else float('nan')
            )
            self.last_gradient_diagnostics.update({
                'learning_step': self.learn_step_counter + 1,
                'q_loss': q_loss.item(),
                'co2_aux_loss': co2_aux_loss.item(),
                'iat_aux_loss': iat_aux_loss.item(),
                'occ_aux_loss': occ_aux_loss.item(),
                'weighted_aux_loss': weighted_aux_loss.item(),
                'total_loss': loss.item(),
                'co2_aux_weight': self.co2_aux_weight,
                'iat_aux_weight': self.iat_aux_weight,
                'occ_aux_weight': self.occ_aux_weight,
            })
            if use_occ_loss:
                # Report both target modes on the same next-occupancy scale.
                prediction = predictions['occ'].detach()
                if self.occ_predict_delta:
                    prediction = current_occ + prediction
                self.last_gradient_diagnostics.update(occupancy_prediction_metrics(
                    prediction, next_occ, current_occ,
                ))
        loss.backward()
        self.q_eval.optimizer.step()

        self.last_q_loss = q_loss.item()
        self.last_co2_aux_loss = co2_aux_loss.item()
        self.last_iat_aux_loss = iat_aux_loss.item()
        self.last_occ_aux_loss = occ_aux_loss.item()

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

