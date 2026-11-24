# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 15:16:56 2024

@author: kalsayed
"""
        
import os
import torch as T
import torch.nn as nn
import torch.optim as optim

# Import our Transformer-based Q model
from eplus_gym.agents.q_transformer.transformer_model import TransformerModel

class DeepQNetwork(nn.Module):
    def __init__(
        self,
        lr,
        n_actions,
        name,
        input_dims,
        chkpt_dir,
        d_model=32,
        nhead=4,
        num_layers=2,
        max_len=96,
        co2_output_dim=1,
        co2_feature_indices=(14,),
        use_gate=True,
        use_co2_gate=True,
        use_iat_gate=False,
        co2_gate_weight=1.0,
        iat_gate_weight=0.0,
        iat_feature_index=13,
        iat_comfort_low_norm=0.0,
        iat_comfort_high_norm=1.0,
        use_co2_aux=True,
        use_iat_aux=False,
        use_occ_aux=False,
    ):
        """
        A Transformer-based DQN that, given a state sequence, outputs Q-values
        for all actions.

        :param lr: learning rate
        :param n_actions: number of discrete actions
        :param name: checkpoint filename
        :param input_dims: not strictly used in the Transformer directly,
                           but kept for potential checks or reference
        :param chkpt_dir: directory to store checkpoints
        :param d_model, nhead, num_layers, max_len: hyperparams for Transformer
        """
        super(DeepQNetwork, self).__init__()

        self.n_actions = n_actions
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        input_dim = input_dims[0] if len(input_dims) == 1 else input_dims[-1]

        # ----------------------------------------------------
        # 1) Create the Transformer Q-Model
        # ----------------------------------------------------
        self.transformer = TransformerModel(
            d_model=d_model,
            nhead=nhead,
            max_len=max_len,
            n_actions=n_actions,  # so the model knows how many Q-values to output
            input_dim=input_dim,
            co2_output_dim=co2_output_dim,
            co2_feature_indices=co2_feature_indices,
            use_gate=use_gate,
            use_co2_gate=use_co2_gate,
            use_iat_gate=use_iat_gate,
            co2_gate_weight=co2_gate_weight,
            iat_gate_weight=iat_gate_weight,
            iat_feature_index=iat_feature_index,
            iat_comfort_low_norm=iat_comfort_low_norm,
            iat_comfort_high_norm=iat_comfort_high_norm,
            use_co2_aux=use_co2_aux,
            use_iat_aux=use_iat_aux,
            use_occ_aux=use_occ_aux,
        )
        
        # ----------------------------------------------------
        # 2) Optimizer and Loss
        # ----------------------------------------------------
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state_seq, return_aux=False, actions=None):
        """
        Forward pass to compute Q-values for all actions.

        :param state_seq: Tensor of shape (B, 96, state_dim)
        :param actions: Executed action IDs, required for enabled auxiliary predictions.
        :return: Tensor of shape (B, n_actions), the predicted Q-values
                 or (Q-values, auxiliary prediction dict) when return_aux=True.
        """
        q_values = self.transformer(state_seq, return_aux=return_aux, actions=actions)
        return q_values

    def save_checkpoint(self):
        print('... saving Transformer Q-network checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading Transformer Q-network checkpoint ...')
        state = T.load(self.checkpoint_file, weights_only=True, map_location=self.device)
        key = "transformer.co2_head.0.weight"
        expected = self.state_dict()[key]
        if key in state and state[key].shape == (expected.shape[0], expected.shape[1] // 2):
            # Legacy heads used memory alone; zero the new action columns.
            state[key] = T.cat([state[key], T.zeros_like(state[key])], dim=1)
            print("... migrated legacy CO2 head; action conditioning requires training ...")
        self.load_state_dict(state, strict=False)

        

        
