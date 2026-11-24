# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 15:55:34 2025

@author: kalsayed
"""



import torch
import torch.nn as nn
from eplus_gym.agents.q_transformer.positional_encoding import PositionalEncoding          # as in your current code

class TransformerModel(nn.Module):
    """
    Encoder–decoder DDQN backbone.
        • Encoder:   processes the 96×18 state sequence
        • Decoder:   one query per discrete action (65)
                     returns one scalar Q per action
    Forward:
        S  : (B, 96, input_dim)
        out: (B, 65)   – Q-values
    """
    def __init__(self,
                 n_actions      = 65,
                 d_model        = 64,
                 nhead          = 4,
                 num_enc_layers = 3,
                 num_dec_layers = 2,
                 dim_ff         = 256,
                 max_len        = 96,
                 input_dim      = 18,
                 co2_output_dim = 1,
                 co2_feature_indices = (14,),
                 use_gate       = True,
                 use_co2_gate   = True,
                 use_iat_gate   = False,
                 co2_gate_weight = 1.0,
                 iat_gate_weight = 0.0,
                 iat_feature_index = 13,
                 iat_comfort_low_norm = 0.0,
                 iat_comfort_high_norm = 1.0,
                 use_co2_aux = True,
                 use_iat_aux = False,
                 use_occ_aux = False):
        super().__init__()
        self.co2_feature_indices = tuple(co2_feature_indices)
        self.use_co2_aux = bool(use_co2_aux)
        self.use_iat_aux = bool(use_iat_aux)
        self.use_occ_aux = bool(use_occ_aux)
        self.use_gate = bool(use_gate)
        self.use_co2_gate = bool(use_co2_gate)
        self.use_iat_gate = bool(use_iat_gate)
        self.co2_gate_weight = float(co2_gate_weight)
        self.iat_gate_weight = float(iat_gate_weight)
        self.register_buffer(
            "co2_feature_index_tensor",
            torch.tensor(self.co2_feature_indices, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "iat_feature_index_tensor",
            torch.tensor([iat_feature_index], dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "iat_comfort_low_norm",
            torch.tensor(float(iat_comfort_low_norm), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "iat_comfort_high_norm",
            torch.tensor(float(iat_comfort_high_norm), dtype=torch.float32),
            persistent=False,
        )

        # ───── Encoder ────────────────────────────────────────────────────
        self.in_embed  = nn.Linear(input_dim, d_model)
        self.pos_enc   = PositionalEncoding(d_model, max_len=max_len)

        enc_layer  = nn.TransformerEncoderLayer(d_model, nhead,
                                                dim_feedforward=dim_ff,
                                                dropout=0,
                                                batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_enc_layers)

        # ───── Decoder ────────────────────────────────────────────────────
        # learnable action-query tokens: (n_actions, d_model)
        self.action_queries = nn.Parameter(torch.randn(n_actions, d_model))

        dec_layer  = nn.TransformerDecoderLayer(d_model, nhead,
                                                dim_feedforward=dim_ff,
                                                dropout=0,
                                                batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_dec_layers)

        # project each decoded token → scalar Q
        self.q_proj = nn.Linear(d_model, 1, bias=False)     # shared projection
        self.co2_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, co2_output_dim),
        )
        self.co2_gate = nn.Sequential(
            nn.Linear(len(self.co2_feature_indices), d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        self.iat_gate = nn.Sequential(
            nn.Linear(2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )

        # (optional) layer-norm on encoder memory to stabilise training
        self.mem_norm = nn.LayerNorm(d_model)
        self.aux_action_embedding = nn.Embedding(n_actions, d_model)
        self.iat_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )
        self.occ_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

        #self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_embed.weight)
        nn.init.constant_(self.in_embed.bias, 0.)
        nn.init.normal_(self.action_queries, std=0.02)

    def forward(self, S, return_aux=False, actions=None):
        """
        S: (B, 96, input_dim)
        actions: executed discrete action IDs at the final timestep, shape (B,) or (B, 1).
        return_aux=True returns (Q-values, {"co2": ..., "iat": ..., "occ": ...});
        only enabled prediction heads are evaluated, with normalized next-state targets.
        """
        B = S.size(0)

        # ── Encode sequence ────────────────────────────────────────────
        x   = self.in_embed(S)                  # (B, L, d)
        x   = self.pos_enc(x)
        mem = self.encoder(x)                   # (B, L, d)
        mem = self.mem_norm(mem)                # (B, L, d)
        gate_terms = []
        if self.use_gate and self.use_co2_gate and self.co2_gate_weight != 0.0:
            co2_features = S.index_select(2, self.co2_feature_index_tensor)[:, -1, :]
            co2_gate = self.co2_gate(co2_features)
            gate_terms.append(self.co2_gate_weight * co2_gate)

        if self.use_gate and self.use_iat_gate and self.iat_gate_weight != 0.0:
            iat_feature = S.index_select(2, self.iat_feature_index_tensor)[:, -1, :]
            iat_low_violation = torch.relu(self.iat_comfort_low_norm - iat_feature)
            iat_high_violation = torch.relu(iat_feature - self.iat_comfort_high_norm)
            iat_gate_features = torch.cat([iat_low_violation, iat_high_violation], dim=1)
            iat_gate = self.iat_gate(iat_gate_features)
            gate_terms.append(self.iat_gate_weight * iat_gate)

        if gate_terms:
            combined_gate = sum(gate_terms).unsqueeze(1)
            decoder_memory = mem * (1.0 + combined_gate)
        else:
            decoder_memory = mem

        # ── Prepare B copies of the action-query tokens ────────────────
        #    queries: (B, n_actions, d)
        queries = self.action_queries.unsqueeze(0).expand(B, -1, -1)

        # ── Decode ─────────────────────────────────────────────────────
        #    tgt_mask not required (no autoregression)
        dec_out = self.decoder(tgt=queries, memory=decoder_memory)  # (B, n_actions, d)

        # ── Projection to scalar Qs ────────────────────────────────────
        q_vals = self.q_proj(dec_out).squeeze(-1)         # (B, n_actions)
        if return_aux:
            predictions = {}
            if self.use_co2_aux or self.use_iat_aux:
                if actions is None:
                    raise ValueError("Auxiliary prediction requires executed actions a(t).")
                if actions.numel() != B or actions.dtype != torch.long:
                    raise ValueError("actions must contain one torch.long action ID per sample.")
                action_features = self.aux_action_embedding(actions.reshape(B))
                aux_features = torch.cat([mem[:, -1, :], action_features], dim=-1)
                if self.use_co2_aux:
                    predictions["co2"] = self.co2_head(aux_features)
                if self.use_iat_aux:
                    predictions["iat"] = self.iat_head(aux_features)
            if self.use_occ_aux:
                predictions["occ"] = self.occ_head(mem[:, -1, :])
            return q_vals, predictions
        return q_vals
