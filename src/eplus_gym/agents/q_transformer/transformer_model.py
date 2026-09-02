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
                 co2_feature_indices = (14,)):
        super().__init__()
        self.co2_feature_indices = tuple(co2_feature_indices)
        self.register_buffer(
            "co2_feature_index_tensor",
            torch.tensor(self.co2_feature_indices, dtype=torch.long),
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
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, co2_output_dim),
        )
        self.co2_gate = nn.Sequential(
            nn.Linear(len(self.co2_feature_indices), d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )

        # (optional) layer-norm on encoder memory to stabilise training
        self.mem_norm = nn.LayerNorm(d_model)

        #self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_embed.weight)
        nn.init.constant_(self.in_embed.bias, 0.)
        nn.init.normal_(self.action_queries, std=0.02)

    def forward(self, S, return_aux=False):
        """
        S: (B, 96, input_dim)
        """
        B = S.size(0)

        # ── Encode sequence ────────────────────────────────────────────
        x   = self.in_embed(S)                  # (B, L, d)
        x   = self.pos_enc(x)
        mem = self.encoder(x)                   # (B, L, d)
        mem = self.mem_norm(mem)                # (B, L, d)
        co2_features = S.index_select(2, self.co2_feature_index_tensor)[:, -1, :]
        co2_gate = self.co2_gate(co2_features).unsqueeze(1)
        gated_mem = mem * (1.0 + co2_gate)

        # ── Prepare B copies of the action-query tokens ────────────────
        #    queries: (B, n_actions, d)
        queries = self.action_queries.unsqueeze(0).expand(B, -1, -1)

        # ── Decode ─────────────────────────────────────────────────────
        #    tgt_mask not required (no autoregression)
        dec_out = self.decoder(tgt=queries, memory=gated_mem)  # (B, n_actions, d)

        # ── Projection to scalar Qs ────────────────────────────────────
        q_vals = self.q_proj(dec_out).squeeze(-1)         # (B, n_actions)
        if return_aux:
            co2_pred = self.co2_head(mem[:, -1, :])
            return q_vals, co2_pred
        return q_vals
