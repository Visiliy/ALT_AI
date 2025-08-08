import math
import torch
import torch.nn as nn
from typing import Tuple, List


class NoiseParamsTransformer(nn.Module):
    def __init__(self, initial_space_dim: int, time_embedding_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.input_proj = nn.Linear(3 * initial_space_dim + time_embedding_dim, hidden_dim)

        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=12, batch_first=False)
        self.gelu = nn.GELU()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.beta_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.eta_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        )

    def forward(self, initial_space: torch.Tensor[torch.Tensor[torch.FloatType]], finite_space: torch.Tensor[torch.Tensor[torch.FloatType]], time_embedding: torch.Tensor[torch.FloatType],
                history: torch.Tensor[torch.Tensor[torch.FloatType]]) -> Tuple[torch.Tensor[torch.FloatType], torch.Tensor[torch.FloatType]]:

        time_embedding = time_embedding.repeat(initial_space.shape[0], 1)
        combined = torch.cat([initial_space, finite_space, time_embedding, history], dim=-1)
        h = self.input_proj(combined)
        h = h.unsqueeze(1)

        h_attn, _ = self.attn(h.transpose(0, 1), h.transpose(0, 1), h.transpose(0, 1))
        h_attn = h_attn.transpose(0, 1)
        h = h + h_attn
        h = self.norm1(h)
        h = self.gelu(h)

        h_attn, _ = self.attn(h.transpose(0, 1), h.transpose(0, 1), h.transpose(0,1))
        h_attn = h_attn.transpose(0, 1)
        h = h + h_attn
        h = self.norm2(h)
        h = self.gelu(h)

        h = h + self.mlp(h)
        beta = self.beta_head(h)
        eta = self.eta_head(h)

        return beta.squeeze(-1), eta.squeeze(-1)


class SpaceConverter(nn.Module):
    def __init__(self, embedding_size: int, iterations: int = 100) -> None:
        super().__init__()
        self.iterations = iterations
        self.embedding_size = embedding_size

        self.time_embedding = nn.Embedding(iterations, embedding_size)
        self.transformer = NoiseParamsTransformer(embedding_size, embedding_size)

        self.loss_weights = {
            'target': 1.0,
            'reg': 0.01,
            'smooth': 0.05,
            'history': 0.1
        }

    def forward(
            self,
            initial_space: torch.Tensor[torch.Tensor[torch.FloatType]],
            finite_space: torch.Tensor[torch.Tensor[torch.FloatType]]
            ) -> torch.Tensor[torch.Tensor[torch.FloatType]]:
        history = initial_space

        for step in range(self.iterations):
            bete, eta = self.transformer(initial_space, finite_space, self.time_embedding, history)
