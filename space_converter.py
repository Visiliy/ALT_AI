import torch
import torch.nn as nn


class SpaceConverter(nn.Module):

    def __init__(self, embedding_size: int, iterations: int) -> None:
        super().__init__()
        self.iterations = iterations
        self.embedding_size = embedding_size
        self.time_embedding = nn.Embedding(self.iterations, self.embedding_size)

    def forward(
            self, 
            initial_space: torch.Tensor[torch.Tensor[torch.FloatType]],
            finite_space: torch.Tensor[torch.Tensor[torch.FloatType]]
            ) -> torch.Tensor[torch.Tensor[torch.FloatType]]:
        for iter in range(self.iterations):
            pass