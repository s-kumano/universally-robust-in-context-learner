import torch
from torch import Tensor, nn


class Transformer(nn.Module):
    def __init__(self, d: int, efficient: bool = False) -> None:
        super().__init__()

        self.efficient = efficient

        if efficient:
            self.P = nn.Parameter(torch.full((1, d+1), .001))
            self.Q = nn.Parameter(torch.full((d+1, d), .001))
        else:
            self.P = nn.Parameter(torch.full((d+1, d+1), .001))
            self.Q = nn.Parameter(torch.full((d+1, d+1), .001))

    def forward(self, demos: Tensor, queries: Tensor) -> Tensor:
        return self.P @ demos @ demos.transpose(1, 2) @ self.Q @ queries / demos.shape[2] # (b, 1, N_queries) if efficient else (b, d+1, N_queries)