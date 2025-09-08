import torch
import torch.nn as nn
from .constants import SCORE_FN_TYPE


class AttentionScoreFunc(nn.Module):
    """
    NAIS: Neural Attentive Item Similarity Model for Recommendation
    He et al., 2018
    """
    def __init__(
        self,
        dim: int,
        score_fn_type: SCORE_FN_TYPE='hadamard',
        dropout: float=0.2,
    ):
        super().__init__()

        self.dim = dim
        self.score_fn_type = score_fn_type
        self.dropout = dropout

        self._init_layers()

    def forward(self, Q, K):
        # Q: (B, K, D)
        # K: (B, K, D)

        if self.score_fn_type == 'concat':
            return self._concat(Q, K)

        elif self.score_fn_type == 'hadamard':
            return self._hadamard(Q, K)

    def _concat(self, Q, K):
        # Q: (B, K, D)
        # K: (B, K, D)
        QK_cat = torch.cat([Q, K], dim=-1)  # (B, K, 2D)
        scores = self.mlp(QK_cat).squeeze(-1)  # (B, K)
        return scores
    
    def _hadamard(self, Q, K):
        # Q: (B, K, D)
        # K: (B, K, D)
        QK_hadamard = Q * K  # (B, K, D)
        scores = self.mlp(QK_hadamard).squeeze(-1)  # (B, K)
        return scores

    def _init_layers(self):
        if self.score_fn_type == 'concat':
            self.mlp = nn.Sequential(
                nn.Linear(self.dim * 2, self.dim),
                nn.LayerNorm(self.dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),

                nn.Linear(self.dim, 1),
            )
        elif self.score_fn_type == 'hadamard':
            self.mlp = nn.Sequential(
                nn.Linear(self.dim, self.dim),
                nn.LayerNorm(self.dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),

                nn.Linear(self.dim, 1),
            )
        else:
            raise ValueError("score_fn_type must be `concat` or `hadamard`")