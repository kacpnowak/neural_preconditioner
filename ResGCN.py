import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


def scale_A_by_spectral_radius(A):
    if A.layout == torch.sparse_csc:

        absA = torch.absolute(A)
        m, n = absA.shape
        row_sum = absA @ torch.ones(n, 1, dtype=A.dtype, device=A.device)
        col_sum = torch.ones(1, m, dtype=A.dtype, device=A.device) @ absA
        gamma = torch.min(torch.max(row_sum), torch.max(col_sum))
        outA = A * (1. / gamma.item())

    elif A.layout == torch.strided:

        absA = torch.absolute(A)
        row_sum = torch.sum(absA, dim=1)
        col_sum = torch.sum(absA, dim=0)
        gamma = torch.min(torch.max(row_sum), torch.max(col_sum))
        outA = A / gamma

    else:

        raise NotImplementedError(
            'A must be either torch.sparse_csc_tensor or torch.tensor')

    return outA


# -----------------------------------------------------------------------------
# An MLP layer.
class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, num_layers, hidden, drop_rate,
                 use_batchnorm=False, is_output_layer=False, dtype=torch.float64):
        super().__init__()
        self.num_layers = num_layers
        self.use_batchnorm = use_batchnorm
        self.is_output_layer = is_output_layer

        self.lin = nn.ModuleList()
        self.lin.append(nn.Linear(in_dim, hidden, dtype=dtype))
        for i in range(1, num_layers - 1):
            self.lin.append(nn.Linear(hidden, hidden, dtype=dtype))
        self.lin.append(nn.Linear(hidden, out_dim, dtype=dtype))
        if use_batchnorm:
            self.batchnorm = nn.ModuleList()
            for i in range(0, num_layers - 1):
                self.batchnorm.append(nn.BatchNorm1d(hidden, dtype=dtype))
            if not is_output_layer:
                self.batchnorm.append(nn.BatchNorm1d(out_dim, dtype=dtype))
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, R):  # R: (*, in_dim)
        assert len(R.shape) >= 2
        for i in range(self.num_layers):
            R = self.lin[i](R)  # (*, hidden)
            if i != self.num_layers - 1 or not self.is_output_layer:
                if self.use_batchnorm:
                    shape = R.shape
                    R = R.view(-1, shape[-1])
                    R = self.batchnorm[i](R)
                    R = R.view(shape)
                R = self.dropout(F.relu(R))
                # (*, out_dim)
        return R


# -----------------------------------------------------------------------------
# A GCN layer.
class GCNConv(nn.Module):

    def __init__(self, AA, in_dim, out_dim, dtype=torch.float64):
        super().__init__()
        self.AA = AA  # normalized A
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.fc = nn.Linear(in_dim, out_dim, dtype=dtype)

    def forward(self, R):  # R: (n, batch_size, in_dim)
        assert len(R.shape) == 3
        n, batch_size, in_dim = R.shape
        assert in_dim == self.in_dim
        if in_dim > self.out_dim:
            R = self.fc(R)  # (n, batch_size, out_dim)
            R = R.view(n, batch_size * self.out_dim)  # (n, batch_size * out_dim)
            R = self.AA @ R  # (n, batch_size * out_dim)
            R = R.view(n, batch_size, self.out_dim)  # (n, batch_size, out_dim)
        else:
            R = R.view(n, batch_size * in_dim)  # (n, batch_size * in_dim)
            R = self.AA @ R  # (n, batch_size * in_dim)
            R = R.view(n, batch_size, in_dim)  # (n, batch_size, in_dim)
            R = self.fc(R)  # (n, batch_size, out_dim)
        return R


# -----------------------------------------------------------------------------
# GCN with residual connections.
class ResGCN(nn.Module):

    def __init__(self, A, num_layers, embed, hidden, drop_rate,
                 scale_input=True, dtype=torch.float64):
        # A: float64, already on device.
        #
        # For graph convolution, A will be normalized and cast to
        # lower precision and named AA.

        super().__init__()
        self.dtype = dtype  # used by GNP.precond.GNP
        self.num_layers = num_layers
        self.embed = embed
        self.scale_input = scale_input

        # Note: scale_A_by_spectral_radius() has been called when
        # defining the problem; hence, it is redundant. We keep the
        # code here to leave open the possibility of normalizing A in
        # another manner.
        self.AA = A.to(dtype)

        self.mlp_initial = MLP(1, embed, 4, hidden, drop_rate)
        self.mlp_final = MLP(embed, 1, 4, hidden, drop_rate,
                             is_output_layer=True, dtype=dtype)
        self.gconv = nn.ModuleList()
        self.skip = nn.ModuleList()
        self.batchnorm = nn.ModuleList()
        for i in range(num_layers):
            self.gconv.append(GCNConv(self.AA, embed, embed, dtype=dtype))
            self.skip.append(nn.Linear(embed, embed, dtype=dtype))
            self.batchnorm.append(nn.BatchNorm1d(embed, dtype=dtype))
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, r):  # r: (n, batch_size)
        assert len(r.shape) == 2
        n, batch_size = r.shape
        if self.scale_input:
            scaling = torch.linalg.vector_norm(r, dim=0) / np.sqrt(n)
            r = r / scaling  # scaling
        r = r.view(n, batch_size, 1)  # (n, batch_size, 1)
        R = self.mlp_initial(r)  # (n, batch_size, embed)

        for i in range(self.num_layers):
            R = self.gconv[i](R) + self.skip[i](R)  # (n, batch_size, embed)
            R = R.view(n * batch_size, self.embed)  # (n * batch_size, embed)
            R = self.batchnorm[i](R)  # (n * batch_size, embed)
            R = R.view(n, batch_size, self.embed)  # (n, batch_size, embed)
            R = self.dropout(F.relu(R))  # (n, batch_size, embed)

        z = self.mlp_final(R)  # (n, batch_size, 1)
        z = z.view(n, batch_size)  # (n, batch_size)
        if self.scale_input:
            z = z * scaling  # scaling back
        return z