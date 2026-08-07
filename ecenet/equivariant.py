"""SO(2)-equivariant layers operating on bond-frame angular features.

Once ACE features are Wigner-rotated into a bond frame (see model.py /
ace_basis.py), each angular mode ``m`` transforms as ``e^{imφ}`` under rotation
about the bond axis. Features are carried as cos/sin Fourier pairs
``(A_cos, A_sin)`` of shape ``(n_edges, n_features, n_angular)`` with
``n_angular = m_max + 1``. This module provides the layer types that act on that
representation while preserving the SO(2) structure:

- ``EquivariantLinear``: per-mode channel mixing (block-diagonal across ``m``),
  the same weights applied to the cos and sin parts, bias only on the ``m=0``
  (invariant) channel.
- ``RealSpaceNonlinearity``: applies a pointwise nonlinearity equivariantly via
  iDFT → σ → DFT on a θ-grid, coupling modes while staying SO(2)-equivariant.
"""

import numpy as np
import torch
import torch.nn as nn


class EquivariantLinear(nn.Module):
    """Block-diagonal linear layer preserving equivariance.

    Same weights for cos/sin parts. Bias only on m=0 (invariant).
    Angular channels: m = 0, 1, ..., m_max (index 0 is m=0).
    """

    def __init__(self, in_features, out_features, n_angular, m_max):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_angular = n_angular
        self.m_max = m_max

        # (n_angular, out_features, in_features)
        std = (2.0 / (in_features + out_features)) ** 0.5
        self.weights = nn.Parameter(torch.randn(n_angular, out_features, in_features) * std)

        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, A_cos, A_sin):
        A_cos_out = torch.einsum('...id,doi->...od', A_cos, self.weights)
        A_sin_out = torch.einsum('...id,doi->...od', A_sin, self.weights)

        # Bias only on m=0 (index 0)
        A_cos_out[..., 0] = A_cos_out[..., 0] + self.bias

        return A_cos_out, A_sin_out


class RealSpaceNonlinearity(nn.Module):
    """Nonlinear layer via real-space transform on the angular coordinate.

    Transforms Fourier coefficients (cos/sin parts for m=0,...,m_max) to
    function values on a uniform θ grid, applies a pointwise nonlinearity,
    and transforms back to Fourier space.

    This preserves equivariance because pointwise operations in angular
    space commute with rotation (θ → θ - φ).

    Args:
        n_features: number of feature channels
        m_max: maximum angular frequency
        n_grid: number of θ grid points (default: 4*m_max + 1)
        activation: pointwise nonlinearity ('silu', 'relu', 'tanh', 'gelu')
    """

    def __init__(self, n_features, m_max, n_grid=None, activation='silu'):
        super().__init__()
        self.n_features = n_features
        self.m_max = m_max
        self.n_angular = m_max + 1

        # Grid size: oversample to reduce aliasing from the nonlinearity.
        if n_grid is None:
            n_grid = 2 * (2 * m_max) + 1
        n_grid = int(n_grid)
        self.n_grid = n_grid

        # Uniform grid on [0, 2π)
        theta = torch.linspace(0, 2 * np.pi, n_grid + 1)[:-1]

        # Synthesis matrix: Fourier → grid
        # f(θ_k) = Σ_d [A_cos[m]*cos(m*θ_k) + A_sin[m]*sin(m*θ_k)]
        cos_synth = torch.stack([torch.cos(m * theta) for m in range(m_max + 1)], dim=0)
        sin_synth = torch.stack([torch.sin(m * theta) for m in range(m_max + 1)], dim=0)
        self.register_buffer('cos_synth', cos_synth)  # (n_angular, n_grid)
        self.register_buffer('sin_synth', sin_synth)  # (n_angular, n_grid)

        # Analysis matrix: grid → Fourier
        # A_cos[m] = norm[m] * Σ_k f(θ_k) * cos(m*θ_k)
        # norm[0] = 1/N, norm[m>0] = 2/N
        norm = torch.ones(m_max + 1) * (2.0 / n_grid)
        norm[0] = 1.0 / n_grid
        self.register_buffer('cos_analysis', cos_synth.T * norm.unsqueeze(0))  # (n_grid, n_angular)
        self.register_buffer('sin_analysis', sin_synth.T * norm.unsqueeze(0))  # (n_grid, n_angular)

        # No pre-activation affine: this is pure σ(f(θ)). Earlier versions kept
        # fixed scale=1 / shift=0 buffers and applied them in forward, which was
        # an exact identity — two elementwise passes over the full (n_edges,
        # n_features, n_grid) grid tensor for nothing. Removed; checkpoints that
        # still carry the buffers load fine (see calculator.from_checkpoint).

        # Nonlinearity
        act_map = {'silu': nn.SiLU, 'relu': nn.ReLU, 'tanh': nn.Tanh, 'gelu': nn.GELU}
        self.activation = act_map[activation]()

    def forward(self, A_cos, A_sin):
        """
        Args:
            A_cos, A_sin: (n_edges, n_features, n_angular)

        Returns:
            A_cos_out, A_sin_out: (n_edges, n_features, n_angular)
        """
        # Synthesis: Fourier coefficients → grid values
        f_grid = A_cos @ self.cos_synth + A_sin @ self.sin_synth

        # Apply nonlinearity
        f_grid = self.activation(f_grid)

        # Analysis: grid values → Fourier coefficients
        A_cos_out = f_grid @ self.cos_analysis
        A_sin_out = f_grid @ self.sin_analysis

        return A_cos_out, A_sin_out
