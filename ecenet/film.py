"""Element- and distance-conditioned FiLM gate.

A small MLP on ``concat[embed(type_i), embed(type_j), φ(r_ij)]`` produces a
per-feature FiLM modulation: a scale ``γ`` and (optionally) a shift ``β``,
applied downstream as ``γ ⊙ x + β``.

Adapted from the template-mix 'mlp' coefficient gate (the old ElementCoeff):
the *generator* is identical — per-element embedding tables ``a[z_i]`` / ``b[z_j]``
concatenated with a radial basis of the bond distance, fed to a SiLU MLP — but
the output modulates features directly instead of mixing weight templates.
"""

import torch
import torch.nn as nn


class ElementFiLM(nn.Module):
    """Per-edge FiLM scalings from the source/target element types and
    (optionally) a radial basis of the bond distance.

    Args:
        n_features: width of the feature vector being modulated (γ, β are this wide)
        n_types:    number of element types
        embed_dim:  width P of each element embedding (default 16)
        n_rbf:      size of the radial basis φ(r) leg (0 → element-only gate)
        hidden:     gate-MLP hidden width(s): int, list/tuple, or None
                    (→ ``[max(2*n_features, 32)]``)
        shift:      if True predict a shift β too (full FiLM), as an extra head on
                    the *same* MLP (the last layer just gets n_features wider — no
                    second network). β is emitted for the **m=0 slot only**, so it
                    is always (E, n_features) regardless of n_modes: m=0 is the
                    rotation-invariant mode, and it is the only slot an additive
                    shift may touch without breaking equivariance (see the note
                    below). Caller applies it to A_cos[..., 0].
        base:       additive base of the scale, γ = base + Δ. Δ is zero at init
                    (zero-init last layer), so base sets the init value of γ:
                    1.0 (default) → identity gate (γ=1, no-op multiply);
                    0.0 → zero gate (γ=0, fully gates the feature off at init —
                    e.g. a residual message that opens up as it learns).
        n_modes:    number of angular modes M to emit a *separate* scale for.
                    1 (default) → one scale per channel, shape (E, n_features),
                    which the caller broadcasts over m. M > 1 → a scale per
                    ``(channel, m)``, shape (E, n_features, M). The MLP's output
                    width grows by M accordingly (γ, β are ``n_features*M`` wide).
        mode_valid: optional (n_features, n_modes) 0/1 mask of the *real* modes.
                    Δ (and β) are zeroed on the masked-out slots, so γ = base
                    there — the gate leaves them exactly as it found them. Used
                    to skip the structural-zero slots of the angular layout
                    (m > l of that channel), which can carry rotation-inconsistent
                    values that a per-mode scale must not touch differentially.

    forward(type_i, type_j, r_basis=None) → (gamma, beta)
        gamma: (E, n_features) or (E, n_features, M)  = base + Δ  (= base at init)
        beta:  (E, n_features), the m=0 shift        = 0 at init (None when shift=False)

    Note on equivariance: in the bond frame the residual symmetry is a per-mode
    SO(2) that rotates the pair ``(A_cos_m, A_sin_m)`` by ``mφ``. Any scale
    commutes with that rotation *provided cos and sin of the same ``(channel, m)``
    share it* — so a per-``(channel, m)`` scale (``n_modes = m_max+1``) is just as
    equivariant as a per-channel one, as long as it is applied to A_cos and A_sin
    alike. An additive *shift* does **not** commute with that rotation: for m > 0 it
    would break equivariance outright. m=0 is the exception — it is the invariant
    mode (e^{i·0·φ} = 1), so adding an invariant scalar (β depends only on the types
    and the distance) leaves the transformation intact. Hence β is m=0-only, and it
    belongs on A_cos[..., 0]: the m=0 *sin* slot is a structural zero (sin(0·φ) = 0)
    and must stay zero.
    """

    def __init__(self, n_features, n_types, embed_dim=16, n_rbf=0,
                 hidden=None, shift=True, base=1.0, n_modes=1, mode_valid=None):
        super().__init__()
        self.n_features = int(n_features)
        self.n_rbf = int(n_rbf or 0)
        self.n_modes = max(1, int(n_modes))
        self.shift = shift
        self.base = float(base)
        P = int(embed_dim)
        in_dim = 2 * P + self.n_rbf
        # One scale per (channel, mode) — n_modes == 1 → per channel — plus, when
        # shift=True, an extra head of n_features for the m=0 shift. Both come out
        # of the same MLP: the last layer is just wider.
        self.n_scale = self.n_features * self.n_modes
        out_dim = self.n_scale + (self.n_features if shift else 0)
        if hidden is None:
            hidden = [max(2 * self.n_features, 32)]
        elif isinstance(hidden, (list, tuple)):
            hidden = [int(h) for h in hidden]
        else:
            hidden = [int(hidden)]

        self.a = nn.Embedding(n_types, P)   # source / centre element code
        self.b = nn.Embedding(n_types, P)   # target / neighbour element code

        if mode_valid is not None:
            mode_valid = torch.as_tensor(mode_valid, dtype=torch.get_default_dtype())
            if tuple(mode_valid.shape) != (self.n_features, self.n_modes):
                raise ValueError(
                    f"mode_valid must have shape ({self.n_features}, {self.n_modes}), "
                    f"got {tuple(mode_valid.shape)}.")
            self.register_buffer('mode_valid', mode_valid)
        else:
            self.mode_valid = None

        dims = [in_dim] + hidden + [out_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.SiLU())
        self.mlp = nn.Sequential(*layers)
        # Zero-init the last layer → γ = 1, β = 0 at init (identity FiLM, so the
        # model starts unchanged and the gate learns from zero).
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def _shape(self, x):
        """(E, n_features*n_modes) → (E, n_features[, n_modes]), zeroed on the
        structural-zero slots so γ = base / β = 0 there."""
        if self.n_modes > 1:
            x = x.view(x.shape[0], self.n_features, self.n_modes)
        if self.mode_valid is not None:
            x = x * self.mode_valid
        return x

    def forward(self, type_i, type_j, r_basis=None):
        feats = [self.a(type_i), self.b(type_j)]          # (E, P) each
        if self.n_rbf:
            if r_basis is None:
                raise ValueError("ElementFiLM built with a radial leg (n_rbf>0) "
                                 "needs r_basis of shape (E, n_rbf).")
            feats.append(r_basis)
        out = self.mlp(torch.cat(feats, dim=-1))          # (E, out_dim)
        if self.shift:
            # [ scale | m=0 shift ] — beta is (E, n_features), never per-mode, and
            # needs no mode_valid mask (m=0 is a real mode for every channel, l >= 0).
            scale_delta, beta = out.split([self.n_scale, self.n_features], dim=-1)
            return self.base + self._shape(scale_delta), beta
        return self.base + self._shape(out), None
