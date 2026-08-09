"""ecenet/les.py — optional LES (Latent Ewald Summation) long-range energy.

ECENet's message passing sees only atoms within ``r_cut_edge``, so it cannot
capture interactions that decay slowly with distance. LES closes that gap: a
small head predicts a scalar *latent* charge per atom from the model's
rotation-invariant embedding ``l0``, and the long-range energy is the
(smeared) Coulomb interaction between those charges — full pairwise for
isolated systems, reciprocal-space Ewald for periodic ones.

The implementation is **not vendored**. This module is a thin wrapper around
the inventors' reference package (github.com/ChengUCB/les), which predicts the
latent charges internally from the descriptor (``use_atomwise=True``) and
returns the long-range energy. The import is lazy so ``import ecenet`` (and
``import ecenet.les``) never requires the package; only constructing
`LESLongRange` does.

IP / licensing: the upstream ``les`` package is CC BY-NC 4.0 (non-commercial),
which is why it is an optional dependency rather than vendored code — users
who install it accept its terms directly. The Latent Ewald Summation
*algorithm* additionally has a UC Berkeley provisional patent (academic use
unrestricted). ECENet's own license covers only the code in this repository.

Integration status: the model exposes the per-atom embeddings —
``forward(..., return_embeddings=True[, l0_only=True])``, with the same flags
on ``forward_pbc`` / ``forward_batch`` / ``forward_batch_multi`` (the batched
paths return per-structure lists). Joint training is available for small
datasets via ``scripts/train_ecenet_xyz.py`` and, under DDP, via
``scripts/train_ecenet_spice.py`` (``use_les=True`` on either); the LES-aware
calculator and joint training in the MPtrj trainer are not yet ported.

    lr = LESLongRange()
    E_sr, l0 = model(pos, types, return_embeddings=True, l0_only=True)
    E = E_sr + lr(l0, pos).sum()                            # one autograd graph
    F = -torch.autograd.grad(E, pos, create_graph=True)[0]
"""

import torch
import torch.nn as nn

# Commit pin, mirrored in pyproject's [project.optional-dependencies] les —
# keep the two in sync. Upstream is not on PyPI and `main` moves.
_LES_PIN = "c8063fad18e3d59cb4d783e0ed5a1efea8d55b8d"

_INSTALL_HINT = (
    "LES support requires the optional 'les' package "
    "(github.com/ChengUCB/les, CC BY-NC 4.0 — non-commercial use only):\n"
    f'    pip install "les @ git+https://github.com/ChengUCB/les@{_LES_PIN}"\n'
    "or, from a source checkout of ecenet:\n"
    '    pip install -e ".[les]"'
)


def _upstream_les():
    """Lazy import of the upstream package; actionable error when missing."""
    try:
        from les import Les
    except ImportError as e:
        raise ImportError(_INSTALL_HINT) from e
    return Les


class LESLongRange(nn.Module):
    """Long-range electrostatic energy from per-atom invariant embeddings.

    Wraps upstream ``les.Les`` with ``use_atomwise=True``: the package's own
    head maps each atom's descriptor to a latent charge, so there is no
    separate charge head to keep in a checkpoint — this module's state dict
    *is* the upstream module's.

    Extra constructor arguments for upstream go in ``les_arguments`` (merged
    over ``{"use_atomwise": True}``); upstream's defaults are documented to
    usually work well.

    Upstream builds the charge MLP **lazily on the first forward** (it infers
    the descriptor width then), with two consequences handled/noted here:
    the lazy build lands in torch's default dtype regardless of the input's,
    so forward scopes the default dtype to the input's for the call; and the
    state dict is empty until one forward has run — run a forward before
    saving or loading a checkpoint of this module.
    """

    def __init__(self, les_arguments: dict | None = None):
        super().__init__()
        Les = _upstream_les()
        args = {"use_atomwise": True}
        if les_arguments:
            args.update(les_arguments)
        self.les = Les(les_arguments=args)

    def forward(self, l0: torch.Tensor, positions: torch.Tensor,
                cell: torch.Tensor | None = None,
                batch: torch.Tensor | None = None,
                return_charges: bool = False):
        """Long-range energy for one structure or a packed batch.

        l0        (N, C)   per-atom invariant descriptor (any flattenable shape)
        positions (N, 3)   in Å, on the same autograd graph as the SR energy
        cell      (B, 3, 3) or (3, 3); None → zero cell (non-periodic)
        batch     (N,) structure index per atom; None → single structure

        Returns the per-structure long-range energy in eV (with
        ``return_charges=True``, also the per-atom latent charges).
        """
        if batch is None:
            batch = torch.zeros(positions.shape[0], dtype=torch.long,
                                device=positions.device)
        n_struct = int(batch.max().item()) + 1
        if cell is None:
            cell = torch.zeros(n_struct, 3, 3, dtype=positions.dtype,
                               device=positions.device)
        # Scope the default dtype to the input's so upstream's lazily built
        # charge MLP (and any default-dtype internals) match float64 inputs.
        prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(positions.dtype)
        try:
            result = self.les(
                desc=l0.reshape(l0.shape[0], -1),
                positions=positions,
                cell=cell.view(-1, 3, 3),
                batch=batch,
                compute_energy=True,
            )
        finally:
            torch.set_default_dtype(prev_dtype)
        if return_charges:
            return result["E_lr"], result["latent_charges"]
        return result["E_lr"]
