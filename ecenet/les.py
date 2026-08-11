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

import math

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
                return_charges: bool = False,
                n_struct: int | None = None,
                l0_is_charge: bool = False,
                les_dipole: bool = False):
        """Long-range energy for one structure or a packed batch.

        l0        (N, C)   per-atom invariant descriptor (any flattenable
                  shape) — or, with ``l0_is_charge=True``, the latent charge
                  itself ((N,) or (N, 1); a model built with
                  ``les_readout='edge'`` emits this), in which case
                  upstream's atomwise head is bypassed entirely and this
                  module holds no parameters. With ``les_dipole=True``
                  (requires ``l0_is_charge``), l0 is the model's packed
                  (N, 4) = [q | u] and the latent atomic dipoles u are passed
                  to upstream's charge–dipole/dipole–dipole terms.
        positions (N, 3)   in Å, on the same autograd graph as the SR energy
        cell      (B, 3, 3) or (3, 3); None → isolated / non-periodic, served
                  by the vectorized batched path below (verified equal to
                  upstream's per-structure loop in tests/test_les.py,
                  charge-only and with dipoles).
        batch     (N,) structure index per atom; None → single structure
        n_struct  number of structures (optional; saves a batch.max() GPU
                  sync on the isolated path when the caller knows it)

        Returns the per-structure long-range energy in eV (with
        ``return_charges=True``, also the per-atom latent charges — the q
        column only under ``les_dipole``; take u from l0 directly).
        """
        if les_dipole and not l0_is_charge:
            raise ValueError("les_dipole=True requires l0_is_charge=True "
                             "(the packed [q | u] comes from the model's "
                             "edge head).")
        if batch is None:
            batch = torch.zeros(positions.shape[0], dtype=torch.long,
                                device=positions.device)
            if n_struct is None:
                n_struct = 1
        # unpack once; the branches below carry u alongside the charges
        q, u = (l0[:, 0], l0[:, 1:4]) if les_dipole else (l0, None)
        if cell is None:
            return self._isolated_batched(q, positions, batch, n_struct,
                                          return_charges, l0_is_charge, u=u)
        # Scope the default dtype to the input's so upstream's lazily built
        # charge MLP (and any default-dtype internals) match float64 inputs.
        prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(positions.dtype)
        try:
            if l0_is_charge:
                result = self.les(
                    latent_charges=q.reshape(-1),
                    latent_dipoles=u,
                    positions=positions,
                    cell=cell.view(-1, 3, 3),
                    batch=batch,
                    compute_energy=True,
                )
            else:
                result = self.les(
                    desc=q.reshape(q.shape[0], -1),
                    positions=positions,
                    cell=cell.view(-1, 3, 3),
                    batch=batch,
                    compute_energy=True,
                )
        finally:
            torch.set_default_dtype(prev_dtype)
        if return_charges:
            charges = (q.reshape(-1) if l0_is_charge
                       else result["latent_charges"])
            return result["E_lr"], charges
        return result["E_lr"]

    def _isolated_batched(self, l0, positions, batch, n_struct, return_charges,
                          l0_is_charge=False, u=None):
        """Isolated (non-periodic) long-range energy, vectorized over the batch.

        Upstream's ``Les.forward`` loops over structures in Python — masked
        gathers plus a dozen small kernels per structure, which is launch-
        latency-bound for many-molecule batches (measured ~2× SPICE step time;
        the same disease forward_batch_multi cured on the model side). This
        path computes the identical quantity in a handful of full-batch
        kernels. No LES physics is reimplemented: the latent charges come from
        upstream's own ``atomwise`` head and the interaction matrix from
        upstream's ``make_kernels`` (smearing kernel, self-term convention,
        constants); this method only masks cross-structure pairs out of the
        quadratic form and scatters it per structure. With latent dipoles
        ``u`` (N, 3) the same masking extends to upstream's charge–dipole
        and dipole–dipole terms, mirroring ``compute_potential_realspace``
        exactly:

            E_b = ½ qᵀf_qq q + (u·f_qu)ᵀq − ½ uᵀf_uu u   (i,j ∈ b)

        (the qu coefficient is 1, not ½, and the self-interaction is removed
        by ``make_kernels`` — both upstream's conventions). Verified equal to
        upstream's loop (energies, charges, and position gradients) in
        tests/test_les.py, charge-only and with dipoles.

        Tradeoff: the dense kernel spans ALL atom pairs, so this does
        (ΣN)² pair work where the loop does Σ(N_b²) — ~batch_size× redundant
        FLOPs. On GPU that is the right trade (a few large kernels vs
        hundreds of tiny launches + syncs); on CPU the loop can be faster,
        but training runs there don't care. Memory is (ΣN)² per kernel
        tensor — fine for typical atom budgets (2000 atoms → 32 MB fp64);
        the dipole–dipole kernel is (ΣN)²·3·3, i.e. 9× that, so dipole runs
        want correspondingly smaller atom budgets.
        """
        from les.module.ewald import make_kernels

        if l0_is_charge:
            q = l0.reshape(-1)          # les_readout='edge': l0 IS the charge
        else:
            prev_dtype = torch.get_default_dtype()
            torch.set_default_dtype(positions.dtype)
            try:
                q = self.les.atomwise(l0.reshape(l0.shape[0], -1), batch)
            finally:
                torch.set_default_dtype(prev_dtype)
        charges = q
        if q.dim() == 1:
            q = q.unsqueeze(1)
        q = q.to(positions.dtype)

        if n_struct is None:
            n_struct = int(batch.max().item()) + 1

        # Structures in a batch are individually centered, so atoms from
        # DIFFERENT structures can (nearly) coincide. make_kernels masks only
        # the exact diagonal: a coincident cross-structure pair yields
        # 1/r = inf and erf(0)·inf = NaN, which the post-hoc mask cannot
        # clean (NaN·0 = NaN; and a torch.where would still leak NaN through
        # the 1/r backward). Shift each structure onto its own site of a
        # coarse 2D grid first: intra-structure distances are translation-
        # invariant (up to fp cancellation noise ~eps·offset), cross
        # distances become large and finite, and the finite cross entries
        # are then masked to exact zero. The spacing is detached, so no
        # gradient flows through the offsets.
        k = max(1, math.isqrt(max(0, n_struct - 1)) + 1)      # grid side ≥ √B
        spacing = 2.0 * positions.detach().abs().max() + 10.0
        gx = (batch % k).to(positions.dtype)
        gy = torch.div(batch, k, rounding_mode='floor').to(positions.dtype)
        shift = torch.stack([gx, gy, torch.zeros_like(gx)], dim=1) * spacing

        ew = self.les.ewald
        f_qq, f_qu, f_uu, _, _ = make_kernels(positions + shift, ew.sigma,
                                              ew.norm_factor / ew.twopi,
                                              compute_u=u is not None,
                                              compute_Q=False)
        same = (batch.unsqueeze(0) == batch.unsqueeze(1)).to(f_qq.dtype)
        e_phi = torch.einsum('iq,ij->jq', q, f_qq * same)
        per_atom = 0.5 * (e_phi * q).sum(dim=-1)                    # (N,)
        if u is not None:
            # upstream's dipole terms with the same cross-structure masking;
            # coefficients (qu: 1, uu: -1/2) mirror compute_potential_realspace.
            # `same` is pair-indexed and component-independent, so it commutes
            # past the c/d contractions — mask the contracted (N, N) results
            # instead of building masked copies of the (N, N, 3[, 3]) kernels
            # (the uu copy alone would double the documented 9× memory).
            u_ = u.to(positions.dtype)                              # (N, 3)
            M = torch.einsum('ic,ijc->ij', u_, f_qu)
            per_atom = per_atom + (M * same).sum(dim=0) * q[:, 0]
            G = torch.einsum('ic,ijcd->ijd', u_, f_uu)              # (N, N, 3)
            T = torch.einsum('ijd,jd->ij', G, u_)
            per_atom = per_atom - 0.5 * (T * same).sum(dim=0)
        e_lr = torch.zeros(n_struct, dtype=per_atom.dtype,
                           device=per_atom.device
                           ).scatter_add(0, batch, per_atom)
        if return_charges:
            return e_lr, charges
        return e_lr


def load_les_module(ckpt_les, model, device, dtype, load_state=True):
    """Rebuild a ready-to-use ``LESLongRange`` from a checkpoint's ``les`` dict.

    The one shared implementation of the materialise-then-load dance that
    trainers, tools, and calculators all need: upstream builds its charge
    head **lazily on the first forward** (it infers the descriptor width
    then), so state loaded into an unmaterialised head would silently no-op.
    This materialises with a synthetic probe of the model's ``l0`` width —
    no dataset, topology, or cell required — then loads the trained state
    (``best_state`` if present) and switches to eval mode. Edge-mode
    read-outs bypass the head entirely (the module is parameter-free), so
    the probe is skipped there.

    ``load_state=False`` returns a materialised but freshly-initialised
    module (what a trainer wants before its own optimiser/restore takes over).
    """
    module = LESLongRange(ckpt_les.get('arguments') if ckpt_les else None)
    flags = model.les_flags
    if not flags['l0_is_charge']:
        with torch.no_grad():
            l0_probe = torch.zeros(2, model._l0_dim, dtype=dtype, device=device)
            pos_probe = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                                     dtype=dtype, device=device)
            module(l0_probe, pos_probe, **flags)
    module = module.to(device=device, dtype=dtype)
    if load_state and ckpt_les:
        module.load_state_dict(ckpt_les.get('best_state')
                               or ckpt_les['state_dict'])
    module.eval()
    return module
