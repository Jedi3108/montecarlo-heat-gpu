# --- Spectral Monte Carlo radiative heat (CuPy) ---
# Extends the existing GPU tracer to:
#  - sample photon wavelengths from a blackbody emitter (Planck distribution)
#  - use wavelength-dependent wall emissivity:  eps_wall(λ) = clamp(eps0 * (λ/λ0)^alpha, 0.01, 0.99)
#  - (hook) temperature dependence via per-wall T if you want later (kept constant here)

import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

# Reuse from your previous cell:
# - WALL_IDS
# - cp_cosine_weighted_hemisphere
# - cp_intersect_box
# - make_heatmaps
# - scatter_wall_energy
# - Scene, Box, EmitterPatch

# --- Physical constants (SI) ---
h  = 6.62607015e-34     # Planck constant [J s]
c  = 2.99792458e8       # speed of light [m/s]
kB = 1.380649e-23       # Boltzmann constant [J/K]

def cp_planck_lambda(lam_m: cp.ndarray, T_K: float) -> cp.ndarray:
    """Spectral radiance (up to a constant) vs wavelength for a blackbody at T."""
    # B_lambda ∝ (1/λ^5) * 1/(exp(hc/(λ k T)) - 1)
    x = (h*c) / (lam_m * kB * T_K)
    return (lam_m**-5) * (cp.exp(x) - 1.0)**-1

def make_blackbody_cdf(T_emit: float, lam_min=3e-7, lam_max=2e-5, N=4096):
    """
    Build wavelength grid + CDF for blackbody at T_emit.
    Returns lam_grid (cp.float32, m) and CDF (cp.float32, 0..1).
    """
    lam = cp.linspace(lam_min, lam_max, N, dtype=cp.float32)
    pdf = cp_planck_lambda(lam, T_emit)
    # numerical stability
    pdf = cp.maximum(pdf, pdf.max() * 1e-12)
    cdf = cp.cumsum(pdf)
    cdf /= cdf[-1]
    return lam.astype(cp.float32), cdf.astype(cp.float32)

def sample_wavelengths(lam_grid, cdf, n, rng):
    """Inverse-CDF sampling of wavelengths on GPU."""
    u = rng.rand(n, dtype=cp.float32)
    idx = cp.searchsorted(cdf, u, side='left')
    idx = cp.clip(idx, 0, lam_grid.size-1)
    return lam_grid[idx]

# ---- Per-wall spectral emissivity model ----------------------------------
# eps(λ) = clamp(eps0 * (λ/λ0)**alpha, eps_min, eps_max)
# You can fit alpha from data; here we just set reasonable signs (e.g., metals: alpha<0 in VIS/NIR).
LAM_REF = 1.0e-6  # 1 micron reference
EPS_MIN, EPS_MAX = 0.01, 0.99

class SpectralWalls:
    def __init__(self, eps0: dict, alpha: dict, T_wall: dict=None):
        """
        eps0:  base emissivity per wall id (as before)
        alpha: spectral exponent per wall id (controls λ-dependence)
        T_wall: optional constant temperature per wall (unused in formula below but kept for future)
        """
        self.eps0 = {k: float(eps0.get(k, 0.8)) for k in WALL_IDS}
        self.alpha = {k: float(alpha.get(k, 0.0)) for k in WALL_IDS}
        self.T_wall = {k: float((T_wall or {}).get(k, 300.0)) for k in WALL_IDS}

    def eps_at(self, wid_idx: cp.ndarray, lam: cp.ndarray) -> cp.ndarray:
        # Gather per-ray (eps0, alpha) from wall index
        eps0 = cp.asarray([self.eps0[WALL_IDS[int(i)]] for i in wid_idx.tolist()], dtype=cp.float32)
        a    = cp.asarray([self.alpha[WALL_IDS[int(i)]] for i in wid_idx.tolist()], dtype=cp.float32)
        val  = eps0 * (lam / LAM_REF)**a
        return cp.clip(val, EPS_MIN, EPS_MAX)

# ---- Spectral tracer ------------------------------------------------------
def trace_scene_cupy_spectral(scene: Scene, spectral: SpectralWalls,
                              T_emit=1200.0, lam_min=3e-7, lam_max=2e-5, Nlam=4096,
                              Nrays=5_000_000, batch=400_000, max_bounces=10,
                              energy_epsilon=1e-4, bins_xy=(240, 180), seed=42):
    """
    As the basic GPU tracer, but each ray carries a sampled wavelength λ and
    wall emissivity depends on λ via the spectral model.
    """
    rng = cp.random.RandomState(seed)
    lam_grid, cdf = make_blackbody_cdf(T_emit, lam_min, lam_max, Nlam)

    box = scene.box
    H = make_heatmaps(bins_xy=bins_xy)
    E0 = scene.emitter_emissive_power / Nrays
    absorbed_totals = {k: 0.0 for k in WALL_IDS}

    done = 0
    while done < Nrays:
        B = min(batch, Nrays - done)

        # Sample emitter origins on z=0 patch
        ox = scene.emitter.cx - scene.emitter.ax/2 + rng.rand(B, dtype=cp.float32) * scene.emitter.ax
        oy = scene.emitter.cy - scene.emitter.ay/2 + rng.rand(B, dtype=cp.float32) * scene.emitter.ay
        oz = cp.zeros(B, dtype=cp.float32)
        o  = cp.stack([ox, oy, oz], axis=1)

        # Initial directions (Lambertian) and wavelengths (blackbody)
        N_emit = cp.tile(cp.asarray([0,0,1], dtype=cp.float32), (B,1))
        d  = cp_cosine_weighted_hemisphere(N_emit, rng)
        lam= sample_wavelengths(lam_grid, cdf, B, rng)

        E = cp.full(B, E0, dtype=cp.float32)
        alive = cp.ones(B, dtype=cp.bool_)

        for _ in range(max_bounces):
            if not bool(cp.any(alive)): break
            idx_alive = cp.where(alive)[0]
            k, hp, nrm, valid = cp_intersect_box(o[alive], d[alive], box)
            valid_idx = idx_alive[valid]
            if valid_idx.size == 0:
                alive[...] = False
                break

            # Absorption with spectral eps(λ)
            wid_idx = k[valid]
            lam_v   = lam[valid_idx]
            eps_vec = spectral.eps_at(wid_idx, lam_v)
            Eabs    = eps_vec * E[valid_idx]
            scatter_wall_energy(wid_idx, hp[valid], Eabs, scene, H)

            # Totals per wall (host-side accumulation)
            for j, wid in enumerate(WALL_IDS):
                mask = wid_idx == j
                if bool(cp.any(mask)):
                    absorbed_totals[wid] += float(cp.sum(Eabs[mask]).get())

            # Reflection (Lambertian, same λ)
            E_ref = (1.0 - eps_vec) * E[valid_idx]
            E[valid_idx] = E_ref
            alive[valid_idx] = E_ref >= energy_epsilon

            d_new = cp_cosine_weighted_hemisphere(nrm[valid], rng)
            d[valid_idx] = d_new
            o[valid_idx] = hp[valid] + 1e-4 * d_new
            # lam unchanged (no fluorescence/shift here)

        done += B
        if Nrays >= 10 and (done % max(1, Nrays//10) == 0 or done == Nrays):
            print(f"progress: {done}/{Nrays} ({done/Nrays*100:.1f}%)")

    return H, absorbed_totals, (lam_grid, cdf)

# ---------- Example: run one spectral case ----------
# Choose per-wall spectral trends:
#   Metals (like xL) often have lower ε in short λ (visible/NIR) → alpha < 0
#   Painted/ceramic walls (zL, y*) more gray → alpha ~ 0..+0.2
eps0 = {'x0':0.85,'xL':0.45,'y0':0.80,'yL':0.80,'z0':0.20,'zL':0.90}
alpha= {'x0':+0.10,'xL':-0.30,'y0':+0.05,'yL':+0.05,'z0':0.00,'zL':+0.10}  # tweak freely

spectral = SpectralWalls(eps0=eps0, alpha=alpha)

# Reuse your scene (or redefine)
try:
    scene
except NameError:
    box = Box(1.0, 0.6, 0.4)
    emitter = EmitterPatch(cx=0.5, cy=0.3, ax=0.2, ay=0.1)
    # walls_eps is not used directly here; emissivity comes from SpectralWalls
    scene = Scene(box=box, emitter=emitter, walls_eps={}, emitter_emissive_power=1.0)

T_emit = 1200.0   # K (dull red-hot)
Nrays   = 500_000_000
batch   = 500_000
bins_xy = (240, 180)

import time
t0 = time.perf_counter()
Hspec, totals_spec, _ = trace_scene_cupy_spectral(
    scene, spectral, T_emit=T_emit, Nrays=Nrays, batch=batch,
    max_bounces=10, bins_xy=bins_xy, seed=123
)
cp.cuda.Device().synchronize()
t1 = time.perf_counter()
print(f"\nSpectral GPU runtime: {t1 - t0:.2f}s for {Nrays/1e6:.1f}M rays @ T_emit={T_emit} K")
print("Absorbed totals (spectral):")
for k in WALL_IDS:
    print(f"  {k}: {totals_spec[k]:.6f}")
print("Sum absorbed ≈", sum(totals_spec.values()), " | emitter power =", scene.emitter_emissive_power)

# Visualize front wall (z=Lz)
def to_cpu_norm(G):
    Gc = cp.asnumpy(G)
    m = Gc.max() if Gc.size else 1.0
    return (Gc / m) if m > 0 else Gc

HzL = to_cpu_norm(Hspec['zL'])
plt.figure(figsize=(6,4))
plt.imshow(HzL.T, origin='lower', aspect='auto',
           extent=[0, scene.box.Lx, 0, scene.box.Ly])
plt.title(f"Front wall heat map (spectral, T_emit={T_emit:.0f}K)")
plt.xlabel("x"); plt.ylabel("y")
plt.colorbar(); plt.tight_layout(); plt.show()

