# --- Tabulated spectral emissivity (optionally temperature-dependent) ---
# Works with your existing trace_scene_cupy_spectral(...) by providing the same
# API: an object with eps_at(wid_idx, lam) -> per-ray emissivity on the GPU.

import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

WALL_IDS  # should already exist from previous py files

def _gather_per_wall(vals_dict, wid_idx):
    """Helper: gather per-ray values from a python dict keyed by wall id."""
    return cp.asarray([vals_dict[WALL_IDS[int(i)]] for i in wid_idx.tolist()], dtype=cp.float32)

class TabulatedSpectralWalls:
    """
    Use real data if you have it:
      - For 1D (wavelength-only): provide lam_grid and eps_lam per wall.
      - For 2D (wavelength & temperature): provide lam_grid, T_grid and eps_table per wall.
    You can mix 1D and 2D walls; 2D takes precedence if provided.

    Example dict for a wall 'zL':
      lam_grid['zL'] = cp.array([...], float32)              (m)
      eps_lam['zL']  = cp.array([...], float32)              (same length)
      T_grid['zL']   = cp.array([...], float32)              (K)  # optional
      eps_table['zL']= cp.array(shape=(len(T_grid), len(lam_grid)), float32)  # optional
    """
    def __init__(self, lam_grid: dict, eps_lam: dict,
                 T_grid: dict=None, eps_table: dict=None,
                 eps_min=0.01, eps_max=0.99):
        self.lam_grid = lam_grid
        self.eps_lam  = eps_lam
        self.T_grid   = T_grid or {}
        self.eps_tab  = eps_table or {}
        self.eps_min  = float(eps_min)
        self.eps_max  = float(eps_max)

    @staticmethod
    def _interp1(lgrid, y, xq):
        # lgrid, y, xq are CuPy arrays; monotonic lgrid assumed.
        # cp.interp is 1D linear interpolation.
        return cp.interp(xq, lgrid, y)

    @staticmethod
    def _interp2(lgrid, Tgrid, table, lam_q, T_q):
        """
        Bilinear interp over (T, lam). Shapes:
          lgrid: (L,), Tgrid: (M,), table: (M,L)
          lam_q, T_q: (N,)
        Returns yq: (N,)
        """
        # find indices
        # For each query, idx_l is right bin index (>=1)
        idx_l = cp.clip(cp.searchsorted(lgrid, lam_q, side='right') - 1, 0, lgrid.size-2)
        idx_t = cp.clip(cp.searchsorted(Tgrid, T_q,  side='right') - 1, 0, Tgrid.size-2)

        l0 = lgrid[idx_l]; l1 = lgrid[idx_l+1]
        t0 = Tgrid[idx_t]; t1 = Tgrid[idx_t+1]
        wl = cp.where(l1>l0, (lam_q - l0)/(l1 - l0), 0.0)
        wt = cp.where(t1>t0, (T_q   - t0)/(t1 - t0), 0.0)

        # gather four corners
        f00 = table[idx_t,   idx_l   ]
        f01 = table[idx_t,   idx_l+1 ]
        f10 = table[idx_t+1, idx_l   ]
        f11 = table[idx_t+1, idx_l+1 ]
        # bilinear
        return (1-wt)*((1-wl)*f00 + wl*f01) + wt*((1-wl)*f10 + wl*f11)

    def eps_at(self, wid_idx: cp.ndarray, lam: cp.ndarray, T_wall_const: dict=None) -> cp.ndarray:
        """
        Return ε for each ray based on its hit wall (wid_idx) and wavelength lam.
        If a wall has a 2D table, we use its T_grid/eps_table and temperature
        from T_wall_const[wall] (scalar K). Otherwise we fallback to 1D λ-only curve.
        """
        out = cp.empty_like(lam, dtype=cp.float32)
        for j, wid in enumerate(WALL_IDS):
            sel = (wid_idx == j)
            if not cp.any(sel):
                continue
            lam_q = lam[sel]
            # Prefer 2D if available
            if (wid in self.T_grid) and (wid in self.eps_tab) and (T_wall_const is not None):
                lgrid = self.lam_grid[wid]; Tgrid = self.T_grid[wid]
                table = self.eps_tab[wid]  # (M,L)
                Tq    = cp.full(lam_q.shape, float(T_wall_const[wid]), dtype=cp.float32)
                out[sel] = self._interp2(lgrid, Tgrid, table, lam_q, Tq)
            else:
                # 1D λ-only
                out[sel] = self._interp1(self.lam_grid[wid], self.eps_lam[wid], lam_q)
        return cp.clip(out, self.eps_min, self.eps_max)

# ---- Quick synthetic data to demonstrate usage (replace with real tables) ----
def demo_make_fake_tables():
    walls = {}
    lam = cp.linspace(0.3e-6, 15e-6, 512, dtype=cp.float32)  # VIS->LWIR
    T   = cp.linspace(300.0, 1200.0, 7, dtype=cp.float32)    # room to red-hot
    for wid in WALL_IDS:
        if wid in ('xL',):  # pretend 'xL' has strong T dependence (like oxidation)
            base = 0.3 + 0.1*cp.sin(cp.linspace(0, 6, lam.size, dtype=cp.float32))
            table = cp.empty((T.size, lam.size), dtype=cp.float32)
            for i, Ti in enumerate(T):
                table[i] = cp.clip(base * (1.0 + 0.0005*(Ti-300)), 0.05, 0.95)
            walls[wid] = {'lam': lam, 'T': T, 'table': table}
        else:
            # gentle λ-only slope (ceramic/painted)
            eps_curve = cp.clip(0.6 + 0.25*(lam/5e-6), 0.1, 0.95)
            walls[wid] = {'lam': lam, 'eps': eps_curve}
    return walls

demo = demo_make_fake_tables()
lam_grid = {w: demo[w]['lam'] for w in WALL_IDS}
eps_lam  = {w: demo[w].get('eps', None) for w in WALL_IDS}
T_grid   = {w: demo[w].get('T', None)   for w in WALL_IDS if 'T' in demo[w]}
eps_tab  = {w: demo[w].get('table', None) for w in WALL_IDS if 'table' in demo[w]}

# Fill missing eps_lam where only a 2D table exists (we'll still prefer 2D)
for w in WALL_IDS:
    if eps_lam[w] is None:
        # fallback: use a mid-temperature row as λ-only curve for preview
        if w in eps_tab:
            mid = eps_tab[w][eps_tab[w].shape[0]//2]
            eps_lam[w] = mid
        else:
            eps_lam[w] = cp.full_like(lam_grid[w], 0.8)

tab_walls = TabulatedSpectralWalls(lam_grid=lam_grid, eps_lam=eps_lam,
                                   T_grid=T_grid, eps_table=eps_tab)

# ---- Run the spectral tracer with tabulated walls ----
# Reuse your existing 'scene' and 'trace_scene_cupy_spectral' from the previous cell.
T_emit = 1200.0   # K
Nrays  = 3_000_000
batch  = 350_000
bins   = (260, 200)

# Set constant wall temperatures for the 2D interpolation (can be distinct per wall)
T_wall_const = {'x0': 450., 'xL': 900., 'y0': 450., 'yL': 450., 'z0': 350., 'zL': 500.}

def eps_at_wrapper(wid_idx, lam):
    # Small wrapper so we can keep using trace_scene_cupy_spectral(...) unchanged.
    return tab_walls.eps_at(wid_idx, lam, T_wall_const=T_wall_const)

# Monkey-patch a tiny adapter class so the tracer accepts it
class _Adapter:
    def eps_at(self, wid_idx, lam):
        return eps_at_wrapper(wid_idx, lam)

spectral_model = _Adapter()

import time
t0 = time.perf_counter()
H_tab, totals_tab, _ = trace_scene_cupy_spectral(
    scene, spectral_model, T_emit=T_emit, Nrays=Nrays, batch=batch,
    max_bounces=10, bins_xy=bins, seed=77
)
cp.cuda.Device().synchronize()
t1 = time.perf_counter()
print(f"\nTabulated spectral run: {t1 - t0:.2f}s for {Nrays/1e6:.1f}M rays, T_emit={T_emit}K")
print("Absorbed totals (tabulated):")
for k in WALL_IDS:
    print(f"  {k}: {totals_tab[k]:.6f}")
print("Sum absorbed ≈", sum(totals_tab.values()), " | emitter power =", scene.emitter_emissive_power)

# Visualize front wall
def to_cpu_norm(G):
    Gc = cp.asnumpy(G); m = Gc.max() if Gc.size else 1.0
    return (Gc / m) if m > 0 else Gc

HzL = to_cpu_norm(H_tab['zL'])
plt.figure(figsize=(6,4))
plt.imshow(HzL.T, origin='lower', aspect='auto',
           extent=[0, scene.box.Lx, 0, scene.box.Ly])
plt.title(f"Front wall heat map (tabulated ε(λ[,T])), T_emit={T_emit:.0f}K")
plt.xlabel("x"); plt.ylabel("y")
plt.colorbar(); plt.tight_layout(); plt.show()

