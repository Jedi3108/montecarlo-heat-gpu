# GPU Monte Carlo Radiative Heat — CuPy/CUDA
# ------------------------------------------------------------
# Simulates radiative transfer in a rectangular box with a hot
# emitter patch on the z=0 wall. Rays are emitted Lambertian,
# reflect diffusely from walls with emissivity ε, and deposit
# absorbed energy into per-wall heatmaps. Batched + vectorized
# with CuPy so you can push to millions of rays.
# ------------------------------------------------------------

import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import time

print("CuPy:", cp.__version__)
_ = cp.arange(1_000_000, dtype=cp.float32) + 1  # quick smoke test
print("GPU OK, sample array:", _.shape)

# -----------------------------
# Scene types
# -----------------------------
@dataclass
class Box:
    Lx: float = 1.0
    Ly: float = 0.6
    Lz: float = 0.4

@dataclass
class EmitterPatch:
    cx: float = 0.5   # center x
    cy: float = 0.3   # center y
    ax: float = 0.2   # width  along x
    ay: float = 0.1   # height along y

@dataclass
class Scene:
    box: Box
    emitter: EmitterPatch
    walls_eps: dict          # {'x0':eps, 'xL':..., 'y0':..., 'yL':..., 'z0':..., 'zL':...}
    emitter_emissive_power: float = 1.0  # arbitrary units

# Wall IDs and inward-facing normals
WALL_IDS = ['x0','xL','y0','yL','z0','zL']
WALL_NORMALS = cp.asarray([[+1,0,0], [-1,0,0],
                           [0,+1,0], [0,-1,0],
                           [0,0,+1], [0,0,-1]], dtype=cp.float32)

# -----------------------------
# Sampling: cosine-weighted hemisphere (batched)
# -----------------------------
def cp_cosine_weighted_hemisphere(N: cp.ndarray, rng: cp.random.RandomState) -> cp.ndarray:
    """N: (B,3) unit normals; return (B,3) cosine-weighted directions."""
    B = N.shape[0]
    u = rng.rand(B, dtype=cp.float32)
    v = rng.rand(B, dtype=cp.float32)
    r  = cp.sqrt(u)
    phi= 2*cp.pi*v
    x = r*cp.cos(phi); y = r*cp.sin(phi); z = cp.sqrt(cp.maximum(0.0, 1 - r*r))

    # Build tangent frames
    cond = cp.abs(N[:,2]) < 0.9
    A = cp.where(cond[:,None],
                 cp.array([0,0,1], dtype=cp.float32)[None,:],
                 cp.array([1,0,0], dtype=cp.float32)[None,:])
    T = cp.cross(A, N)
    T = T / (cp.linalg.norm(T, axis=1, keepdims=True) + 1e-20)
    Bv = cp.cross(N, T)
    dirs = x[:,None]*T + y[:,None]*Bv + z[:,None]*N
    return dirs

# -----------------------------
# Intersections: vectorized ray vs 6 planes of the box
# -----------------------------
def cp_intersect_box(o: cp.ndarray, d: cp.ndarray, box: Box):
    """
    o, d: (B,3) origins and directions.
    Returns:
      k:   (B,) int wall index in [0..5]
      hp:  (B,3) hit point
      nrm: (B,3) inward normal at wall
      valid: (B,) bool mask of rays that hit a wall
    """
    B = o.shape[0]
    inf = cp.float32(1e30)
    eps = 1e-12

    t_list, hp_list = [], []

    # x=0 (normal +x)
    mask = cp.abs(d[:,0]) > eps
    t = cp.where(mask, (-o[:,0]) / d[:,0], inf)
    y = o[:,1] + t*d[:,1]; z = o[:,2] + t*d[:,2]
    inside = (t > 1e-9) & (y >= 0) & (y <= box.Ly) & (z >= 0) & (z <= box.Lz)
    hp = cp.stack([cp.zeros_like(y), y, z], axis=1)
    t_list.append(cp.where(inside, t, inf)); hp_list.append(hp)

    # x=Lx (normal -x)
    t = cp.where(mask, (box.Lx - o[:,0]) / d[:,0], inf)
    y = o[:,1] + t*d[:,1]; z = o[:,2] + t*d[:,2]
    inside = (t > 1e-9) & (y >= 0) & (y <= box.Ly) & (z >= 0) & (z <= box.Lz)
    hp = cp.stack([cp.full_like(y, box.Lx), y, z], axis=1)
    t_list.append(cp.where(inside, t, inf)); hp_list.append(hp)

    # y=0 (normal +y)
    mask_y = cp.abs(d[:,1]) > eps
    t = cp.where(mask_y, (-o[:,1]) / d[:,1], inf)
    x = o[:,0] + t*d[:,0]; z = o[:,2] + t*d[:,2]
    inside = (t > 1e-9) & (x >= 0) & (x <= box.Lx) & (z >= 0) & (z <= box.Lz)
    hp = cp.stack([x, cp.zeros_like(x), z], axis=1)
    t_list.append(cp.where(inside, t, inf)); hp_list.append(hp)

    # y=Ly (normal -y)
    t = cp.where(mask_y, (box.Ly - o[:,1]) / d[:,1], inf)
    x = o[:,0] + t*d[:,0]; z = o[:,2] + t*d[:,2]
    inside = (t > 1e-9) & (x >= 0) & (x <= box.Lx) & (z >= 0) & (z <= box.Lz)
    hp = cp.stack([x, cp.full_like(x, box.Ly), z], axis=1)
    t_list.append(cp.where(inside, t, inf)); hp_list.append(hp)

    # z=0 (normal +z)
    mask_z = cp.abs(d[:,2]) > eps
    t = cp.where(mask_z, (-o[:,2]) / d[:,2], inf)
    x = o[:,0] + t*d[:,0]; y = o[:,1] + t*d[:,1]
    inside = (t > 1e-9) & (x >= 0) & (x <= box.Lx) & (y >= 0) & (y <= box.Ly)
    hp = cp.stack([x, y, cp.zeros_like(x)], axis=1)
    t_list.append(cp.where(inside, t, inf)); hp_list.append(hp)

    # z=Lz (normal -z)
    t = cp.where(mask_z, (box.Lz - o[:,2]) / d[:,2], inf)
    x = o[:,0] + t*d[:,0]; y = o[:,1] + t*d[:,1]
    inside = (t > 1e-9) & (x >= 0) & (x <= box.Lx) & (y >= 0) & (y <= box.Ly)
    hp = cp.stack([x, y, cp.full_like(x, box.Lz)], axis=1)
    t_list.append(cp.where(inside, t, inf)); hp_list.append(hp)

    T = cp.stack(t_list, axis=1)   # (B,6)
    k = cp.argmin(T, axis=1)       # nearest wall id per ray
    valid = cp.isfinite(T[cp.arange(B), k])

    hp_out  = cp.zeros_like(o)
    nrm_out = cp.zeros_like(o)
    for j in range(6):
        sel = k == j
        if cp.any(sel):
            hp_out[sel]  = hp_list[j][sel]
            nrm_out[sel] = WALL_NORMALS[j]
    return k, hp_out, nrm_out, valid

# -----------------------------
# Heatmap buffers + scatter-add
# -----------------------------
def make_heatmaps(bins_xy=(240, 180)):
    """
    Returns dict of per-wall heatmaps (CuPy arrays).
    Shapes:
      x0,xL -> (Ny, Nz)  (y–z)
      y0,yL -> (Nx, Nz)  (x–z)
      z0,zL -> (Nx, Ny)  (x–y)
    """
    Nx, Ny = bins_xy[0], bins_xy[1]
    # For simplicity we reuse Ny as Nz; tweak if you want non-square depth grids
    Nz = Ny
    return {
        'x0': cp.zeros((Ny, Nz), dtype=cp.float32),  # y,z
        'xL': cp.zeros((Ny, Nz), dtype=cp.float32),
        'y0': cp.zeros((Nx, Nz), dtype=cp.float32),  # x,z
        'yL': cp.zeros((Nx, Nz), dtype=cp.float32),
        'z0': cp.zeros((Nx, Ny), dtype=cp.float32),  # x,y
        'zL': cp.zeros((Nx, Ny), dtype=cp.float32),
    }

def scatter_wall_energy(wid_idx: cp.ndarray, hp: cp.ndarray, Eabs: cp.ndarray, scene: Scene, H: dict):
    """Scatter-add absorbed energy into the wall heatmaps."""
    box = scene.box
    for j, wid in enumerate(WALL_IDS):
        sel = wid_idx == j
        if not cp.any(sel):
            continue
        hpj = hp[sel]
        Ej  = Eabs[sel]
        if wid in ('x0','xL'):
            Ny, Nz = H[wid].shape
            y = cp.clip(hpj[:,1], 0, box.Ly - 1e-12)
            z = cp.clip(hpj[:,2], 0, box.Lz - 1e-12)
            iy = cp.minimum((y / box.Ly * Ny).astype(cp.int32), Ny-1)
            iz = cp.minimum((z / box.Lz * Nz).astype(cp.int32), Nz-1)
            cp.add.at(H[wid], (iy, iz), Ej)
        elif wid in ('y0','yL'):
            Nx, Nz = H[wid].shape
            x = cp.clip(hpj[:,0], 0, box.Lx - 1e-12)
            z = cp.clip(hpj[:,2], 0, box.Lz - 1e-12)
            ix = cp.minimum((x / box.Lx * Nx).astype(cp.int32), Nx-1)
            iz = cp.minimum((z / box.Lz * Nz).astype(cp.int32), Nz-1)
            cp.add.at(H[wid], (ix, iz), Ej)
        elif wid in ('z0','zL'):
            Nx, Ny = H[wid].shape
            x = cp.clip(hpj[:,0], 0, box.Lx - 1e-12)
            y = cp.clip(hpj[:,1], 0, box.Ly - 1e-12)
            ix = cp.minimum((x / box.Lx * Nx).astype(cp.int32), Nx-1)
            iy = cp.minimum((y / box.Ly * Ny).astype(cp.int32), Ny-1)
            cp.add.at(H[wid], (ix, iy), Ej)

# -----------------------------
# GPU tracer (batched)
# -----------------------------
def trace_scene_cupy(scene: Scene, Nrays=2_000_000, batch=250_000, max_bounces=10,
                     energy_epsilon=1e-4, bins_xy=(240, 180), seed=7):
    rng  = cp.random.RandomState(seed)
    box  = scene.box
    H    = make_heatmaps(bins_xy=bins_xy)
    eps  = {k: float(scene.walls_eps.get(k, 0.8)) for k in WALL_IDS}
    E0   = scene.emitter_emissive_power / Nrays
    absorbed_totals = {k: 0.0 for k in WALL_IDS}

    done = 0
    while done < Nrays:
        B = min(batch, Nrays - done)

        # Sample batch origins on the emitter patch (z=0)
        ox = scene.emitter.cx - scene.emitter.ax/2 + rng.rand(B, dtype=cp.float32) * scene.emitter.ax
        oy = scene.emitter.cy - scene.emitter.ay/2 + rng.rand(B, dtype=cp.float32) * scene.emitter.ay
        oz = cp.zeros(B, dtype=cp.float32)
        o  = cp.stack([ox, oy, oz], axis=1)

        # Emitter normals are +z
        N_emit = cp.tile(cp.asarray([0,0,1], dtype=cp.float32), (B,1))
        d = cp_cosine_weighted_hemisphere(N_emit, rng)

        E = cp.full(B, E0, dtype=cp.float32)
        alive = cp.ones(B, dtype=cp.bool_)

        for _ in range(max_bounces):
            if not bool(cp.any(alive)):
                break
            idx_alive = cp.where(alive)[0]
            k, hp, nrm, valid = cp_intersect_box(o[alive], d[alive], box)
            valid_idx = idx_alive[valid]
            if valid_idx.size == 0:
                alive[...] = False
                break

            # Absorption on hit walls
            wid_idx = k[valid]
            eps_vec = cp.asarray([eps[WALL_IDS[int(i)]] for i in wid_idx.tolist()], dtype=cp.float32)
            Eabs    = eps_vec * E[valid_idx]
            scatter_wall_energy(wid_idx, hp[valid], Eabs, scene, H)

            # Accumulate totals (fast enough to pull sums per wall)
            for j, wid in enumerate(WALL_IDS):
                mask = wid_idx == j
                if bool(cp.any(mask)):
                    absorbed_totals[wid] += float(cp.sum(Eabs[mask]).get())

            # Reflect remaining energy and continue
            E_ref = (1.0 - eps_vec) * E[valid_idx]
            E[valid_idx] = E_ref
            alive[valid_idx] = E_ref >= energy_epsilon

            d_new = cp_cosine_weighted_hemisphere(nrm[valid], rng)
            d[valid_idx] = d_new
            o[valid_idx] = hp[valid] + 1e-4 * d_new

        done += B
        if Nrays >= 10 and (done % max(1, Nrays//10) == 0 or done == Nrays):
            print(f"progress: {done}/{Nrays} ({done/Nrays*100:.1f}%)")

    return H, absorbed_totals

# -----------------------------
# Example run
# -----------------------------
box = Box(1.0, 0.6, 0.4)
emitter = EmitterPatch(cx=0.5, cy=0.3, ax=0.2, ay=0.1)
walls_eps = {'x0':0.9, 'xL':0.5, 'y0':0.8, 'yL':0.8, 'z0':0.2, 'zL':0.9}
scene = Scene(box=box, emitter=emitter, walls_eps=walls_eps, emitter_emissive_power=1.0)

Nrays   = 500_000_000   # try 5_000_000 or 10_000_000 if your GPU allows
batch   = 500_000     # lower if VRAM is tight; raise for speed if memory allows
bins_xy = (240, 180)  # heatmap resolution (Nx, Ny)

t0 = time.perf_counter()
H, totals = trace_scene_cupy(scene, Nrays=Nrays, batch=batch, max_bounces=10,
                             energy_epsilon=1e-4, bins_xy=bins_xy, seed=7)
cp.cuda.Device().synchronize()
t1 = time.perf_counter()
print(f"\nGPU runtime: {t1 - t0:.2f} s for {Nrays/1e6:.1f}M rays")
print("Absorbed totals (approx):")
for k in WALL_IDS:
    print(f"  {k}: {totals[k]:.6f}")
print("Sum absorbed ≈", sum(totals.values()), " | emitter power =", scene.emitter_emissive_power)

# Plot a wall heatmap (front wall z=Lz)
def to_cpu_norm(G):
    Gc = cp.asnumpy(G)
    m = Gc.max() if Gc.size else 1.0
    return (Gc / m) if m > 0 else Gc

HzL = to_cpu_norm(H['zL'])

plt.figure(figsize=(6,4))
plt.imshow(HzL.T, origin='lower', aspect='auto',
           extent=[0, scene.box.Lx, 0, scene.box.Ly])
plt.title(f"Front wall heat map (z=Lz), {Nrays/1e6:.1f}M rays")
plt.xlabel("x"); plt.ylabel("y")
plt.colorbar(); plt.tight_layout(); plt.show()

