# GPU Monte Carlo Radiative Heat — CuPy / CUDA

### Overview
This project implements a high-performance **Monte Carlo radiative heat transfer simulator** using **CuPy** (Python’s CUDA-accelerated NumPy equivalent).  
It models photon emission, reflection, and absorption inside a rectangular cavity, supporting both **monochromatic** and **spectrally-resolved** radiative transport.

The solver is designed for GPU acceleration, vectorized operations, and flexible material models — enabling simulation of millions of rays efficiently on consumer GPUs.

---

## 1. Base Module — Radiative Heat Simulation

**File:** `MCRGPU.py`

**Description:**  
Implements a Monte Carlo solver for radiative energy exchange between walls of a 3D box with a hot emitter patch.

**Key Features**
- Lambertian emission from a defined emitter patch (on z = 0 wall)  
- Diffuse reflections from all walls with configurable emissivity ε  
- Energy deposition into per-wall heatmaps  
- Batched GPU simulation supporting millions of rays  
- Fully vectorized with **CuPy** for high throughput

**Output**
- Per-wall heatmaps showing absorbed radiant energy  
- Optional visualization via matplotlib or GPU texture export  

---

## 2. Spectral Monte Carlo Extension

**File:** `SMCRGPU.py`

**Description:**  
Adds **spectral sampling** of photon wavelengths according to Planck’s law (blackbody radiation).  
This extension introduces wavelength-dependent optical properties.

**Key Features**
- Photon wavelength sampling from the **Planck distribution** for a given temperature  
- Wavelength-dependent emissivity modeled as  
  \[
  \varepsilon(\lambda) = \text{clamp}\left(\varepsilon_0 \left(\frac{\lambda}{\lambda_0}\right)^{\alpha}, 0.01, 0.99\right)
  \]
- Hooks for wall-dependent temperature models (constant or spatially varying)  
- Fully compatible with the base GPU tracer interface

---

## 3. Tabulated Spectral Emissivity

**File:** `TSEMCGPU.py`

**Description:**  
Provides a modular emissivity model that can be directly plugged into the spectral tracer.  
Supports both **static tabulated data** and **temperature-dependent lookup**.

**API:**
```python
eps = TabulatedEmissivity(table)
value = eps.eps_at(wall_idx, wavelength)
```
## Key Features

- Interpolation of measured or tabulated emissivity data
- Optional temperature correction
- Designed for GPU execution (CuPy arrays)
- Drop-in compatible with trace_scene_cupy_spectral(...)

