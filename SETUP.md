# Running VERA on a fresh GPU box — setup notes & fork changelog

This fork adds the missing glue to get the **Wave-1 PushT + MimicGen** demos running end-to-end on a
clean single-GPU machine (tested on an **H100 80GB**, Ubuntu 22.04, Python 3.11, CUDA 12.4, `uv`-managed
venv). Upstream's `pip install -e ".[...]"` does **not** produce a working environment out of the box —
several dependencies are unpinned/too-new, two are GitHub-only, and the sim + headless-rendering stack
needs system libraries. Everything we hit and fixed is below.

> TL;DR: follow **Part 2** top-to-bottom on a fresh box. **Part 1** explains what changed vs upstream
> (`github.com/sizhe-li/VERA`) and why.

---

## Part 1 — What this fork changes

### Code
| File | Change | Why |
|---|---|---|
| `pyproject.toml` | Pin `torchmetrics==1.4.0.post0`, add `torch-fidelity`, `setuptools<81`, `pyzmq`; point `eval` extra's `robomimic`/`mimicgen` at their git repos | Make `pip install -e ".[idm,video,eval]"` actually import & run (see gotchas) |
| `vera/idm/common/metrics/video/shared_registry.py` | `NoTrainLpips` import falls back to `_NoTrainLpips` | torchmetrics ≥1.9 made the symbol private |
| `vera/idm/common/metrics/video/lpips.py` | `_valid_img` import falls back to `torchmetrics.functional.image.lpips` | symbol moved in newer torchmetrics |
| `vera/server/start_server_mimicgen.py` | Load the MimicGen IDM from the **local** `vera-ckpts/idm-mimicgen-37oa162u/` (env `VERA_MIMICGEN_DYNAMICS_CKPT`) instead of a wandb run; fall back to wandb if absent | the hosted Wave-1 IDM is a local checkpoint; no wandb access needed |
| `examples/mimicgen_stack.ipynb` | `DATASET` now reads `VERA_MIMICGEN_DATASET` with a sane default | portable across machines |

### New files
| File | What |
|---|---|
| `examples/run_idm_on_video.py` | Run the IDM standalone on **your own video** (frames → recovered actions). Bypasses the planner; tracks consecutive frame pairs and solves `du` per step. |
| `SETUP.md` | This document. |

Nothing in `vera/` core algorithms was changed beyond the import-compat shims above.

---

## Part 2 — Fresh-box runbook

### 0. System packages
```bash
sudo apt-get update
sudo apt-get install -y cmake build-essential ffmpeg \
    libegl1 libgles2 libglvnd0        # headless MuJoCo (EGL) + egl-probe build + video tooling
```
- `cmake`/`build-essential` — `egl-probe` (a robomimic dep) compiles a C++ extension.
- `libegl1`/`libgles2`/`libglvnd0` — provide the GLVND loader `libEGL.so.1`. The NVIDIA EGL *vendor*
  lib is usually present but the loader is not, which makes MuJoCo's EGL backend fail with
  `'NoneType' object has no attribute 'eglQueryString'`.

### 1. Python env
```bash
git clone https://github.com/<you>/VERA.git && cd VERA
uv venv .venv --python 3.11
export VIRTUAL_ENV=$PWD/.venv

# core + IDM + video planner (server side)
uv pip install -e ".[idm,video]"

# sim/eval (client side) — robomimic 0.5.0 / mimicgen 1.0.0 are GitHub-only, hence the git URLs
uv pip install -e ".[eval]"
```
Verify the core import chain:
```bash
.venv/bin/python -c "import vera, vera.policy, vera.idm, vera.server; print('vera ok')"
```

### 2. Checkpoints
```bash
pip install huggingface_hub   # or: uv pip install huggingface_hub
HF=.venv/bin/python

# Wave-1 trained artifacts (PushT + MimicGen)  (~3.8 GB)
$HF - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('sizhe-lester-li/VERA', local_dir='./vera-ckpts',
    allow_patterns=['pusht-dfot/*','pusht-idm/*','mimicgen-wan-1.3b/*','idm-mimicgen-37oa162u/*'])
PY

# Frozen Wan2.1 base (text-enc + VAE + base DiT) for the MimicGen WAN planner  (~17 GB)
$HF - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('Wan-AI/Wan2.1-T2V-1.3B', local_dir='./vera-ckpts/Wan2.1-T2V-1.3B')
PY

# MimicGen stack_d0 initial-states dataset  (~1.1 GB)
$HF - <<'PY'
from huggingface_hub import hf_hub_download
hf_hub_download('amandlek/mimicgen_datasets', 'core/stack_d0.hdf5', repo_type='dataset',
    local_dir='./vera-ckpts/mimicgen-data')
PY
```
- **VGGT-1B** (IDM backbone) and **DINOv2** auto-download from HF on first server start.
- `vera-ckpts/` and `third_party/` are gitignored — never committed.

### 3. Trackers (external; load lazily on first inference)
The IDM needs a pixel-motion source. PushT/IDM-on-video use **AllTracker**; MimicGen uses **CoTracker**
(AllTracker gives wrong-direction flow on MimicGen — don't switch it).
```bash
# AllTracker — not pip-installable; clone it. Inference needs only numpy/torch/torchvision/einops.
git clone https://github.com/aharley/alltracker third_party/alltracker
# It imports as a package `alltracker.*` AND uses bare `import utils/nets`, so BOTH dirs go on the path:
export VERA_ALLTRACKER_ROOT=$PWD/third_party
export PYTHONPATH=$PWD/third_party:$PWD/third_party/alltracker
```
CoTracker loads via `torch.hub.load("facebookresearch/co-tracker", ...)` on the MimicGen server's first
action step — needs internet, fetches once.

### 4. PushT (fast — loads in seconds)
```bash
export VERA_PUSHT_PLANNER_CKPT=$PWD/vera-ckpts/pusht-dfot/model.ckpt
export VERA_PUSHT_DYNAMICS_CKPT=$PWD/vera-ckpts/pusht-idm/model.ckpt
# --no-teacache: teacache imports the WAN/zmq stack, irrelevant for PushT's DFoT planner
.venv/bin/python -m vera.server.start_vera_server --embodiment pusht \
    --port 8820 --vis-port 8821 --no-teacache
```
Client: open `examples/pusht_dfot_stack.ipynb` → Run All.

### 5. MimicGen (WAN planner)
```bash
export VERA_WAN_CKPT_ROOT=$PWD/vera-ckpts/Wan2.1-T2V-1.3B
export VERA_MIMICGEN_CKPT_DIR=$PWD/vera-ckpts/mimicgen-wan-1.3b
.venv/bin/python -m vera.server.start_vera_server --embodiment mimicgen \
    --port 8800 --vis-port 8801 \
    --algo-config $VERA_MIMICGEN_CKPT_DIR/algo_config.yaml \
    --text "A robot arm stacks one block on top of another block"
```
The IDM auto-loads from `vera-ckpts/idm-mimicgen-37oa162u/` (this fork's change). Run the server in
**tmux** so it survives SSH disconnects:
```bash
tmux new -s vera   # launch inside; detach with Ctrl-b d, reattach: tmux attach -t vera
```
Client: open `examples/mimicgen_stack.ipynb` → Run All (cells **in order** — the config cell defines
`DATASET`/`DEMO_KEYS`).

### 6. Jupyter on a remote box
RunPod-style Jupyter (port 8888) runs the **system** Python, not the venv. Register the venv as a kernel:
```bash
uv pip install ipykernel
.venv/bin/python -m ipykernel install --user --name vera --display-name "VERA (.venv)"
```
Then add the rendering + tracker env to that kernel (`~/.local/share/jupyter/kernels/vera/kernel.json`):
```json
"env": {
  "MUJOCO_GL": "egl",
  "PYOPENGL_PLATFORM": "egl",
  "VERA_ALLTRACKER_ROOT": "/workspace/VERA/third_party",
  "PYTHONPATH": "/workspace/VERA/third_party:/workspace/VERA/third_party/alltracker"
}
```
Select **VERA (.venv)** in the notebook and restart the kernel.

### 7. Reaching the viewer over SSH
Only `:22` and Jupyter `:8888` are usually exposed. Tunnel the policy/viewer ports:
```bash
ssh root@<host> -p <port> -i ~/.ssh/id_ed25519 -L 8801:localhost:8801 -L 8800:localhost:8800
```
Then open `http://localhost:8801/` locally for the live viewer.

### 8. Bonus: run the IDM on your own video
```bash
.venv/bin/python examples/run_idm_on_video.py --video clip.mp4 --out actions.npy
# accepts .mp4 / a frames dir / an .npy [T,H,W,3]; --denormalize for physical action units
```

---

## Gotchas (why the above is exact)

- **torchmetrics drift.** ≥1.9 removed/renamed `NoTrainLpips`, `_valid_img`, and only exports `FID` when
  `torch-fidelity` is installed → `import vera` fails. Pinned to `1.4.0.post0` + `torch-fidelity`, plus
  defensive import shims.
- **setuptools ≥81** drops `pkg_resources`, which the DFoT package imports → pinned `setuptools<81`.
- **robomimic 0.5.0 / mimicgen 1.0.0 are not on PyPI** — install from git (baked into the `eval` extra).
- **⚠️ Server/client dependency conflict.** robomimic/mimicgen pin **old** `transformers` (4.41) /
  `diffusers` (0.11), which **conflict** with the WAN planner (needs `transformers>=4.51`). VERA is
  client/server precisely to decouple this. On a single venv: install the server deps, start the server,
  *then* install the `eval` extra for the client — and **don't restart the server in that venv** (WAN
  will fail to re-import). For a robust setup, use **two venvs**: one for the server (WAN/new
  transformers), one for the notebook client (sim/old transformers).
- **teacache needs zmq.** The WAN speedup path imports `zmq`; `pyzmq` is now in the `video` extra. PushT
  doesn't use it — pass `--no-teacache`.
- **Headless rendering** needs the GLVND `libEGL.so.1` loader + `MUJOCO_GL=egl` (see step 0/6). A harmless
  `EGLError` may print in MuJoCo's `GLContext.__del__` at teardown — ignore it.
- **CoTracker, not AllTracker, for MimicGen** — AllTracker produces wrong-direction flow there.
