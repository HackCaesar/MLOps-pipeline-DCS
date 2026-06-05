# Host-side DCS World capture

DCS World is a Windows-only simulator. All capture happens on the **Windows
host**, never inside Docker. The dockerized pipeline only consumes the dataset
the host produces.

1. **DCS World runs only on the Windows host.** It cannot run in a container.
2. **Docker never launches DCS.** The GPU/CPU services only train, evaluate,
   export and track.
3. **The Windows DCS Runner Agent is part of host-side capture.** It is a small
   HTTP service (`apps/windows_dcs_runner_agent`) running on the host that drives
   mission generation + capture and reports status.
4. **The dockerized pipeline talks to the agent over HTTP** at
   `host.docker.internal:8765` (`apps/dcs_capture/agent_client.py`). Compose adds
   `extra_hosts: ["host.docker.internal:host-gateway"]` so Linux daemons resolve it.
5. **Capture output goes straight to the canonical raw location**
   `STORAGE_ROOT/datasets/raw/{dataset_id}` — no intermediate `dcs_source` copy.
   (Direct-to-raw writing is wired up in Stage 2.)

## Components (`apps/dcs_capture_host/`)

| Dir | Role |
|---|---|
| `orchestrator/` | capture loop, camera planning, DCS export hooks, metadata |
| `scenario_builder/` | generate `.miz` missions (caucasus static neutral, smoke, bbox) |
| `projection/` | 3D→2D bbox projection (`model_dims.json`) |
| `exporters/` | COCO / YOLO label export |
| `validators/` | bbox geometry / refiners / dataset validation |
| `dcs_export/` | `Export.lua` / `Hooks.lua` installed into DCS Saved Games |
| `configs/` | capture configs (e.g. `caucasus_static_neutral_capture.yaml`) |
| `scripts/`, `tests/` | rebuild-coco, regression runner, projection tests |

## Import model (important)

The host code imports its own packages as **top-level** modules
(`from orchestrator import ...`, `from projection import ...`). This works when
the process cwd is `apps/dcs_capture_host`. The agent is configured accordingly:

```yaml
# configs/dcs_agent.yaml
subprocess:
  orchestrator_module: orchestrator.main
  orchestrator_cwd:    D:/MLOps_pipeline/apps/dcs_capture_host
```

Do **not** add an `__init__.py` at `apps/dcs_capture_host/` — that would shadow
these top-level imports. (A future cleanup could convert them to
`apps.dcs_capture_host.*` package imports, but it is intentionally deferred so
the working capture is not broken.)

## Installing on the Windows host (no Docker)

```powershell
# from D:\MLOps_pipeline
py -m venv .dcs_venv
.\.dcs_venv\Scripts\Activate.ps1
pip install -r requirements-dcs-capture.txt   # numpy, pyyaml, opencv-python, pydcs
# (equivalently: pip install -e ".[dcs-capture]")

# install the DCS export hooks (copy apps/dcs_capture_host/dcs_export/*.lua into
# your DCS "Saved Games\...\Scripts" folder — see that dir's README)

# start the agent
python -m apps.windows_dcs_runner_agent.agent --config configs/dcs_agent.yaml
```
