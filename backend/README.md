# AL-MedLit Backend

FastAPI backend package for AL-MedLit. Runtime dependencies are declared in `pyproject.toml`; `requirements.txt` and `requirements-dev.txt` are thin wrappers retained for Docker and convenience installs.

The default install contains API and orchestration dependencies only. Training
stacks are opt-in extras:

| Runtime | Extra |
|---|---|
| Classical CPU | `runtime-classical-cpu` |
| PyTorch CPU | `runtime-torch-cpu` |
| Transformers CPU | `runtime-transformer-cpu` |
| LoRA accelerator | `runtime-peft-accelerator` |
| QLoRA CUDA | `runtime-qlora-cuda` |

CPU Torch extras are locked against PyTorch's CPU-only package index. CUDA
packages are resolved only for the accelerator extras. Worker containers start
through `python -m scripts.start_worker`, which runs the runtime preflight before
consuming their dedicated queue. Startup requires the matching immutable
`AL_MEDLIT_*_IMAGE_DIGEST`, and the resulting image-bound report can be supplied
when an administrator creates a named execution environment.

Training runs are dispatched to the queue selected by their verified
execution environment. Workers compose the pinned label layers, enforce split
governance, invoke the matching lazy trainer plugin, and publish an immutable
model package and `ModelVersion`. Custom plugin contracts remain extensible, but
execution currently fails closed unless the recipe key is registered in the
trusted recipe catalog, which supplies required model-family and artifact-format
metadata.
