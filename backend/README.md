# AL-MedLit Backend

FastAPI backend package for AL-MedLit. Runtime dependencies are declared in `pyproject.toml`; `requirements.txt` and `requirements-dev.txt` are thin wrappers retained for Docker and convenience installs.

Public self-registration is disabled by default. Instance administrators can
invite users or explicitly enable registration through instance policy or
`AL_MEDLIT_ALLOW_SELF_REGISTRATION=true`. Provision the first administrator with
`python -m scripts.bootstrap_admin` after configuring a bootstrap password.
New workspaces receive a 10 GiB artifact
allocation, configurable with `AL_MEDLIT_DEFAULT_WORKSPACE_ARTIFACT_QUOTA_BYTES`.
Workspace managers can lower allocations; only instance administrators can
increase or remove them. Existing allocations are preserved on upgrade.

When a TLS reverse proxy fronts the bundled frontend, set
`AL_MEDLIT_TRUSTED_PROXY_CIDRS` to its source addresses or CIDRs, separated by
spaces. The proxy must append or overwrite `X-Forwarded-For` with the real client
address. nginx then keys authentication limits on that client. Empty is the
default and trusts no forwarded addresses; never trust arbitrary client networks.

Login checks credentials before returning a name-based failure threshold. This
keeps correct passwords usable under attack and gives known and unknown names
the same failure responses. It still verifies each submitted password; nginx's
source-address limiter controls attempt rate.

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

Training workers use prefetch one and concurrency one. CPU workers can set
`AL_MEDLIT_WORKER_CONCURRENCY` to a positive integer. Accelerator workers require
concurrency one and share an exclusive lock under `AL_MEDLIT_LOCAL_ATTEMPT_ROOT`
across PEFT and QLoRA profiles. Keep that root shared when workers use the same
GPU; the bundled Compose services already do. Use separate roots only for
separate accelerator allocations. Waiting for admission remains subject to
the training deadline and cancellation checks.

Training runs are dispatched to the queue selected by their verified
execution environment. Workers compose the pinned label layers, enforce split
governance, invoke the matching lazy trainer plugin, and publish an immutable
model package and `ModelVersion`. Custom plugin contracts remain extensible, but
execution currently fails closed unless the recipe key is registered in the
trusted recipe catalog, which supplies required model-family and artifact-format
metadata.

Training tasks use a POSIX watchdog and a separate compute process. The watchdog
checks cancellation every five seconds and allows ten seconds for termination
before it kills the compute process group. It also stops compute if the task
process disappears. A cancelled status records the cancellation request before
process cleanup finishes.

The API and worker environments accept these limits:

| Variable | Default | Meaning |
|---|---|---|
| `AL_MEDLIT_TRAINING_MAX_RUNTIME_SECONDS` | `86400` | Maximum compute duration, 24 hours. |
| `AL_MEDLIT_TRAINING_MAX_QUEUE_WAIT_SECONDS` | `604800` | Maximum queue wait, seven days. |
| `AL_MEDLIT_TRAINING_SUPERVISOR_POLL_SECONDS` | `5` | Interval between worker lease checks. |
| `AL_MEDLIT_TRAINING_TERMINATION_GRACE_SECONDS` | `10` | Grace before forced termination. |

Training reconciliation runs every minute through Celery beat or the eager API.
It closes abandoned queues and stale workers, including launches with expired
or unavailable artifact reservations.

Protected evaluation validates its inputs and supported metrics before training.
Failed or cancelled runs archive their new, unreferenced candidates. Legal
holds, explicit manager pins, and package dependencies preserve protection.
Physical quota is reclaimed after the seven-day grace, manager purge approval,
and garbage collection. Legacy candidates with ambiguous pin provenance remain
protected and require manager review.

In the default eager deployment, the API retries queued orphaned-object
deletions every fifteen minutes while it is running. Worker-backed deployments
run the same cleanup through Celery beat. Failed attempts remain in the database
with retry backoff, so a temporary storage outage does not discard the backlog.

SSH/Slurm login hosts also need the util-linux `flock` command. Submission uses
it to serialize the job-ID check and `sbatch` call. Keep `.submission.lock` in
the shared job directory; its lock is released when the submission shell exits.
Slurm profiles bound control commands to 60 seconds and transfers to 900 seconds
by default. Set `command_timeout_seconds` and `transfer_timeout_seconds` in the
profile when larger transfers need more time. SSH/scp also use a ten-second
connection timeout and server keepalives.
