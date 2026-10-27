#!/bin/sh
set -eu

profile=${1:?runtime profile is required}
shift
case "$profile" in
    classical-cpu) digest_variable=AL_MEDLIT_CLASSICAL_CPU_IMAGE_DIGEST ;;
    torch-cpu) digest_variable=AL_MEDLIT_TORCH_CPU_IMAGE_DIGEST ;;
    transformer-cpu) digest_variable=AL_MEDLIT_TRANSFORMER_CPU_IMAGE_DIGEST ;;
    peft-accelerator) digest_variable=AL_MEDLIT_PEFT_ACCELERATOR_IMAGE_DIGEST ;;
    qlora-cuda) digest_variable=AL_MEDLIT_QLORA_CUDA_IMAGE_DIGEST ;;
    *) printf 'Unknown runtime profile: %s\n' "$profile" >&2; exit 1 ;;
esac

docker compose "$@" --profile lab --profile "$profile" build
digest=$(docker image inspect --format '{{.Id}}' "al-medlit-worker-$profile:local")
env "$digest_variable=$digest" docker compose "$@" --profile lab --profile "$profile" up -d
