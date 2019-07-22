#!/usr/bin/env sh

# Small script to run tests for a target (or all targets) inside all the
# respective docker images.

set -ex

run() {
    target=$(echo "${1}" | sed 's/-emulated//')
    echo "Building docker container for TARGET=${1}"
    case ${target} in
         mipsisa*)
            export MOUNT_XARGO="--volume ${HOME}/.xargo:/xargo-h"
            export ENV_XARGO="--env XARGO_HOME=/xargo-h"
            ;;
    esac
    docker build -t stdarch -f "ci/docker/${1}/Dockerfile" ci/
    mkdir -p target
    echo "Running docker"
    # shellcheck disable=SC2016
    # shellcheck disable=SC2086
    docker run \
      --rm \
      --user "$(id -u)":"$(id -g)" \
      ${MOUNT_XARGO} \
      ${ENV_XARGO} \
      --env CARGO_HOME=/cargo \
      --env CARGO_TARGET_DIR=/checkout/target \
      --env TARGET="${target}" \
      --env STDARCH_TEST_EVERYTHING \
      --env STDARCH_ASSERT_INSTR_IGNORE \
      --env STDARCH_DISABLE_ASSERT_INSTR \
      --env NOSTD \
      --env NORUN \
      --env RUSTFLAGS \
      --env STDARCH_TEST_NORUN \
      --volume "$(dirname "$(dirname "$(command -v cargo)")")":/cargo \
      --volume "$(rustc --print sysroot)":/rust:ro \
      --volume "$(pwd)":/checkout \
      --volume "$(pwd)"/target:/checkout/target \
      --init \
      --workdir /checkout \
      --privileged \
      stdarch \
      sh -c "HOME=/tmp PATH=\$PATH:/rust/bin exec ci/run.sh ${1}"
}

if [ -z "$1" ]; then
  for d in ci/docker/*; do
    run "${d}"
  done
else
  run "${1}"
fi
