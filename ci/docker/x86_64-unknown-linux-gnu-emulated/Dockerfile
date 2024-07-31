FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends \
  gcc \
  libc6-dev \
  file \
  make \
  ca-certificates \
  qemu-system

RUN lscpu
RUN chown root:kvm /dev/kvm
ENV CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER="qemu-system-x86_64 -enable-kvm -cpu max -nographic -kernel"
