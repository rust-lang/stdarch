FROM ubuntu:25.10
RUN apt-get update && apt-get install -y --no-install-recommends \
  gcc-multilib \
  libc6-dev \
  file \
  make \
  ca-certificates \
  wget \
  xz-utils

RUN wget http://ci-mirrors.rust-lang.org/stdarch/sde-external-10.5.0-2026-01-13-lin.tar.xz -O sde.tar.xz
RUN mkdir intel-sde
RUN tar -xJf sde.tar.xz --strip-components=1 -C intel-sde
ENV CARGO_TARGET_I686_UNKNOWN_LINUX_GNU_RUNNER="/intel-sde/sde \
            -cpuid-in /checkout/ci/cpuid.def \
            -rtm-mode full -tsx --"
# SDE doesn't support AMD extensions
# FIXME: find a way to test these
ENV STDARCH_TEST_SKIP_FEATURE="sse4a,tbm,xop"
