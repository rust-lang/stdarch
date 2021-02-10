FROM ubuntu:20.04
RUN apt-get update && apt-get install -y --no-install-recommends \
  gcc \
  libc6-dev \
  file \
  make \
  ca-certificates \
  wget \
  bzip2

RUN wget https://software.intel.com/content/dam/develop/external/us/en/documents/downloads/sde-external-8.59.0-2020-10-05-lin.tar.bz2
RUN tar -xjf sde-external-8.59.0-2020-10-05-lin.tar.bz2
ENV CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER="/sde-external-8.59.0-2020-10-05-lin/sde64 --help && /sde-external-8.59.0-2020-10-05-lin/sde64 -rtm_mode full --"
