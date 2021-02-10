FROM ubuntu:20.04
RUN apt-get update && apt-get install -y --no-install-recommends \
  gcc \
  libc6-dev \
  file \
  make \
  ca-certificates \
  wget \
  bzip2

#RUN wget https://software.intel.com/content/dam/develop/external/us/en/documents/downloads/sde-external-8.59.0-2020-10-05-lin.tar.bz2
#RUN tar -xjf sde-external-8.59.0-2020-10-05-lin.tar.bz2
#RUN ls -l /sde-external-8.59.0-2020-10-05-lin/sde64 > /log.txt 2>&1
#RUN sha1sum /sde-external-8.59.0-2020-10-05-lin/sde64 >> /log.txt 2>&1
#RUN /sde-external-8.59.0-2020-10-05-lin/sde64 -rtm_mode full -- cat /proc/cpuinfo >> log.txt 2>&1
#RUN cat /log.txt
#ENV CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER="/sde-external-8.59.0-2020-10-05-lin/sde64 -rtm_mode full --"
RUN wget https://github.com/gnzlbg/intel_sde/raw/master/sde-external-8.35.0-2019-03-11-lin.tar.bz2
RUN tar -xjf sde-external-8.35.0-2019-03-11-lin.tar.bz2
RUN /sde-external-8.35.0-2019-03-11-lin/sde64 -rtm_mode full -- cat /proc/cpuinfo
ENV CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER="/sde-external-8.35.0-2019-03-11-lin/sde64 -rtm_mode full --"
