FROM japaric/armv7-unknown-linux-gnueabihf:latest
RUN apt-get update && apt-get install -y --no-install-recommends \
  libc6-dev-armhf-cross \
  file
ENV OBJDUMP=arm-linux-gnueabihf-objdump