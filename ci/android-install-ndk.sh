#!/usr/bin/env sh
# Copyright 2016 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

set -ex

curl --retry 5 -O \
     https://dl.google.com/android/repository/android-ndk-r25b-linux.zip
unzip -q android-ndk-r25b-linux.zip
mv android-ndk-r25b "/android/ndk-${1}"
rm -rf ./android-ndk-r25b-linux.zip
