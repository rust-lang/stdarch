//! Run-time feature detection on Linux

#[cfg(not(target_arch = "mips64"))]
pub mod cpuinfo;

#[cfg(any(target_arch = "aarch64",
          target_arch = "arm",
          target_arch = "powerpc64",
          target_arch = "mips64"))]
pub mod auxvec;
