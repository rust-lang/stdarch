//! Run-time feature detection on Linux

mod auxvec;

#[cfg(feature = "std_detect_file_io")]
mod cpuinfo;

cfg_if::cfg_if! {
    if #[cfg(target_arch = "aarch64")] {
        mod aarch64;
        pub(crate) use self::aarch64::detect_features;
    } else if #[cfg(target_arch = "arm")] {
        mod arm;
        pub(crate) use self::arm::detect_features;
    } else  if #[cfg(any(target_arch = "mips", target_arch = "mips64"))] {
        mod mips;
        pub(crate) use self::mips::detect_features;
    } else if #[cfg(any(target_arch = "powerpc", target_arch = "powerpc64"))] {
        mod powerpc;
        pub(crate) use self::powerpc::detect_features;
    } else {
        use crate::detect::cache;
        /// Performs run-time feature detection.
        pub(crate) fn detect_features() -> cache::Initializer {
            cache::Initializer::default()
        }
    }
}
