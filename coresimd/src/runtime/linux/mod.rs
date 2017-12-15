//! Run-time feature detection for ARM and PowerPC64 on Linux.

#[cfg(target_arch = "arm")]
mod arm;

#[cfg(target_arch = "aarch64")]
mod aarch64;

#[cfg(target_arch = "powerpc64")]
mod powerpc64;

pub mod auxv;

/// Detects CPU features in `coresimd`.
pub fn detect_features() -> usize {
    // Try to read the ELF Auxiliary Vector using libc's getauxval:
    if let Ok(v) = auxv::libc::auxv() {
        return super::arch::detect_features(v);
    }
    // Otherwise all features are disabled
    0
}
