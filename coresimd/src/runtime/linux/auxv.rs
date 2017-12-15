//! ELF Auxiliary Vector
//!
//! The auxiliary vector is a memory region in a running ELF program's stack
//! composed of (key: usize, value: usize) pairs.
//!
//! The keys used in the aux vector are platform dependent. For Linux, they are
//! defined in [linux/auxvec.h][auxvec_h]. The hardware capabilities of a given
//! CPU can be queried with the  `AT_HWCAP` and `AT_HWCAP2` keys.
//!
//! There is no perfect way of reading the auxiliary vector.
//!
//! - `coresimd`: if `getauxval` is available, `coresimd` will try to use it.
//! - `stdsimd`: if `getauxval` is not available, or it returns that a feature
//! is not supported, it will try to read `/proc/self/auxv`. If that fails,
//! it   will try to read `/proc/cpuinfo`.
//!
//! For more information about when `getauxval` is available check the great
//! [`auxv` crate documentation][auxv_docs].
//!
//! [auxvec_h]: https://github.com/torvalds/linux/blob/master/include/uapi/linux/auxvec.h
//! [auxv_docs]: https://docs.rs/auxv/0.3.3/auxv/

/// CPU Hardware capabilities (bitfield).
pub const AT_HWCAP: usize = 16;
/// CPU Hardware capabilities 2 (bitfield).
pub const AT_HWCAP2: usize = 26;

/// Cache HWCAP entries of the ELF Auxiliary Vector
#[cfg(any(target_arch = "arm", target_arch = "powerpc64"))]
#[derive(Debug, Copy, Clone)]
pub struct AuxVec {
    pub hwcap: usize,
    pub hwcap2: usize,
}

/// Cache HWCAP entries of the ELF Auxiliary Vector
#[cfg(target_arch = "aarch64")]
#[derive(Debug, Copy, Clone)]
pub struct AuxVec {
    pub hwcap: usize,
}

pub mod libc {

    use super::*;

    extern "C" {
        fn getauxval_wrapper(key: usize, success: *mut usize) -> i32;
    }

    fn getauxval(key: usize) -> Result<usize, ()> {
        let mut auxv = 0;
        unsafe {
            match getauxval_wrapper(key, &mut auxv) {
                1 => Ok(auxv),
                _ => Err(()),
            }
        }
    }

    /// Computes the entries of the Auxiliary Vector cache by
    /// calling libc's `getauxval(3)`.
    pub fn auxv() -> Result<AuxVec, ()> {
        if let Ok(hwcap) = getauxval(AT_HWCAP) {
            #[cfg(target_arch = "aarch64")]
            {
                return Ok(AuxVec { hwcap });
            }
            #[cfg(any(target_arch = "arm", target_arch = "powerpc64"))]
            {
                if let Ok(hwcap2) = getauxval(AT_HWCAP2) {
                    return Ok(AuxVec { hwcap, hwcap2 });
                }
            }
        }
        Err(())
    }

    #[cfg(test)]
    mod tests {
        extern crate auxv as auxv_crate;
        use super::*;

        // Reads the Auxiliary Vector key from getauxval()
        // using the auxv crate.
        fn auxv_crate_get(key: usize) -> Option<usize> {
            use self::auxv_crate::AuxvType;
            use self::auxv_crate::getauxval::Getauxval;
            let q = auxv_crate::getauxval::NativeGetauxval {};
            match q.getauxval(key as AuxvType) {
                Ok(v) => Some(v as usize),
                Err(_) => None,
            }
        }

        #[test]
        fn auxv_dump() {
            if let Ok(auxvec) = auxv() {
                println!("{:?}", auxvec);
            } else {
                println!("reading /proc/self/auxv failed!");
            }
        }

        #[cfg(any(target_arch = "arm", target_arch = "powerpc64"))]
        #[test]
        fn auxv_crate() {
            let v = auxv();
            if let Some(hwcap) = auxv_crate_get(AT_HWCAP) {
                assert_eq!(v.unwrap().hwcap, hwcap);
            }
            if let Some(hwcap2) = auxv_crate_get(AT_HWCAP2) {
                assert_eq!(v.unwrap().hwcap2, hwcap2);
            }
        }

        #[cfg(target_arch = "aarch64")]
        #[test]
        fn auxv_crate() {
            let v = auxv();
            if let Some(hwcap) = auxv_crate_get(AT_HWCAP) {
                assert_eq!(v.unwrap().hwcap, hwcap);
            }
        }
    }
}
