//! Reads ELF Auxiliary Vector

pub use coresimd::__vendor_runtime::__runtime::linux::auxv::libc;

/// Reads the ELF Auxiliary Vector from `/proc/self/auxv`.
pub mod proc_self {
    pub use coresimd::__vendor_runtime::__runtime::linux::auxv::{AT_HWCAP2,
                                                                 AuxVec,
                                                                 AT_HWCAP};
    use std::mem;

    /// Tries to read the ELF Auxiliary Vector from `/proc/self/auxv`.
    ///
    /// Errors if the file cannot be read. If a component of the auxvector
    /// cannot be read, all the bits in its bitset are set to zero.
    pub fn auxv() -> Result<AuxVec, ()> {
        auxv_from_file("/proc/self/auxv")
    }

    fn auxv_from_file(file: &str) -> Result<AuxVec, ()> {
        use std::io::Read;
        let mut file = ::std::fs::File::open(file).or_else(|_| Err(()))?;

        // See https://github.com/torvalds/linux/blob/v3.19/include/uapi/linux/auxvec.h
        //
        // The auxiliary vector contains at most 32 (key,value) fields: from
        // `AT_EXECFN = 31` to `AT_NULL = 0`. That is, a buffer of
        // 2*32 `usize` elements is enough to read the whole vector.
        let mut buf = [0usize; 64];
        {
            let raw: &mut [u8; 64 * mem::size_of::<usize>()] =
                unsafe { mem::transmute(&mut buf) };
            file.read(raw).or_else(|_| Err(()))?;
        }
        auxv_from_buf(&buf)
    }

    fn auxv_from_buf(buf: &[usize; 64]) -> Result<AuxVec, ()> {
        #[cfg(target_arch = "aarch64")]
        {
            for el in buf.chunks(2) {
                match el[0] {
                    AT_HWCAP => return Ok(AuxVec { hwcap: el[1] }),
                    _ => println!("reading auxv: ({},{})", el[0], el[1]),
                }
            }
            return Ok(AuxVec { hwcap: 0 });
        }

        #[cfg(any(target_arch = "arm", target_arch = "powerpc64"))]
        {
            let mut hwcap = 0;
            let mut hwcap2 = 0;
            for el in buf.chunks(2) {
                match el[0] {
                    AT_HWCAP => hwcap = el[1],
                    AT_HWCAP2 => hwcap2 = el[1],
                    _ => println!("reading auxv: ({},{})", el[0], el[1]),
                }
            }
            return Ok(AuxVec { hwcap, hwcap2 });
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[cfg(target_os = "linux")]
        #[test]
        fn auxv_dump() {
            if let Ok(auxvec) = auxv() {
                println!("{:?}", auxvec);
            } else {
                println!("reading /proc/self/auxv failed!");
            }
        }
    }
}
