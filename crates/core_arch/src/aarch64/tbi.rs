/// Provides additional methods on pointers to get and set any values set in the top byte of the pointer,
/// as per AArch64's Top-Byte Ignore (TBI) feature.
#[unstable(feature = "stdarch_aarch64_tbi", issue = "none")]
pub trait TBIPointer: Sized {
    /// Returns a new pointer with the top byte set to a given value, invalidating the original pointer.
    ///
    /// Continuing to use the pointer passed in to this function is Undefined Behavior; you should replace it with the returned pointer instead.
    #[unstable(feature = "stdarch_aarch64_tbi", issue = "none")]
    fn with_top_byte(self, b: u8) -> Self;

    /// Returns the value (if any) stored in the top byte of the pointer.
    #[unstable(feature = "stdarch_aarch64_tbi", issue = "none")]
    fn top_byte(&self) -> u8;
}

macro_rules! tbi_ptr_impl {
    () => {
        fn with_top_byte(self, b: u8) -> Self {
            // We can't actually call `realloc` here as we don't have `std`, and the pointer may not be valid for `realloc` anyway (if the value's stored on the stack, for example).
            let addr = self.addr() & 0x00ffffffffffffff;
            let p = addr | ((b as usize) << 56);
            self.with_addr(p)
        }

        fn top_byte(&self) -> u8 {
            (self.addr() >> 56) as u8
        }
    };
}

#[unstable(feature = "stdarch_aarch64_tbi", issue = "none")]
impl<T> TBIPointer for *const T {
    tbi_ptr_impl!();
}

#[unstable(feature = "stdarch_aarch64_tbi", issue = "none")]
impl<T> TBIPointer for *mut T {
    tbi_ptr_impl!();
}

#[unstable(feature = "stdarch_aarch64_tbi", issue = "none")]
impl<T> TBIPointer for *const [T] {
    tbi_ptr_impl!();
}

#[unstable(feature = "stdarch_aarch64_tbi", issue = "none")]
impl<T> TBIPointer for *mut [T] {
    tbi_ptr_impl!();
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn tbi_const() {
        let value: u32 = 10;
        let mut address = &value as *const u32;
        address = address.with_top_byte(0x80);
        assert_eq!(address.top_byte(), 0x80);
        assert_eq!(unsafe { *address }, 10);
    }

    #[test]
    fn tbi_mut() {
        let mut value: u32 = 10;
        let mut address = &mut value as *mut u32;
        address = address.with_top_byte(0x80);
        assert_eq!(address.top_byte(), 0x80);
        assert_eq!(unsafe { *address }, 10);
        unsafe { *address = 255 };
        assert_eq!(unsafe { *address }, 255);
    }

    #[test]
    fn tbi_const_array() {
        let value: [u32; 4] = [10, 255, 65535, 0xffffffff];
        let mut address = &value as *const [u32; 4];
        address = address.with_top_byte(0x80);
        assert_eq!(address.top_byte(), 0x80);
        assert_eq!(unsafe { *address }, [10, 255, 65535, 0xffffffff]);
    }

    #[test]
    fn tbi_mut_array() {
        let mut value: [u32; 4] = [10, 255, 65535, 0xffffffff];
        let mut address = &mut value as *mut [u32; 4];
        address = address.with_top_byte(0x80);
        assert_eq!(address.top_byte(), 0x80);
        assert_eq!(unsafe { *address }, [10, 255, 65535, 0xffffffff]);
        unsafe { (*address)[0] = 25 };
        assert_eq!(unsafe { *address }, [25, 255, 65535, 0xffffffff]);
    }
}
