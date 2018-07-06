//! Horizontal swap bytes.

macro_rules! impl_swap_bytes {
    ($id:ident) => {
        impl $id {
            /// Reverses the byte order of the vector.
            #[inline]
            pub fn swap_bytes(self) -> Self {
                unsafe { super::codegen::swap_bytes::SwapBytes::swap_bytes(self) }
            }

            /// Converts self to little endian from the target's endianness.
            ///
            /// On little endian this is a no-op. On big endian the bytes are swapped.
            #[inline]
            pub fn to_le(self) -> Self {
                #[cfg(target_endian = "little")]
                {
                    self
                }
                #[cfg(not(target_endian = "little"))]
                {
                    self.swap_bytes()
                }
            }
        }
    };
}

#[cfg(test)]
macro_rules! test_swap_bytes {
    (u8x2) => {
        #[test]
        fn swap_bytes() {
            use coresimd::simd::u8x2;
            use std::mem;

            let x: u16 = 0x2d99;
            let expected = x.swap_bytes();

            let vec: u8x2 = unsafe { mem::transmute(x) };
            let actual = unsafe { mem::transmute(vec.swap_bytes()) };

            assert_eq!(expected, actual);
        }
    };
    (u8x4) => {
        #[test]
        fn swap_bytes() {
            use coresimd::simd::u8x4;
            use std::mem;

            let x: u32 = 0x2d997872;
            let expected = x.swap_bytes();

            let vec: u8x4 = unsafe { mem::transmute(x) };
            let actual = unsafe { mem::transmute(vec.swap_bytes()) };

            assert_eq!(expected, actual);
        }
    };
    (u8x8) => {
        #[test]
        fn swap_bytes() {
            use coresimd::simd::u8x8;
            use std::mem;

            let x: u64 = 0x2d99787926d46932;
            let expected = x.swap_bytes();

            let vec: u8x8 = unsafe { mem::transmute(x) };
            let actual = unsafe { mem::transmute(vec.swap_bytes()) };

            assert_eq!(expected, actual);
        }
    };
    (u8x16) => {
        #[test]
        fn swap_bytes() {
            use coresimd::simd::u8x16;
            use std::mem;

            let x: u128 = 0x2d99787926d46932a4c1f32680f70c55;
            let expected = x.swap_bytes();

            let vec: u8x16 = unsafe { mem::transmute(x) };
            let actual = unsafe { mem::transmute(vec.swap_bytes()) };

            assert_eq!(expected, actual);
        }
    };
    // testing larger vectors is less simple
    ($id:ident) => {};
}
