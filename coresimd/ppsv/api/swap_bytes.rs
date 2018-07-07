//! Horizontal swap bytes.

macro_rules! impl_swap_bytes {
    ($id:ident) => {
        impl $id {
            /// Reverses the byte order of the vector.
            #[inline]
            pub fn swap_bytes(self) -> Self {
                unsafe {
                    super::codegen::swap_bytes::SwapBytes::swap_bytes(self)
                }
            }

            /// Converts self to little endian from the target's endianness.
            ///
            /// On little endian this is a no-op. On big endian the bytes are
            /// swapped.
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

            /// Converts self to big endian from the target's endianness.
            ///
            /// On big endian this is a no-op. On little endian the bytes are
            /// swapped.
            #[inline]
            pub fn to_be(self) -> Self {
                #[cfg(target_endian = "big")]
                {
                    self
                }
                #[cfg(not(target_endian = "big"))]
                {
                    self.swap_bytes()
                }
            }
        }
    };
}

macro_rules! impl_test {
    ($id:ident, $vec8:ident) => {
        use coresimd::simd::$id as Vector;
        use coresimd::simd::$vec8;
        use coresimd::simd::FromBits;

        macro_rules! test_swap {
            ($func: ident) => {
                let mut x = $vec8::splat(0);
                let mut exp = $vec8::splat(0);
                for i in 0..$vec8::lanes() {
                    x = x.replace(i, i as u8);
                    exp = exp.replace(i, ($vec8::lanes() - 1 - i) as u8);
                }
                let actual = Vector::from_bits(x).$func();

                assert_eq!(Vector::from_bits(exp), actual);
            };
        }

        macro_rules! test_no_swap {
            ($func: ident) => {
                let mut x = $vec8::splat(0);
                for i in 0..$vec8::lanes() {
                    x = x.replace(i, i as u8);
                }

                let exp = Vector::from_bits(x);
                let actual = Vector::from_bits(x).$func();

                assert_eq!(exp, actual);
            };
        }

        #[test]
        fn swap_bytes() {
            test_swap!(swap_bytes);
        }

        #[test]
        fn to_le() {
            #[cfg(target_endian = "little")]
            {
                test_no_swap!(to_le);
            }
            #[cfg(not(target_endian = "little"))]
            {
                test_swap!(to_le);
            }
        }

        #[test]
        fn to_be() {
            #[cfg(target_endian = "big")]
            {
                test_no_swap!(to_be);
            }
            #[cfg(not(target_endian = "big"))]
            {
                test_swap!(to_be);
            }
        }
    };
}

#[cfg(test)]
macro_rules! test_swap_bytes {
    (u8x2)   => { impl_test! { u8x2,   u8x2  } };
    (i8x2)   => { impl_test! { i8x2,   u8x2  } };

    (u8x4)   => { impl_test! { u8x4,   u8x4  } };
    (i8x4)   => { impl_test! { i8x4,   u8x4  } };
    (u16x2)  => { impl_test! { u16x2,  u8x4  } };
    (i16x2)  => { impl_test! { i16x2,  u8x4  } };

    (u8x8)   => { impl_test! { u8x8,   u8x8  } };
    (i8x8)   => { impl_test! { i8x8,   u8x8  } };
    (u16x4)  => { impl_test! { u16x4,  u8x8  } };
    (i16x4)  => { impl_test! { i16x4,  u8x8  } };
    (u32x2)  => { impl_test! { u32x2,  u8x8  } };
    (i32x2)  => { impl_test! { i32x2,  u8x8  } };

    (u8x16)  => { impl_test! { u8x16,  u8x16 } };
    (i8x16)  => { impl_test! { i8x16,  u8x16 } };
    (u16x8)  => { impl_test! { u16x8,  u8x16 } };
    (i16x8)  => { impl_test! { i16x8,  u8x16 } };
    (u32x4)  => { impl_test! { u32x4,  u8x16 } };
    (i32x4)  => { impl_test! { i32x4,  u8x16 } };
    (u64x2)  => { impl_test! { u64x2,  u8x16 } };
    (i64x2)  => { impl_test! { i64x2,  u8x16 } };

    (u8x32)  => { impl_test! { u8x32,  u8x32 } };
    (i8x32)  => { impl_test! { i8x32,  u8x32 } };
    (u16x16) => { impl_test! { u16x16, u8x32 } };
    (i16x16) => { impl_test! { i16x16, u8x32 } };
    (u32x8)  => { impl_test! { u32x8,  u8x32 } };
    (i32x8)  => { impl_test! { i32x8,  u8x32 } };
    (u64x4)  => { impl_test! { u64x4,  u8x32 } };
    (i64x4)  => { impl_test! { i64x4,  u8x32 } };

    (u8x64)  => { impl_test! { u8x64,  u8x64 } };
    (i8x64)  => { impl_test! { i8x64,  u8x64 } };
    (u16x32) => { impl_test! { u16x32, u8x64 } };
    (i16x32) => { impl_test! { i16x32, u8x64 } };
    (u32x16) => { impl_test! { u32x16, u8x64 } };
    (i32x16) => { impl_test! { i32x16, u8x64 } };
    (u64x8)  => { impl_test! { u64x8,  u8x64 } };
    (i64x8)  => { impl_test! { i64x8,  u8x64 } };
}
