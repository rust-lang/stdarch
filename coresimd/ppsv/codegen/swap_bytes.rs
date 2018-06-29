//! swap bytes horizontally

use coresimd::simd::*;

pub(crate) trait SwapBytes {
    fn swap_bytes(self) -> Self;
}

macro_rules! impl_swap_bytes {
    ($vec8:ident, $shuf:ident, $indices:expr, $id:ident) => (
        impl SwapBytes for $id {
            fn swap_bytes(self) -> Self {
                let vec8 = $vec8::from_bits(self);
                let shuffled: $vec8 = unsafe { $shuf(vec8, vec8, $indices) };
                $id::from_bits(shuffled)
            }
        }
    );

    // bulk impl for a vector width
    ($vec8:ident, $shuf:ident, $indices:expr, $($id:ident,)+) => ($(
        impl_swap_bytes! { $vec8, $shuf, $indices, $id }
    )+);
}

impl_swap_bytes! {
    u8x2,
    simd_shuffle2,
    [1, 0],
    u8x2, i8x2,
}

impl_swap_bytes! {
    u8x4,
    simd_shuffle4,
    [3, 2, 1, 0],
    u8x4, i8x4,
    u16x2, i16x2,
}

impl_swap_bytes! {
    u8x8,
    simd_shuffle8,
    [7, 6, 5, 4, 3, 2, 1, 0],
    u8x8, i8x8,
    u16x4, i16x4,
    u32x2, i32x2,
}

impl_swap_bytes! {
    u8x16,
    simd_shuffle16,
    [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
    u8x16, i8x16,
    u16x8, i16x8,
    u32x4, i32x4,
    u64x2, i64x2,
}

impl_swap_bytes! {
    u8x32,
    simd_shuffle32,
    [
        31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16,
        15, 14, 13, 12, 11, 10, 9,  8,  7,  6,  5,  4,  3,  2,  1,  0,
    ],
    u8x32, i8x32,
    u16x16, i16x16,
    u32x8, i32x8,
    u64x4, i64x4,
}

impl_swap_bytes! {
    u8x64,
    simd_shuffle64,
    [
        63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48,
        47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32,
        31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16,
        15, 14, 13, 12, 11, 10, 9,  8,  7,  6,  5,  4,  3,  2,  1,  0,
    ],
    u8x64, i8x64,
    u16x32, i16x32,
    u32x16, i32x16,
    u64x8, i64x8,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem;

    // testing larger vectors is less simple
    #[test]
    #[cfg(feature = "simd_support")]
    fn swap_bytes_128() {
        let x: u128 = 0x2d99787926d46932a4c1f32680f70c55;
        let expected = x.swap_bytes();

        let vec: u8x16 = unsafe { mem::transmute(x) };
        let actual = unsafe { mem::transmute(vec.swap_bytes()) };

        assert_eq!(expected, actual);
    }

    #[test]
    #[cfg(feature = "simd_support")]
    fn swap_bytes_64() {
        let x: u64 = 0x2d99787926d46932;
        let expected = x.swap_bytes();

        let vec: u8x8 = unsafe { mem::transmute(x) };
        let actual = unsafe { mem::transmute(vec.swap_bytes()) };

        assert_eq!(expected, actual);
    }

    #[test]
    #[cfg(feature = "simd_support")]
    fn swap_bytes_32() {
        let x: u32 = 0x2d997872;
        let expected = x.swap_bytes();

        let vec: u8x4 = unsafe { mem::transmute(x) };
        let actual = unsafe { mem::transmute(vec.swap_bytes()) };

        assert_eq!(expected, actual);
    }

    #[test]
    #[cfg(feature = "simd_support")]
    fn swap_bytes_16() {
        let x: u16 = 0x2d99;
        let expected = x.swap_bytes();

        let vec: u8x2 = unsafe { mem::transmute(x) };
        let actual = unsafe { mem::transmute(vec.swap_bytes()) };

        assert_eq!(expected, actual);
    }
}
