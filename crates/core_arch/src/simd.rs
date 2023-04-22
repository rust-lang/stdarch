//! Internal `#[repr(simd)]` types

#![allow(non_camel_case_types)]

macro_rules! simd_ty {
    ($id:ident: [$ety:ident; $ecount:literal]) => {
        #[repr(simd)]
        #[derive(Copy, Clone, Debug, PartialEq)]
        pub(crate) struct $id(pub [$ety; $ecount]);

        #[allow(clippy::use_self)]
        impl $id {
            #[inline(always)]
            pub(crate) const fn new(val: [$ety; $ecount]) -> Self {
                $id(val)
            }

            // FIXME: Workaround rust@60637
            #[inline(always)]
            pub(crate) const fn splat(value: $ety) -> Self {
                $id([value; $ecount])
            }

            // FIXME: Workaround rust@60637
            #[inline(always)]
            pub(crate) fn extract(self, index: usize) -> $ety {
                unsafe {
                    crate::core_arch::simd_llvm::simd_extract(self, index as u32)
                }
            }
        }
    }
}

macro_rules! simd_m_ty {
    ($id:ident: [$ety:ident; $ecount:literal]) => {
        #[repr(simd)]
        #[derive(Copy, Clone, Debug, PartialEq)]
        pub(crate) struct $id([$ety; $ecount]);

        #[allow(clippy::use_self)]
        impl $id {
            #[inline(always)]
            const fn bool_to_internal(x: bool) -> $ety {
                [0 as $ety, !(0 as $ety)][x as usize]
            }

            #[inline(always)]
            pub(crate) const fn new(val: [$ety; $ecount]) -> Self {
                $id(val)
            }

            // FIXME: Workaround rust@60637
            #[inline(always)]
            pub(crate) const fn splat(value: $ety) -> Self {
                $id([value; $ecount])
            }

            // FIXME: Workaround rust@60637
            #[inline(always)]
            pub(crate) fn extract(self, index: usize) -> $ety {
                unsafe {
                    crate::core_arch::simd_llvm::simd_extract(self, index as u32)
                }
            }
        }
    }
}

// 16-bit wide types:

simd_ty!(u8x2: [u8; 2]);

simd_ty!(i8x2: [i8; 2]);

// 32-bit wide types:

simd_ty!(u8x4: [u8; 4]);
simd_ty!(u16x2: [u16; 2]);

simd_ty!(i8x4: [i8; 4]);
simd_ty!(i16x2: [i16; 2]);


// 64-bit wide types:

simd_ty!(u8x8: [u8; 8]);
simd_ty!(u16x4: [u16; 4]);
simd_ty!(u32x2: [u32; 2]);
simd_ty!(u64x1: [u64; 1]);

simd_ty!(i8x8: [i8; 8]);
simd_ty!(i16x4: [i16; 4]);
simd_ty!(i32x2: [i32; 2]);
simd_ty!(i64x1: [i64; 1]);

simd_ty!(f32x2: [f32; 2]);
simd_ty!(f64x1: [f64; 1]);

// 128-bit wide types:

simd_ty!(u8x16: [u8; 16]);
simd_ty!(u16x8: [u16; 8]);
simd_ty!(u32x4: [u32; 4]);
simd_ty!(u64x2: [u64; 2]);

simd_ty!(i8x16: [i8; 16]);
simd_ty!(i16x8: [i16; 8]);
simd_ty!(i32x4: [i32; 4]);
simd_ty!(i64x2: [i64; 2]);

simd_ty!(f32x4: [f32; 4]);
simd_ty!(f64x2: [f64; 2]);
simd_ty!(f64x4: [f64; 4]);

simd_m_ty!(m8x16: [i8; 16]);
simd_m_ty!(m16x8: [i16; 8]);
simd_m_ty!(m32x4: [i32; 4]);
simd_m_ty!(m64x2: [i64; 2]);

// 256-bit wide types:

simd_ty!(u8x32: [u8; 32]);
simd_ty!(u16x16: [u16; 16]);
simd_ty!(u32x8: [u32; 8]);
simd_ty!(u64x4: [u64; 4]);

simd_ty!(i8x32: [i8; 32]);
simd_ty!(i16x16: [i16; 16]);
simd_ty!(i32x8: [i32; 8]);
simd_ty!(i64x4: [i64; 4]);

simd_ty!(f32x8: [f32; 8]);

// 512-bit wide types:

simd_ty!(u8x64: [u8; 64]);
simd_ty!(u16x32: [u16; 32]);
simd_ty!(u32x16: [u32; 16]);
simd_ty!(u64x8: [u64; 8]);

simd_ty!(i8x64: [i8; 64]);
simd_ty!(i16x32: [i16; 32]);
simd_ty!(i32x16: [i32; 16]);
simd_ty!(i64x8: [i64; 8]);

simd_ty!(f32x16: [f32; 16]);
simd_ty!(f64x8: [f64; 8]);
