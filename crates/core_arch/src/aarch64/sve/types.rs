use super::*;
#[allow(improper_ctypes)]
use crate::marker::ConstParamTy;

pub(super) trait AsUnsigned {
    type Unsigned: ?Sized;
    unsafe fn as_unsigned(self) -> Self::Unsigned;
}

pub(super) trait AsSigned {
    type Signed: ?Sized;
    unsafe fn as_signed(self) -> Self::Signed;
}

pub(super) trait Into<T: ?Sized> {
    fn into(self) -> T;
}

macro_rules! impl_sve_type {
    ($(($v:vis, $elem_type:ty, $name:ident, $elt:literal))*) => ($(
        #[repr(simd, scalable($elt))]
        #[allow(non_camel_case_types)]
        $v struct $name {
            _ty: [$elem_type],
        }
    )*)
}

macro_rules! impl_sign_conversions_sv {
    ($(($signed:ty, $unsigned:ty))*) => ($(
        impl AsUnsigned for $signed {
            type Unsigned = $unsigned;

            #[inline]
            #[target_feature(enable = "sve")]
            unsafe fn as_unsigned(self) -> $unsigned {
                simd_reinterpret(self)
            }
        }

        impl AsSigned for $unsigned {
            type Signed = $signed;

            #[inline]
            #[target_feature(enable = "sve")]
            unsafe fn as_signed(self) -> $signed {
                simd_reinterpret(self)
            }
        }
    )*)
}

macro_rules! impl_sign_conversions {
    ($(($signed:ty, $unsigned:ty))*) => ($(
        impl AsUnsigned for $signed {
            type Unsigned = $unsigned;

            #[inline]
            #[target_feature(enable = "sve")]
            unsafe fn as_unsigned(self) -> $unsigned {
                crate::mem::transmute(self)
            }
        }

        impl AsSigned for $unsigned {
            type Signed = $signed;

            #[inline]
            #[target_feature(enable = "sve")]
            unsafe fn as_signed(self) -> $signed {
                crate::mem::transmute(self)
            }
        }
    )*)
}

/// LLVM requires the predicate lane count to be the same as the lane count
/// it's working with. However the ACLE only defines one bool type and the
/// instruction set doesn't have this distinction. As a result we have to
/// create these internal types so we can match the LLVM signature. Each of
/// these internal types can be converted to the public `svbool_t` type and
/// the `svbool_t` type can be converted into these.
macro_rules! impl_internal_sve_predicate {
    ($(($name:ident, $elt:literal))*) => ($(
        #[repr(simd, scalable($elt))]
        #[allow(non_camel_case_types)]
        pub(super) struct $name {
            _ty: [bool],
        }

        impl Into<svbool_t> for $name {
            #[inline(always)]
            fn into(self) -> svbool_t {
                #[allow(improper_ctypes)]
                extern "C" {
                    #[cfg_attr(
                        target_arch = "aarch64",
                        link_name = concat!("llvm.aarch64.sve.convert.to.svbool.nxv", $elt, "i1")
                    )]
                    fn convert_to_svbool(b: $name) -> svbool_t;
                }
                unsafe { convert_to_svbool(self) }
            }
        }

        impl Into<$name> for svbool_t {
            #[inline(always)]
            fn into(self) -> $name {
                #[allow(improper_ctypes)]
                extern "C" {
                    #[cfg_attr(
                        target_arch = "aarch64",
                        link_name = concat!("llvm.aarch64.sve.convert.from.svbool.nxv", $elt, "i1")
                    )]
                    fn convert_from_svbool(b: svbool_t) -> $name;
                }
                unsafe { convert_from_svbool(self) }
            }
        }
    )*)
}

impl_sve_type! {
    (pub, bool, svbool_t, 16)

    (pub, i8, svint8_t, 16)
    (pub, u8, svuint8_t, 16)

    (pub, i16, svint16_t, 8)
    (pub, u16, svuint16_t, 8)
    (pub, f32, svfloat32_t, 4)
    (pub, i32, svint32_t, 4)
    (pub, u32, svuint32_t, 4)
    (pub, f64, svfloat64_t, 2)
    (pub, i64, svint64_t, 2)
    (pub, u64, svuint64_t, 2)

    (pub, i8, svint8x2_t, 32)
    (pub, u8, svuint8x2_t, 32)
    (pub, i16, svint16x2_t, 16)
    (pub, u16, svuint16x2_t, 16)
    (pub, f32, svfloat32x2_t, 8)
    (pub, i32, svint32x2_t, 8)
    (pub, u32, svuint32x2_t, 8)
    (pub, f64, svfloat64x2_t, 4)
    (pub, i64, svint64x2_t, 4)
    (pub, u64, svuint64x2_t, 4)

    (pub, i8, svint8x3_t, 48)
    (pub, u8, svuint8x3_t, 48)
    (pub, i16, svint16x3_t, 24)
    (pub, u16, svuint16x3_t, 24)
    (pub, f32, svfloat32x3_t, 12)
    (pub, i32, svint32x3_t, 12)
    (pub, u32, svuint32x3_t, 12)
    (pub, f64, svfloat64x3_t, 6)
    (pub, i64, svint64x3_t, 6)
    (pub, u64, svuint64x3_t, 6)

    (pub, i8, svint8x4_t, 64)
    (pub, u8, svuint8x4_t, 64)
    (pub, i16, svint16x4_t, 32)
    (pub, u16, svuint16x4_t, 32)
    (pub, f32, svfloat32x4_t, 16)
    (pub, i32, svint32x4_t, 16)
    (pub, u32, svuint32x4_t, 16)
    (pub, f64, svfloat64x4_t, 8)
    (pub, i64, svint64x4_t, 8)
    (pub, u64, svuint64x4_t, 8)

// Internal types:
    (pub(super), i8, nxv2i8, 2)
    (pub(super), i8, nxv4i8, 4)
    (pub(super), i8, nxv8i8, 8)

    (pub(super), i16, nxv2i16, 2)
    (pub(super), i16, nxv4i16, 4)

    (pub(super), i32, nxv2i32, 2)

    (pub(super), u8, nxv2u8, 2)
    (pub(super), u8, nxv4u8, 4)
    (pub(super), u8, nxv8u8, 8)

    (pub(super), u16, nxv2u16, 2)
    (pub(super), u16, nxv4u16, 4)

    (pub(super), u32, nxv2u32, 2)
}

impl_sign_conversions! {
    (i8, u8)
    (i16, u16)
    (i32, u32)
    (i64, u64)
    (*const i8, *const u8)
    (*const i16, *const u16)
    (*const i32, *const u32)
    (*const i64, *const u64)
    (*mut i8, *mut u8)
    (*mut i16, *mut u16)
    (*mut i32, *mut u32)
    (*mut i64, *mut u64)
}

impl_sign_conversions_sv! {
    (svint8_t, svuint8_t)
    (svint16_t, svuint16_t)
    (svint32_t, svuint32_t)
    (svint64_t, svuint64_t)

    (svint8x2_t, svuint8x2_t)
    (svint16x2_t, svuint16x2_t)
    (svint32x2_t, svuint32x2_t)
    (svint64x2_t, svuint64x2_t)

    (svint8x3_t, svuint8x3_t)
    (svint16x3_t, svuint16x3_t)
    (svint32x3_t, svuint32x3_t)
    (svint64x3_t, svuint64x3_t)

    (svint8x4_t, svuint8x4_t)
    (svint16x4_t, svuint16x4_t)
    (svint32x4_t, svuint32x4_t)
    (svint64x4_t, svuint64x4_t)

    // Internal types:
    (nxv2i8, nxv2u8)
    (nxv4i8, nxv4u8)
    (nxv8i8, nxv8u8)

    (nxv2i16, nxv2u16)
    (nxv4i16, nxv4u16)

    (nxv2i32, nxv2u32)
}

impl_internal_sve_predicate! {
    (svbool2_t, 2)
    (svbool4_t, 4)
    (svbool8_t, 8)
}

#[repr(i32)]
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, PartialEq, Eq, ConstParamTy)]
#[non_exhaustive]
pub enum svpattern {
    SV_POW2 = 0,
    SV_VL1 = 1,
    SV_VL2 = 2,
    SV_VL3 = 3,
    SV_VL4 = 4,
    SV_VL5 = 5,
    SV_VL6 = 6,
    SV_VL7 = 7,
    SV_VL8 = 8,
    SV_VL16 = 9,
    SV_VL32 = 10,
    SV_VL64 = 11,
    SV_VL128 = 12,
    SV_VL256 = 13,
    SV_MUL4 = 29,
    SV_MUL3 = 30,
    SV_ALL = 31,
}

#[repr(i32)]
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, PartialEq, Eq, ConstParamTy)]
#[non_exhaustive]
pub enum svprfop {
    SV_PLDL1KEEP = 0,
    SV_PLDL1STRM = 1,
    SV_PLDL2KEEP = 2,
    SV_PLDL2STRM = 3,
    SV_PLDL3KEEP = 4,
    SV_PLDL3STRM = 5,
    SV_PSTL1KEEP = 8,
    SV_PSTL1STRM = 9,
    SV_PSTL2KEEP = 10,
    SV_PSTL2STRM = 11,
    SV_PSTL3KEEP = 12,
    SV_PSTL3STRM = 13,
}
