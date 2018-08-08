//! This module implements the [WebAssembly `SIMD128` ISA].
//!
//! [WebAssembly `SIMD128` ISA]:
//! https://github.com/WebAssembly/simd/blob/master/proposals/simd/SIMD.md

#![allow(non_camel_case_types)]

/// A single unconstrained byte (0-255).
#[derive(Copy, Clone, Debug)]
pub struct ImmByte(u8);
impl ImmByte {
    /// Constructor
    #[inline]
    #[rustc_args_required_const(0)]
    pub const fn new(value: u8) -> Self {
        ImmByte(value)
    }
}

macro_rules! impl_laneidx {
    ($id:ident($ty:ty): [$_from:expr, $_to:expr] | $(#[$doc:meta])*) => {
        #[derive(Copy, Clone, Debug)]
        pub struct $id($ty);
        impl $id {
            #[inline]
            #[rustc_args_required_const(0)]
            pub const fn new(x: $ty) -> Self {
                // FIXME: not allowed in const fn:
                // * if statements
                // * unreachable_unchecked / panic / abort
                //
                // if x < $from || x > $to {
                //     unsafe { ::_core::hint::unreachable_unchecked() };
                //     debug_assert!(...)
                // }
                $id(x)
            }
        }
    };
}
impl_laneidx!(LaneIdx2(u8): [0, 1] | /// A byte with values in the range 0–1 identifying a lane.
);
impl_laneidx!(LaneIdx4(u8): [0, 3] | /// A byte with values in the range 0–3 identifying a lane.
);
impl_laneidx!(LaneIdx8(u8): [0, 7] | /// A byte with values in the range 0–7 identifying a lane.
);
impl_laneidx!(LaneIdx16(u8): [0, 15] | /// A byte with values in the range 0–15 identifying a lane.
);
impl_laneidx!(LaneIdx32(u8): [0, 31] | /// A byte with values in the range 0–31 identifying a lane.
);

types! {
    /// WASM-specific 128-bit wide SIMD vector type
    pub struct v128(i128);
}

mod sealed {
    types! {
        /// 128-bit wide SIMD vector type with 8 16-bit wide signed lanes
        pub struct v8x16(
            pub i8, pub i8, pub i8, pub i8, pub i8, pub i8, pub i8, pub i8,
            pub i8, pub i8, pub i8, pub i8, pub i8, pub i8, pub i8, pub i8,
        );
        /// 128-bit wide SIMD vector type with 8 16-bit wide signed lanes
        pub struct v16x8(
            pub i16, pub i16, pub i16, pub i16,
            pub i16, pub i16, pub i16, pub i16
        );
        /// 128-bit wide SIMD vector type with 4 32-bit wide signed lanes
        pub struct v32x4(pub i32, pub i32, pub i32, pub i32);
        /// 128-bit wide SIMD vector type with 2 64-bit wide signed lanes
        pub struct v64x2(pub i64, pub i64);

        /// 128-bit wide SIMD vector type with 8 16-bit wide unsigned lanes
        pub struct u8x16(
            pub u8, pub u8, pub u8, pub u8, pub u8, pub u8, pub u8, pub u8,
            pub u8, pub u8, pub u8, pub u8, pub u8, pub u8, pub u8, pub u8,
        );
        /// 128-bit wide SIMD vector type with 8 16-bit wide unsigned lanes
        pub struct u16x8(
            pub u16, pub u16, pub u16, pub u16,
            pub u16, pub u16, pub u16, pub u16
        );
        /// 128-bit wide SIMD vector type with 4 32-bit wide unsigned lanes
        pub struct u32x4(pub u32, pub u32, pub u32, pub u32);
        /// 128-bit wide SIMD vector type with 2 64-bit wide unsigned lanes
        pub struct u64x2(pub u64, pub u64);

        /// 128-bit wide SIMD vector type with 4 32-bit wide floating-point lanes
        pub struct f32x4(pub f32, pub f32, pub f32, pub f32);
        /// 128-bit wide SIMD vector type with 2 64-bit wide floating-point lanes
        pub struct f64x2(pub f64, pub f64);
    }

    #[allow(improper_ctypes)]
    extern "C" {
        #[link_name = "llvm.fabs.v4f32"]
        fn abs_v4f32(x: f32x4) -> f32x4;
        #[link_name = "llvm.fabs.v2f64"]
        fn abs_v2f64(x: f64x2) -> f64x2;
        #[link_name = "llvm.sqrt.v4f32"]
        fn sqrt_v4f32(x: f32x4) -> f32x4;
        #[link_name = "llvm.sqrt.v2f64"]
        fn sqrt_v2f64(x: f64x2) -> f64x2;
    }
    impl f32x4 {
        #[inline(always)]
        pub unsafe fn abs(self) -> Self {
            abs_v4f32(self)
        }
        #[inline(always)]
        pub unsafe fn sqrt(self) -> Self {
            sqrt_v4f32(self)
        }
    }
    impl f64x2 {
        #[inline(always)]
        pub unsafe fn abs(self) -> Self {
            abs_v2f64(self)
        }
        #[inline(always)]
        pub unsafe fn sqrt(self) -> Self {
            sqrt_v2f64(self)
        }
    }
}

/// WASM-specific v8x16 instructions
pub struct v8x16;
/// WASM-specific v16x8 instructions
pub struct v16x8;
/// WASM-specific v32x4 instructions
pub struct v32x4;
/// WASM-specific v64x2instructions
pub struct v64x2;

/// WASM-specific v8x16 instructions with modulo-arithmetic semantics
pub struct i8x16;
/// WASM-specific v16x8 instructions with modulo-arithmetic semantics
pub struct i16x8;
/// WASM-specific v32x4 instructions with modulo-arithmetic semantics
pub struct i32x4;
/// WASM-specific v64x2 instructions with modulo-arithmetic semantics
pub struct i64x2;

/// WASM-specific v32x4 floating-point instructions
pub struct f32x4;
/// WASM-specific v64x2 floating-point instructions
pub struct f64x2;

impl v128 {
    /// Materialize a constant SIMD value from the immediate operands.
    ///
    /// The `v128.const` instruction is encoded with 16 immediate bytes
    /// `imm` which provide the bits of the vector directly.
    #[inline]
    // #[target_feature(enable = "simd128")]
    // FIXME: #[cfg_attr(test, assert_instr(v128.const, imm = [ImmByte::new(42); 16]))]
    #[rustc_args_required_const(0)]
    pub const unsafe fn const_(imm: [ImmByte; 16]) -> v128 {
        union U {
            imm: [ImmByte; 16],
            vec: v128,
        }
        U { imm }.vec
    }
}

macro_rules! impl_splat {
    ($id:ident[$ivec_ty:ident : $elem_ty:ident] <= $x_ty:ident | $($lane_id:ident),*) => {
        impl $id {
            /// Create vector with identical lanes
            ///
            /// Construct a vector with `x` replicated to all lanes.
            #[inline]
            // #[target_feature(enable = "simd128")]
            // FIXME: #[cfg_attr(test, assert_instr($ident.splat))]
            pub const unsafe fn splat(x: $x_ty) -> v128 {
                union U {
                    vec: self::sealed::$ivec_ty,
                    res: v128
                }
                U { vec: self::sealed::$ivec_ty($({ struct $lane_id; x as $elem_ty}),*) }.res
            }
        }
    }
}
impl_splat!(i8x16[v8x16:i8] <= i32 |
            x0, x1, x2, x3, x4, x5, x6, x7,
            x8, x9, x10, x11, x12, x13, x14, x15
);
impl_splat!(i16x8[v16x8:i16] <= i32 | x0, x1, x2, x3, x4, x5, x6, x7);
impl_splat!(i32x4[v32x4:i32] <= i32 | x0, x1, x2, x3);
impl_splat!(i64x2[v64x2:i64] <= i64 | x0, x1);
impl_splat!(f32x4[f32x4:f32] <= f32 | x0, x1, x2, x3);
impl_splat!(f64x2[f64x2:f64] <= f64 | x0, x1);

macro_rules! impl_extract_lane {
    ($id:ident[$ivec_ty:ident : $selem_ty:ident|$uelem_ty:ident]($lane_idx:ty)
     => $x_ty:ident) => {
        impl $id {
            /// Extract lane as a scalar (sign-extend)
            ///
            /// Extract the scalar value of lane specified in the immediate mode
            /// operand `imm` from `a` by sign-extending it.
            #[inline]
            // #[target_feature(enable = "simd128")]
            // FIXME: #[cfg_attr(test, assert_instr($id.extract_lane_s, imm = 0))]
            #[rustc_args_required_const(1)]
            pub unsafe fn extract_lane_s(a: v128, imm: $lane_idx) -> $x_ty {
                use coresimd::simd_llvm::simd_extract;
                union U {
                    vec: self::sealed::$ivec_ty,
                    a: v128
                }
                // the vectors store a signed integer => extract into it
                let v: $selem_ty = simd_extract(U { a }.vec, imm.0 as u32 /* zero-extends index */);
                v as $x_ty
            }

            /// Extract lane as a scalar (zero-extend)
            ///
            /// Extract the scalar value of lane specified in the immediate mode
            /// operand `imm` from `a` by zero-extending it.
            #[inline]
            // #[target_feature(enable = "simd128")]
            // FIXME: #[cfg_attr(test, assert_instr($id.extract_lane_u, imm = 0))]
            #[rustc_args_required_const(1)]
            pub unsafe fn extract_lane_u(a: v128, imm: $lane_idx) -> $x_ty {
                use coresimd::simd_llvm::simd_extract;
                union U {
                    vec: self::sealed::$ivec_ty,
                    a: v128
                }
                // the vectors store a signed integer => extract into it
                let v: $selem_ty = simd_extract(U { a }.vec, imm.0 as u32  /* zero-extends index */);
                // re-interpret the signed integer as an unsigned one of the same size (no-op)
                let v: $uelem_ty= ::mem::transmute(v);
                // cast the internal unsigned integer to a larger signed integer (zero-extends)
                v as $x_ty
            }
        }
    };
    ($id:ident[$ivec_ty:ident]($lane_idx:ty) => $x_ty:ident) => {
        impl $id {
            /// Extract lane as a scalar
            ///
            /// Extract the scalar value of lane specified in the immediate mode
            /// operand `imm` from `a`.
            #[inline]
            // #[target_feature(enable = "simd128")]
            // FIXME: #[cfg_attr(test, assert_instr($id.extract_lane_u, imm = 0))]
            #[rustc_args_required_const(1)]
            pub unsafe fn extract_lane(a: v128, imm: $lane_idx) -> $x_ty {
                use coresimd::simd_llvm::simd_extract;
                union U {
                    vec: self::sealed::$ivec_ty,
                    a: v128
                }
                // the vectors store a signed integer => extract into it
                simd_extract(U { a }.vec, imm.0 as u32  /* zero-extends index */)
            }
        }
    };
}
impl_extract_lane!(i8x16[v8x16:i8|u8](LaneIdx16) => i32);
impl_extract_lane!(i16x8[v16x8:i16|u16](LaneIdx8) => i32);
impl_extract_lane!(i32x4[v32x4](LaneIdx4) => i32);
impl_extract_lane!(i64x2[v64x2](LaneIdx2) => i64);
impl_extract_lane!(f32x4[f32x4](LaneIdx4) => f32);
impl_extract_lane!(f64x2[f64x2](LaneIdx2) => f64);

macro_rules! impl_replace_lane {
    ($id:ident[$ivec_ty:ident:$ielem_ty:ident]($lane_idx:ty) <= $x_ty:ident) => {
        impl $id {
            /// Replace lane value
            ///
            /// Return a new vector with lanes identical to `a`, except for lane
            /// specified in the immediate mode argument `i` which has the value
            /// `x`.
            #[inline]
            // #[target_feature(enable = "simd128")]
            // FIXME: #[cfg_attr(test, assert_instr($id.extract_lane_u))]
            #[rustc_args_required_const(1)]
            pub unsafe fn replace_lane(a: v128, imm: $lane_idx, x: $x_ty) -> v128 {
                use coresimd::simd_llvm::simd_insert;
                union U {
                    vec: self::sealed::$ivec_ty,
                    a: v128
                }
                // the vectors store a signed integer => extract into it
                ::mem::transmute(
                    simd_insert(U { a }.vec,
                                imm.0 as u32  /* zero-extends index */,
                                x as $ielem_ty)
                )
            }
        }
    };
}

impl_replace_lane!(i8x16[v8x16:i8](LaneIdx16) <= i32);
impl_replace_lane!(i16x8[v16x8:i16](LaneIdx8) <= i32);
impl_replace_lane!(i32x4[v32x4:i32](LaneIdx4) <= i32);
impl_replace_lane!(i64x2[v64x2:i64](LaneIdx2) <= i64);
impl_replace_lane!(f32x4[f32x4:f32](LaneIdx4) <= f32);
impl_replace_lane!(f64x2[f64x2:f64](LaneIdx2) <= f64);

impl v8x16 {
    /// Shuffle lanes
    ///
    /// Create vector with lanes selected from the lanes of two input vectors
    /// `a` and `b` by the indices specified in the immediate mode operand
    /// `imm`. Each index selects an element of the result vector, where the
    /// indices `i` in range `[0, 15]` select the `i`-th elements of `a`, and
    /// the indices in range `[16, 31]` select the `i - 16`-th element of `b`.
    #[inline]
    // #[target_feature(enable = "simd128")]
    // FIXME: #[cfg_attr(test, assert_instr(v8x16.shuffle))]
    #[rustc_args_required_const(2)]
    pub unsafe fn shuffle(a: v128, b: v128, imm: [LaneIdx32; 16]) -> v128 {
        // FIXME: LLVM does not support v8x16.shuffle (use inline assembly?)
        let result: v128;
        asm!("v8x16.shuffle $0, $1, $2" : "=r"(result) : "r"(a), "r"(b), "r"(imm) : : );
        result
    }
}

macro_rules! impl_wrapping_add_sub_neg {
    ($id:ident[$ivec_ty:ident]) => {
        impl $id {
            /// Lane-wise wrapping integer addition
            #[inline]
            // #[target_feature(enable = "simd128")]
            // FIXME: #[cfg_attr(test, assert_instr($id.add))]
            pub unsafe fn add(a: v128, b: v128) -> v128 {
                use coresimd::simd_llvm::simd_add;
                let a: sealed::$ivec_ty = ::mem::transmute(a);
                let b: sealed::$ivec_ty = ::mem::transmute(b);
                ::mem::transmute(simd_add(a, b))
            }

            /// Lane-wise wrapping integer subtraction
            #[inline]
            // #[target_feature(enable = "simd128")]
            // FIXME: #[cfg_attr(test, assert_instr($id.sub))]
            pub unsafe fn sub(a: v128, b: v128) -> v128 {
                use coresimd::simd_llvm::simd_sub;
                let a: sealed::$ivec_ty = ::mem::transmute(a);
                let b: sealed::$ivec_ty = ::mem::transmute(b);
                ::mem::transmute(simd_sub(a, b))
            }

            /// Lane-wise wrapping integer negation
            #[inline]
            // #[target_feature(enable = "simd128")]
            // FIXME: #[cfg_attr(test, assert_instr($id.neg))]
            pub unsafe fn neg(a: v128) -> v128 {
                use coresimd::simd_llvm::simd_mul;
                let a: sealed::$ivec_ty = ::mem::transmute(a);
                let b: sealed::$ivec_ty = ::mem::transmute($id::splat(-1));
                ::mem::transmute(simd_mul(b, a))
            }

            // note: multiplication explicitly omitted (see below)
        }
    }
}

impl_wrapping_add_sub_neg!(i8x16[v8x16]);
impl_wrapping_add_sub_neg!(i16x8[v16x8]);
impl_wrapping_add_sub_neg!(i32x4[v32x4]);
impl_wrapping_add_sub_neg!(i64x2[v64x2]);

macro_rules! impl_wrapping_mul {
    ($id:ident[$ivec_ty:ident]) => {
        impl $id {
            /// Lane-wise wrapping integer multiplication
            #[inline]
            // #[target_feature(enable = "simd128")]
            // FIXME: #[cfg_attr(test, assert_instr($id.mul))]
            pub unsafe fn mul(a: v128, b: v128) -> v128 {
                use coresimd::simd_llvm::simd_mul;
                let a: sealed::$ivec_ty = ::mem::transmute(a);
                let b: sealed::$ivec_ty = ::mem::transmute(b);
                ::mem::transmute(simd_mul(a, b))
            }
        }
    };
}

impl_wrapping_mul!(i8x16[v8x16]);
impl_wrapping_mul!(i16x8[v16x8]);
impl_wrapping_mul!(i32x4[v32x4]);
// note: wrapping multiplication for i64x2 is not part of the spec

// TODO: Saturating integer arithmetic
// need to add intrinsics to rustc

macro_rules! impl_shl_scalar {
    ($id:ident[$ivec_ty:ident : $t:ty]) => {
        impl $id {
            /// Left shift by scalar.
            ///
            /// Shift the bits in each lane to the left by the same amount.
            /// Only the low bits of the shift amount are used.
            #[inline]
            // #[target_feature(enable = "simd128")]
            // FIXME: #[cfg_attr(test, assert_instr($id.shl))]
            pub unsafe fn shl(a: v128, y: i32) -> v128 {
                use coresimd::simd_llvm::simd_shl;
                let a: sealed::$ivec_ty = ::mem::transmute(a);
                let b: sealed::$ivec_ty = ::mem::transmute($id::splat(y as $t));
                ::mem::transmute(simd_shl(a, b))
            }
        }
    }
}

impl_shl_scalar!(i8x16[v8x16:i32]);
impl_shl_scalar!(i16x8[v16x8:i32]);
impl_shl_scalar!(i32x4[v32x4:i32]);
impl_shl_scalar!(i64x2[v64x2:i64]);

macro_rules! impl_shr_scalar {
    ($id:ident[$svec_ty:ident : $uvec_ty:ident : $t:ty]) => {
        impl $id {
            /// Arithmetic right shift by scalar.
            ///
            /// Shift the bits in each lane to the right by the same amount. 
            #[inline]
            // #[target_feature(enable = "simd128")]
            // FIXME: #[cfg_attr(test, assert_instr($id.shr))]
            pub unsafe fn shr_s(a: v128, y: i32) -> v128 {
                use coresimd::simd_llvm::simd_shr;
                let a: sealed::$svec_ty = ::mem::transmute(a);
                let b: sealed::$svec_ty = ::mem::transmute($id::splat(y as $t));
                ::mem::transmute(simd_shr(a, b))
            }

            /// Logical right shift by scalar.
            ///
            /// Shift the bits in each lane to the right by the same amount. 
            #[inline]
            // #[target_feature(enable = "simd128")]
            // FIXME: #[cfg_attr(test, assert_instr($id.shr))]
            pub unsafe fn shr_u(a: v128, y: i32) -> v128 {
                use coresimd::simd_llvm::simd_shr;
                let a: sealed::$uvec_ty = ::mem::transmute(a);
                let b: sealed::$uvec_ty = ::mem::transmute($id::splat(y as $t));
                ::mem::transmute(simd_shr(a, b))
            }

        }
    }
}

impl_shr_scalar!(i8x16[v8x16:u8x16:i32]);
impl_shr_scalar!(i16x8[v16x8:u16x8:i32]);
impl_shr_scalar!(i32x4[v32x4:u32x4:i32]);
impl_shr_scalar!(i64x2[v64x2:u64x2:i64]);


// Bitwise logical operations
impl v128 {
    /// Bitwise logical and
    #[inline]
    // #[target_feature(enable = "simd128")]
    // FIXME: #[cfg_attr(test, assert_instr($id.and))]
    pub unsafe fn and(a: v128, b: v128) -> v128 {
        use coresimd::simd_llvm::simd_and;
        simd_and(a, b)
    }

    /// Bitwise logical or
    #[inline]
    // #[target_feature(enable = "simd128")]
    // FIXME: #[cfg_attr(test, assert_instr($id.or))]
    pub unsafe fn or(a: v128, b: v128) -> v128 {
        use coresimd::simd_llvm::simd_or;
        simd_or(a, b)
    }

    /// Bitwise logical xor
    #[inline]
    // #[target_feature(enable = "simd128")]
    // FIXME: #[cfg_attr(test, assert_instr($id.xor))]
    pub unsafe fn xor(a: v128, b: v128) -> v128 {
        use coresimd::simd_llvm::simd_xor;
        simd_xor(a, b)
    }

    /// Bitwise logical not
    #[inline]
    // #[target_feature(enable = "simd128")]
    // FIXME: #[cfg_attr(test, assert_instr($id.not))]
    pub unsafe fn not(a: v128) -> v128 {
        union U {
            v: u128,
            c: [ImmByte; 16]
        }
        // FIXME: https://github.com/rust-lang/rust/issues/53193
        const C: [ImmByte; 16] = unsafe { U { v: ::_core::u128::MAX }.c };
        Self::xor(v128::const_(C), a)
    }

    /// Bitwise select
    ///
    /// Use the bits in the control mask `c` to select the corresponding bit
    /// from `v1` when `1` and `v2` when `0`.
    #[inline]
    // #[target_feature(enable = "simd128")]
    // FIXME: #[cfg_attr(test, assert_instr($id.bitselectnot))]
    pub unsafe fn bitselect(v1: v128, v2: v128, c: v128) -> v128 {
        // FIXME: use llvm.select instead - we need to add a `simd_bitselect`
        // intrinsic to rustc that converts a v128 vector into a i1x128. The
        // `simd_select` intrinsic converts e.g. a i8x16 into a i1x16 which is not
        // what we want here:
        Self::or(Self::and(v1, c), Self::and(v2, Self::not(c)))
    }
}

macro_rules! impl_boolean_reduction {
    ($id:ident[$ivec_ty:ident]) => {
        impl $id {
            /// Any lane true
            ///
            /// Returns `1` if any lane in `a` is non-zero, `0` otherwise.
            #[inline]
            // #[target_feature(enable = "simd128")]
            // FIXME: #[cfg_attr(test, assert_instr($id.any_true))]
            pub unsafe fn any_true(a: v128) -> i32 {
                use coresimd::simd_llvm::simd_reduce_any;
                let a: sealed::$ivec_ty = ::mem::transmute(a);
                if simd_reduce_any(a) { 1 } else { 0 }
            }

            /// All lanes true
            ///
            /// Returns `1` if all lanes in `a` are non-zero, `0` otherwise.
            #[inline]
            // #[target_feature(enable = "simd128")]
            // FIXME: #[cfg_attr(test, assert_instr($id.all_true))]
            pub unsafe fn all_true(a: v128) -> i32 {
                use coresimd::simd_llvm::simd_reduce_all;
                let a: sealed::$ivec_ty = ::mem::transmute(a);
                if simd_reduce_all(a) { 1 } else { 0 }
            }
        }
    }
}

impl_boolean_reduction!(i8x16[v8x16]);
impl_boolean_reduction!(i16x8[v16x8]);
impl_boolean_reduction!(i32x4[v32x4]);
impl_boolean_reduction!(i64x2[v64x2]);

macro_rules! impl_comparisons {
    ($id:ident[$ivec_ty:ident]) => {
        impl $id {
            /// Equality
            #[inline]
            // #[target_feature(enable = "simd128")]
            // FIXME: #[cfg_attr(test, assert_instr($id.eq))]
            pub unsafe fn eq(a: v128, b: v128) -> v128 {
                use coresimd::simd_llvm::simd_eq;
                let a: sealed::$ivec_ty = ::mem::transmute(a);
                let b: sealed::$ivec_ty = ::mem::transmute(b);
                let c: sealed::$ivec_ty = simd_eq(a, b);
                ::mem::transmute(c)
            }
            /// Non-Equality
            #[inline]
            // #[target_feature(enable = "simd128")]
            // FIXME: #[cfg_attr(test, assert_instr($id.ne))]
            pub unsafe fn ne(a: v128, b: v128) -> v128 {
                use coresimd::simd_llvm::simd_ne;
                let a: sealed::$ivec_ty = ::mem::transmute(a);
                let b: sealed::$ivec_ty = ::mem::transmute(b);
                let c: sealed::$ivec_ty = simd_ne(a, b);
                ::mem::transmute(c)
            }
            /// Less-than
            #[inline]
            // #[target_feature(enable = "simd128")]
            // FIXME: #[cfg_attr(test, assert_instr($id.lt))]
            pub unsafe fn lt(a: v128, b: v128) -> v128 {
                use coresimd::simd_llvm::simd_lt;
                let a: sealed::$ivec_ty = ::mem::transmute(a);
                let b: sealed::$ivec_ty = ::mem::transmute(b);
                let c: sealed::$ivec_ty = simd_lt(a, b);
                ::mem::transmute(c)
            }

            /// Less-than or equal
            #[inline]
            // #[target_feature(enable = "simd128")]
            // FIXME: #[cfg_attr(test, assert_instr($id.le))]
            pub unsafe fn le(a: v128, b: v128) -> v128 {
                use coresimd::simd_llvm::simd_le;
                let a: sealed::$ivec_ty = ::mem::transmute(a);
                let b: sealed::$ivec_ty = ::mem::transmute(b);
                let c: sealed::$ivec_ty = simd_le(a, b);
                ::mem::transmute(c)
            }

            /// Greater-than
            #[inline]
            // #[target_feature(enable = "simd128")]
            // FIXME: #[cfg_attr(test, assert_instr($id.gt))]
            pub unsafe fn gt(a: v128, b: v128) -> v128 {
                use coresimd::simd_llvm::simd_gt;
                let a: sealed::$ivec_ty = ::mem::transmute(a);
                let b: sealed::$ivec_ty = ::mem::transmute(b);
                let c: sealed::$ivec_ty = simd_gt(a, b);
                ::mem::transmute(c)
            }

            /// Greater-than or equal
            #[inline]
            // #[target_feature(enable = "simd128")]
            // FIXME: #[cfg_attr(test, assert_instr($id.ge))]
            pub unsafe fn ge(a: v128, b: v128) -> v128 {
                use coresimd::simd_llvm::simd_ge;
                let a: sealed::$ivec_ty = ::mem::transmute(a);
                let b: sealed::$ivec_ty = ::mem::transmute(b);
                let c: sealed::$ivec_ty = simd_ge(a, b);
                ::mem::transmute(c)
            }
        }
    }
}

impl_comparisons!(i8x16[v8x16]);
impl_comparisons!(i16x8[v16x8]);
impl_comparisons!(i32x4[v32x4]);
impl_comparisons!(i64x2[v64x2]);
impl_comparisons!(f32x4[f32x4]);
impl_comparisons!(f64x2[f64x2]);

// Load and store
impl v128 {
    /// Load a `v128` vector from the given heap address.
    #[inline]
    // #[target_feature(enable = "simd128")]
    // FIXME: #[cfg_attr(test, assert_instr($id.load))]
    pub unsafe fn load(m: *const v128) -> v128 {
        ::_core::ptr::read(m)

    }

    /// Store a `v128` vector to the given heap address.
    #[inline]
    // #[target_feature(enable = "simd128")]
    // FIXME: #[cfg_attr(test, assert_instr($id.store))]
    pub unsafe fn store(m: *mut v128, a: v128) {
        ::_core::ptr::write(m, a)
    }
}

// Floating-point operations
macro_rules! impl_floating_point_ops {
    ($id:ident) => {
        impl $id {
            /// Negation
            ///
            /// Apply the IEEE `negate(x)` function to each lane. This simply
            /// inverts the sign bit, preserving all other bits, even for `NaN`
            /// inputs.
            #[inline]
            // #[target_feature(enable = "simd128")]
            // FIXME: #[cfg_attr(test, assert_instr($id.neg))]
            pub unsafe fn neg(a: v128) -> v128 {
                use coresimd::simd_llvm::simd_mul;
                let a: sealed::$id = ::mem::transmute(a);
                let b: sealed::$id = ::mem::transmute($id::splat(-1.));
                ::mem::transmute(simd_mul(b, a))
            }
            /// Absolute value
            ///
            /// Apply the IEEE `abs(x)` function to each lane. This simply
            /// clears the sign bit, preserving all other bits, even for `NaN`
            /// inputs.
            #[inline]
            // #[target_feature(enable = "simd128")]
            // FIXME: #[cfg_attr(test, assert_instr($id.abs))]
            pub unsafe fn abs(a: v128) -> v128 {
                let a: sealed::$id = ::mem::transmute(a);
                ::mem::transmute(a.abs())
            }

            /// NaN-propagating minimum
            ///
            /// Lane-wise minimum value, propagating `NaN`s.
            #[inline]
            // #[target_feature(enable = "simd128")]
            // FIXME: #[cfg_attr(test, assert_instr($id.min))]
            pub unsafe fn min(a: v128, b: v128) -> v128 {
                use coresimd::simd_llvm::simd_fmin;
                let a: sealed::$id = ::mem::transmute(a);
                let b: sealed::$id = ::mem::transmute(b);
                ::mem::transmute(simd_fmin(a, b))
            }

            /// NaN-propagating maximum
            ///
            /// Lane-wise maximum value, propagating `NaN`s.
            #[inline]
            // #[target_feature(enable = "simd128")]
            // FIXME: #[cfg_attr(test, assert_instr($id.max))]
            pub unsafe fn max(a: v128, b: v128) -> v128 {
                use coresimd::simd_llvm::simd_fmax;
                let a: sealed::$id = ::mem::transmute(a);
                let b: sealed::$id = ::mem::transmute(b);
                ::mem::transmute(simd_fmax(a, b))
            }

            /// Square-root
            ///
            /// Lane-wise square-root.
            #[inline]
            // #[target_feature(enable = "simd128")]
            // FIXME: #[cfg_attr(test, assert_instr($id.sqrt))]
            pub unsafe fn sqrt(a: v128) -> v128 {
                let a: sealed::$id = ::mem::transmute(a);
                ::mem::transmute(a.sqrt())
            }

            /// Lane-wise addition
            #[inline]
            // #[target_feature(enable = "simd128")]
            // FIXME: #[cfg_attr(test, assert_instr($id.add))]
            pub unsafe fn add(a: v128, b: v128) -> v128 {
                use coresimd::simd_llvm::simd_add;
                let a: sealed::$id = ::mem::transmute(a);
                let b: sealed::$id = ::mem::transmute(b);
                ::mem::transmute(simd_add(a, b))
            }

            /// Lane-wise subtraction
            #[inline]
            // #[target_feature(enable = "simd128")]
            // FIXME: #[cfg_attr(test, assert_instr($id.sub))]
            pub unsafe fn sub(a: v128, b: v128) -> v128 {
                use coresimd::simd_llvm::simd_sub;
                let a: sealed::$id = ::mem::transmute(a);
                let b: sealed::$id = ::mem::transmute(b);
                ::mem::transmute(simd_sub(a, b))
            }

            /// Lane-wise multiplication
            #[inline]
            // #[target_feature(enable = "simd128")]
            // FIXME: #[cfg_attr(test, assert_instr($id.mul))]
            pub unsafe fn mul(a: v128, b: v128) -> v128 {
                use coresimd::simd_llvm::simd_mul;
                let a: sealed::$id = ::mem::transmute(a);
                let b: sealed::$id = ::mem::transmute(b);
                ::mem::transmute(simd_mul(a, b))
            }

            /// Lane-wise division
            #[inline]
            // #[target_feature(enable = "simd128")]
            // FIXME: #[cfg_attr(test, assert_instr($id.div))]
            pub unsafe fn div(a: v128, b: v128) -> v128 {
                use coresimd::simd_llvm::simd_div;
                let a: sealed::$id = ::mem::transmute(a);
                let b: sealed::$id = ::mem::transmute(b);
                ::mem::transmute(simd_div(a, b))
            }
        }
    };
}

impl_floating_point_ops!(f32x4);
impl_floating_point_ops!(f64x2);

macro_rules! impl_conversion {
    ($conversion:ident[$instr:expr]: $from_ty:ident => $to_ty:ident | $id:ident) => {
        impl $id {
            #[inline]
            // #[target_feature(enable = "simd128")]
            // FIXME: #[cfg_attr(test, assert_instr($instr))]
            pub unsafe fn $conversion(a: v128) -> v128 {
                use coresimd::simd_llvm::simd_cast;
                let a: sealed::$from_ty = ::mem::transmute(a);
                let b: sealed::$to_ty = simd_cast(a);
                ::mem::transmute(b)
            }
        }
    }
}

// Integer to floating point
impl_conversion!(convert_s_i32x4["f32x4.convert_s/i32x4"]: v32x4 => f32x4 | f32x4);
impl_conversion!(convert_u_i32x4["f32x4.convert_u/i32x4"]: v32x4 => f32x4 | f32x4);
impl_conversion!(convert_s_i64x2["f64x2.convert_s/i64x2"]: v64x2 => f64x2 | f64x2);
impl_conversion!(convert_u_i64x2["f64x2.convert_u/i64x2"]: v64x2 => f64x2 | f64x2);

// Floating point to integer with saturation
impl_conversion!(trunc_s_f32x4_sat["i32x4.trunc_s/f32x4:sat"]: f32x4 => v32x4 | i32x4);
impl_conversion!(trunc_u_f32x4_sat["i32x4.trunc_s/f32x4:sat"]: f32x4 => u32x4 | i32x4);
impl_conversion!(trunc_s_f64x2_sat["i64x2.trunc_s/f64x2:sat"]: f64x2 => v64x2 | i64x2);
impl_conversion!(trunc_u_f64x2_sat["i64x2.trunc_s/f64x2:sat"]: f64x2 => u64x2 | i64x2);
