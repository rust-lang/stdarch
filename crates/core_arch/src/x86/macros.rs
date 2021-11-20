//! Utility macros.
//!
// Helper struct used to trigger const eval errors when the const generic immediate value `imm` is
// not a round number.
pub(crate) struct ValidateConstRound<const IMM: i32>;
impl<const IMM: i32> ValidateConstRound<IMM> {
    pub(crate) const VALID: () = {
        assert!(
            IMM == 4 || IMM == 8 || IMM == 9 || IMM == 10 || IMM == 11,
            "Invalid IMM value"
        );
    };
}

#[allow(unused)]
macro_rules! static_assert_rounding {
    ($imm:ident) => {
        let _ = $crate::core_arch::x86::macros::ValidateConstRound::<$imm>::VALID;
    };
}

// Helper struct used to trigger const eval errors when the const generic immediate value `imm` is
// not a sae number.
pub(crate) struct ValidateConstSae<const IMM: i32>;
impl<const IMM: i32> ValidateConstSae<IMM> {
    pub(crate) const VALID: () = {
        assert!(IMM == 4 || IMM == 8, "Invalid IMM value");
    };
}

#[allow(unused)]
macro_rules! static_assert_sae {
    ($imm:ident) => {
        let _ = $crate::core_arch::x86::macros::ValidateConstSae::<$imm>::VALID;
    };
}

// Helper struct used to trigger const eval errors when the const generic immediate value `imm` is
// not a mantissas sae number.
pub(crate) struct ValidateConstMantissasSae<const IMM: i32>;
impl<const IMM: i32> ValidateConstMantissasSae<IMM> {
    pub(crate) const VALID: () = {
        assert!(IMM == 4 || IMM == 8 || IMM == 12, "Invalid IMM value");
    };
}

#[allow(unused)]
macro_rules! static_assert_mantissas_sae {
    ($imm:ident) => {
        let _ = $crate::core_arch::x86::macros::ValidateConstMantissasSae::<$imm>::VALID;
    };
}

// Helper struct used to trigger const eval errors when the unsigned const generic immediate value
// `IMM` is out of `[MIN-MAX]` range.
pub(crate) struct ValidateConstImmU32<const IMM: u32, const MIN: u32, const MAX: u32>;
impl<const IMM: u32, const MIN: u32, const MAX: u32> ValidateConstImmU32<IMM, MIN, MAX> {
    pub(crate) const VALID: () = {
        assert!(IMM >= MIN && IMM <= MAX, "IMM value not in expected range");
    };
}

#[allow(unused_macros)]
macro_rules! static_assert_imm_u8 {
    ($imm:ident) => {
        let _ =
            $crate::core_arch::x86::macros::ValidateConstImmU32::<$imm, 0, { (1 << 8) - 1 }>::VALID;
    };
}

// Helper struct used to trigger const eval errors when the const generic immediate value `SCALE` is
// not valid for gather instructions: the only valid scale values are 1, 2, 4 and 8.
pub(crate) struct ValidateConstGatherScale<const SCALE: i32>;
impl<const SCALE: i32> ValidateConstGatherScale<SCALE> {
    pub(crate) const VALID: () = {
        assert!(
            SCALE == 1 || SCALE == 2 || SCALE == 4 || SCALE == 8,
            "Invalid SCALE value"
        );
    };
}

#[allow(unused)]
macro_rules! static_assert_imm8_scale {
    ($imm:ident) => {
        let _ = $crate::core_arch::x86::macros::ValidateConstGatherScale::<$imm>::VALID;
    };
}


macro_rules! define_masked_load_aligned {
    ($feature:literal, $name:ident, $name_zero_masked:ident, $element_description:literal, $instruction:ident, $simd_type:path, $reg_type:ident, $mask_type:path, $lane_type:path, $alignment_description:literal) => {
        define_masked_load!($feature, $name, $name_zero_masked, $element_description, $instruction, $simd_type, $reg_type, $mask_type, $lane_type, "mem_addr must be aligned on a ", $alignment_description, " boundary or a general-protection exception may be generated.");
    }
}

macro_rules! define_masked_load_unaligned {
    ($feature:literal, $name:ident, $name_zero_masked:ident, $element_description:literal, $instruction:ident, $simd_type:path, $reg_type:ident, $mask_type:path, $lane_type:path) => {
        define_masked_load!($feature, $name, $name_zero_masked, $element_description, $instruction, $simd_type, $reg_type, $mask_type, $lane_type, "mem_addr does not need to be aligned on any particular boundary.");
    };
}

macro_rules! define_masked_load {
    ($feature:literal, $name:ident, $name_zero_masked:ident, $element_description:literal, $instruction:ident, $simd_type:path, $reg_type:ident, $mask_type:path, $lane_type:path, $($additional_doc:literal),+) => {
        #[inline]
        #[doc = "Load packed "]
        #[doc = $element_description]
        #[doc = " from memory into dst using writemask k (elements are copied from src when the corresponding mask bit is not set)."]
        $(#[doc = $additional_doc])+
        #[doc = ""]
        #[doc = concat!("[Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=", stringify!($name), ")")]
        #[target_feature(enable = $feature)]
        #[cfg_attr(test, assert_instr($instruction))]
        pub unsafe fn $name(src: $simd_type, k: $mask_type, mem_addr: *const $lane_type) -> $simd_type {
            let mut result: $simd_type = src;
            asm!(
                concat!(stringify!($instruction), " {r}{{{k}}}, [{p}]"),
                p = in(reg) mem_addr,
                k = in(kreg) k,
                r = inout($reg_type) result,
                options(nostack), options(pure), options(readonly)
            );
            result
        }

        #[inline]
        #[doc = "Load packed "]
        #[doc = $element_description]
        #[doc = " from memory into dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set)."]
        $(#[doc = $additional_doc])+
        #[doc = ""]
        #[doc = concat!("[Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=", stringify!($name_zero_masked), ")")]
        #[target_feature(enable = $feature)]
        #[cfg_attr(test, assert_instr($instruction))]
        pub unsafe fn $name_zero_masked(k: $mask_type, mem_addr: *const $lane_type) -> $simd_type {
            let mut result: $simd_type;
            asm!(
                concat!(stringify!($instruction), " {r}{{{k}}} {{z}}, [{p}]"),
                p = in(reg) mem_addr,
                k = in(kreg) k,
                r = out($reg_type) result,
                options(nostack), options(pure), options(readonly)
            );
            result
        }
    };
}

macro_rules! define_masked_store {
    ($feature:literal, $name:ident, $element_description:literal, $instruction:ident, $simd_type:path, $reg_type:ident, $mask_type:path, $lane_type:path, $($additional_doc:literal),+) => {
        #[inline]
        #[doc = "Store packed "]
        #[doc = $element_description]
        #[doc = " from from a into memory using writemask k."]
        $(#[doc = $additional_doc])+
        #[doc = ""]
        #[doc = concat!("[Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=", stringify!($name), ")")]
        #[target_feature(enable = $feature)]
        #[cfg_attr(test, assert_instr($instruction))]
        pub unsafe fn $name(mem_addr: *mut $lane_type, mask: $mask_type, a: $simd_type) {
            asm!(
                concat!(stringify!($instruction), " [{p}]{{{k}}}, {a}"),
                p = in(reg) mem_addr,
                k = in(kreg) mask,
                a = in($reg_type) a,
                options(nostack)
            );
        }
    }
}

macro_rules! define_masked_store_aligned {
    ($feature:literal, $name:ident, $element_description:literal, $instruction:ident, $simd_type:path, $reg_type:ident, $mask_type:path, $lane_type:path, $alignment_description:literal) => {
        define_masked_store!($feature, $name, $element_description, $instruction, $simd_type, $reg_type, $mask_type, $lane_type, "mem_addr must be aligned on a ", $alignment_description, " boundary or a general-protection exception may be generated.");
    }
}

macro_rules! define_masked_store_unaligned {
    ($feature:literal, $name:ident, $element_description:literal, $instruction:ident, $simd_type:path, $reg_type:ident, $mask_type:path, $lane_type:path) => {
        define_masked_store!($feature, $name, $element_description, $instruction, $simd_type, $reg_type, $mask_type, $lane_type, "mem_addr does not need to be aligned on any particular boundary.");
    };
}


#[cfg(test)]
macro_rules! assert_approx_eq {
    ($a:expr, $b:expr, $eps:expr) => {{
        let (a, b) = (&$a, &$b);
        assert!(
            (*a - *b).abs() < $eps,
            "assertion failed: `(left !== right)` \
             (left: `{:?}`, right: `{:?}`, expect diff: `{:?}`, real diff: `{:?}`)",
            *a,
            *b,
            $eps,
            (*a - *b).abs()
        );
    }};
}
