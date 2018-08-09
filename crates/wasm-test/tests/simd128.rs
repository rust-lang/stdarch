#![feature(use_extern_macros, stdsimd, asm, stmt_expr_attributes)]

#[macro_use]
extern crate coresimd;
extern crate wasm_bindgen_test;

use coresimd::arch::wasm32::*;
use std::mem;
use wasm_bindgen_test::*;

fn compare_bytes(a: v128, b: v128) {
    let a: [u8; 16] = unsafe { mem::transmute(a) };
    let b: [u8; 16] = unsafe { mem::transmute(b) };
    assert_eq!(a, b);
}

#[wasm_bindgen_test]
fn v128_const() {
    const A: v128 = unsafe {
        v128::const_([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    };
    compare_bytes(A, A);
}

macro_rules! test_splat {
    ($test_id:ident: $id:ident($val:expr) => $($vals:expr),*) => {
        #[wasm_bindgen_test]
        fn $test_id() {
            const A: v128 = unsafe {
                $id::splat($val)
            };
            const B: v128 = unsafe {
                v128::const_([$($vals),*])
            };
            compare_bytes(A, B);
        }
    }
}

test_splat!(i8x16_splat: i8x16(42) => 42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42);
test_splat!(i16x8_splat: i16x8(42) => 42, 0, 42, 0, 42, 0, 42, 0, 42, 0, 42, 0, 42, 0, 42, 0);
test_splat!(i32x4_splat: i32x4(42) => 42, 0, 0, 0, 42, 0, 0, 0, 42, 0, 0, 0, 42, 0, 0, 0);
test_splat!(i64x2_splat: i64x2(42) => 42, 0, 0, 0, 0, 0, 0, 0, 42, 0, 0, 0, 0, 0, 0, 0);
test_splat!(f32x4_splat: f32x4(42.) => 0, 0, 40, 66, 0, 0, 40, 66, 0, 0, 40, 66, 0, 0, 40, 66);
test_splat!(f64x2_splat: f64x2(42.) => 0, 0, 0, 0, 0, 0, 69, 64, 0, 0, 0, 0, 0, 0, 69, 64);

// tests extract and replace lanes
macro_rules! test_extract {
    ($test_id:ident: $id:ident[$ety:ident] => $extract_fn:ident | [$val:expr; $count:expr]
     | [$($vals:expr),*] => ($other:expr)
     | $($ids:expr),*) => {
        #[wasm_bindgen_test]
        fn $test_id() {
            unsafe {
                // splat vector and check that all indices contain the same value
                // splatted:
                const A: v128 = unsafe {
                    $id::splat($val)
                };
                $(
                    assert_eq!($id::$extract_fn(A, $ids) as $ety, $val);
                )*;

                // create a vector from array and check that the indices contain
                // the same values as in the array:
                let arr: [$ety; $count] = [$($vals),*];
                let mut vec: v128 = mem::transmute(arr);
                $(
                    assert_eq!($id::$extract_fn(vec, $ids) as $ety, arr[$ids]);
                )*;

                // replace lane 0 with another value
                vec = $id::replace_lane(vec, 0, $other);
                assert_ne!($id::$extract_fn(vec, 0) as $ety, arr[0]);
                assert_eq!($id::$extract_fn(vec, 0) as $ety, $other);
            }
        }
    }
}

test_extract!(i8x16_extract_u: i8x16[u8] => extract_lane_u | [255; 16]
              | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] => (42)
              | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
);
test_extract!(i8x16_extract_s: i8x16[i8] => extract_lane_s | [-122; 16]
              | [0, -1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11, 12, -13, 14, -15] => (-42)
              | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
);

test_extract!(i16x8_extract_u: i16x8[u16] => extract_lane_u | [255; 8]
              | [0, 1, 2, 3, 4, 5, 6, 7]  => (42) | 0, 1, 2, 3, 4, 5, 6, 7
);
test_extract!(i16x8_extract_s: i16x8[i16] => extract_lane_s | [-122; 8]
              | [0, -1, 2, -3, 4, -5, 6, -7]  => (-42) | 0, 1, 2, 3, 4, 5, 6, 7
);
test_extract!(i32x4_extract: i32x4[i32] => extract_lane | [-122; 4]
              | [0, -1, 2, -3]  => (42) | 0, 1, 2, 3
);
test_extract!(i64x2_extract: i64x2[i64] => extract_lane | [-122; 2]
              | [0, -1]  => (42) | 0, 1
);
test_extract!(f32x4_extract: f32x4[f32] => extract_lane | [-122.; 4]
              | [0., -1., 2., -3.]  => (42.) | 0, 1, 2, 3
);
test_extract!(f64x2_extract: f64x2[f64] => extract_lane | [-122.; 2]
              | [0., -1.]  => (42.) | 0, 1
);

#[wasm_bindgen_test]
fn v8x16_shuffle() {
    unsafe {
        let a = [0_u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let b = [
            16_u8, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        ];

        let vec_a: v128 = mem::transmute(a);
        let vec_b: v128 = mem::transmute(b);

        let vec_r = v8x16_shuffle!(
            vec_a,
            vec_b,
            [0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30]
        );

        let e = [0_u8, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30];
        let vec_e: v128 = mem::transmute(e);
        compare_bytes(vec_r, vec_e);
    }
}

macro_rules! floating_point {
    (f32) => {
        true
    };
    (f64) => {
        true
    };
    ($id:ident) => {
        false
    };
}

trait IsNan: Sized {
    fn is_nan(self) -> bool {
        false
    }
}
impl IsNan for i8 {}
impl IsNan for i16 {}
impl IsNan for i32 {}
impl IsNan for i64 {}

macro_rules! test_bop {
    ($id:ident[$ety:ident; $ecount:expr] |
     $binary_op:ident [$op_test_id:ident] :
     ([$($in_a:expr),*], [$($in_b:expr),*]) => [$($out:expr),*]) => {
        test_bop!(
            $id[$ety; $ecount] => $ety | $binary_op [ $op_test_id ]:
            ([$($in_a),*], [$($in_b),*]) => [$($out),*]
        );

    };
    ($id:ident[$ety:ident; $ecount:expr] => $oty:ident |
     $binary_op:ident [$op_test_id:ident] :
     ([$($in_a:expr),*], [$($in_b:expr),*]) => [$($out:expr),*]) => {
        #[wasm_bindgen_test]
        fn $op_test_id() {
            unsafe {
                let a_input: [$ety; $ecount] = [$($in_a),*];
                let b_input: [$ety; $ecount] = [$($in_b),*];
                let output: [$oty; $ecount] = [$($out),*];

                let a_vec_in: v128 = mem::transmute(a_input);
                let b_vec_in: v128 = mem::transmute(b_input);
                let vec_res: v128 = $id::$binary_op(a_vec_in, b_vec_in);

                let res: [$oty; $ecount] = mem::transmute(vec_res);

                if !floating_point!($ety) {
                    assert_eq!(res, output);
                } else {
                    for i in 0..$ecount {
                        let r = res[i];
                        let o = output[i];
                        assert_eq!(r.is_nan(), o.is_nan());
                        if !r.is_nan() {
                            assert_eq!(r, o);
                        }
                    }
                }
            }
        }
    }
}

macro_rules! test_bops {
    ($id:ident[$ety:ident; $ecount:expr] |
     $binary_op:ident [$op_test_id:ident]:
     ([$($in_a:expr),*], $in_b:expr) => [$($out:expr),*]) => {
        #[wasm_bindgen_test]
        fn $op_test_id() {
            unsafe {
                let a_input: [$ety; $ecount] = [$($in_a),*];
                let output: [$ety; $ecount] = [$($out),*];

                let a_vec_in: v128 = mem::transmute(a_input);
                let vec_res: v128 = $id::$binary_op(a_vec_in, $in_b);

                let res: [$ety; $ecount] = mem::transmute(vec_res);
                assert_eq!(res, output);
            }
        }
    }
}

macro_rules! test_uop {
    ($id:ident[$ety:ident; $ecount:expr] |
     $unary_op:ident [$op_test_id:ident]: [$($in_a:expr),*] => [$($out:expr),*]) => {
        #[wasm_bindgen_test]
        fn $op_test_id() {
            unsafe {
                let a_input: [$ety; $ecount] = [$($in_a),*];
                let output: [$ety; $ecount] = [$($out),*];

                let a_vec_in: v128 = mem::transmute(a_input);
                let vec_res: v128 = $id::$unary_op(a_vec_in);

                let res: [$ety; $ecount] = mem::transmute(vec_res);
                assert_eq!(res, output);
            }
        }
    }
}

test_bop!(i8x16[i8; 16] | add[i8x16_add_test]:
          ([0, -1, 2, 3, 4, 5, 6, i8::max_value(), 1, 1, 1, 1, 1, 1, 1, 1],
           [8, i8::min_value(), 10, 11, 12, 13, 14, 1, 1, 1, 1, 1, 1, 1, 1, 1]) =>
          [8, i8::max_value(), 12, 14, 16, 18, 20, i8::min_value(), 2, 2, 2, 2, 2, 2, 2, 2]);
test_bop!(i8x16[i8; 16] | sub[i8x16_sub_test]:
          ([0, -1, 2, 3, 4, 5, 6, -1, 1, 1, 1, 1, 1, 1, 1, 1],
           [8, i8::min_value(), 10, 11, 12, 13, 14, i8::max_value(), 1, 1, 1, 1, 1, 1, 1, 1]) =>
          [-8, i8::max_value(), -8, -8, -8, -8, -8, i8::min_value(), 0, 0, 0, 0, 0, 0, 0, 0]);
test_bop!(i8x16[i8; 16] | mul[i8x16_mul_test]:
          ([0, -2, 2, 3, 4, 5, 6, 2, 1, 1, 1, 1, 1, 1, 1, 1],
           [8, i8::min_value(), 10, 11, 12, 13, 14, i8::max_value(), 1, 1, 1, 1, 1, 1, 1, 1]) =>
          [0, 0, 20, 33, 48, 65, 84, -2, 1, 1, 1, 1, 1, 1, 1, 1]);
test_uop!(i8x16[i8; 16] | neg[i8x16_neg_test]:
          [8, i8::min_value(), 10, 11, 12, 13, 14, i8::max_value(), 1, 1, 1, 1, 1, 1, 1, 1] =>
          [-8, i8::min_value(), -10, -11, -12, -13, -14, i8::min_value() + 1, -1, -1, -1, -1, -1, -1, -1, -1]);

test_bop!(i16x8[i16; 8] | add[i16x8_add_test]:
          ([0, -1, 2, 3, 4, 5, 6, i16::max_value()],
           [8, i16::min_value(), 10, 11, 12, 13, 14, 1]) =>
          [8, i16::max_value(), 12, 14, 16, 18, 20, i16::min_value()]);
test_bop!(i16x8[i16; 8] | sub[i16x8_sub_test]:
          ([0, -1, 2, 3, 4, 5, 6, -1],
           [8, i16::min_value(), 10, 11, 12, 13, 14, i16::max_value()]) =>
          [-8, i16::max_value(), -8, -8, -8, -8, -8, i16::min_value()]);
test_bop!(i16x8[i16; 8] | mul[i16x8_mul_test]:
          ([0, -2, 2, 3, 4, 5, 6, 2],
           [8, i16::min_value(), 10, 11, 12, 13, 14, i16::max_value()]) =>
          [0, 0, 20, 33, 48, 65, 84, -2]);
test_uop!(i16x8[i16; 8] | neg[i16x8_neg_test]:
          [8, i16::min_value(), 10, 11, 12, 13, 14, i16::max_value()] =>
          [-8, i16::min_value(), -10, -11, -12, -13, -14, i16::min_value() + 1]);

test_bop!(i32x4[i32; 4] | add[i32x4_add_test]:
          ([0, -1, 2, i32::max_value()],
           [8, i32::min_value(), 10, 1]) =>
          [8, i32::max_value(), 12, i32::min_value()]);
test_bop!(i32x4[i32; 4] | sub[i32x4_sub_test]:
          ([0, -1, 2, -1],
           [8, i32::min_value(), 10, i32::max_value()]) =>
          [-8, i32::max_value(), -8, i32::min_value()]);
test_bop!(i32x4[i32; 4] | mul[i32x4_mul_test]:
          ([0, -2, 2, 2],
           [8, i32::min_value(), 10, i32::max_value()]) =>
          [0, 0, 20, -2]);
test_uop!(i32x4[i32; 4] | neg[i32x4_neg_test]:
          [8, i32::min_value(), 10, i32::max_value()] =>
          [-8, i32::min_value(), -10, i32::min_value() + 1]);

test_bop!(i64x2[i64; 2] | add[i64x2_add_test]:
          ([-1, i64::max_value()],
           [i64::min_value(), 1]) =>
          [i64::max_value(), i64::min_value()]);
test_bop!(i64x2[i64; 2] | sub[i64x2_sub_test]:
          ([-1, -1],
           [i64::min_value(), i64::max_value()]) =>
          [ i64::max_value(), i64::min_value()]);
// note: mul for i64x2 is not part of the spec
test_uop!(i64x2[i64; 2] | neg[i64x2_neg_test]:
          [i64::min_value(), i64::max_value()] =>
          [i64::min_value(), i64::min_value() + 1]);

test_bops!(i8x16[i8; 16] | shl[i8x16_shl_test]:
          ([0, -1, 2, 3, 4, 5, 6, i8::max_value(), 1, 1, 1, 1, 1, 1, 1, 1], 1) =>
           [0, -2, 4, 6, 8, 10, 12, -2, 2, 2, 2, 2, 2, 2, 2, 2]);
test_bops!(i16x8[i16; 8] | shl[i16x8_shl_test]:
          ([0, -1, 2, 3, 4, 5, 6, i16::max_value()], 1) =>
           [0, -2, 4, 6, 8, 10, 12, -2]);
test_bops!(i32x4[i32; 4] | shl[i32x4_shl_test]:
           ([0, -1, 2, 3], 1) => [0, -2, 4, 6]);
test_bops!(i64x2[i64; 2] | shl[i64x2_shl_test]:
           ([0, -1], 1) => [0, -2]);

test_bops!(i8x16[i8; 16] | shr_s[i8x16_shr_s_test]:
           ([0, -1, 2, 3, 4, 5, 6, i8::max_value(), 1, 1, 1, 1, 1, 1, 1, 1], 1) =>
           [0, -1, 1, 1, 2, 2, 3, 63, 0, 0, 0, 0, 0, 0, 0, 0]);
test_bops!(i16x8[i16; 8] | shr_s[i16x8_shr_s_test]:
           ([0, -1, 2, 3, 4, 5, 6, i16::max_value()], 1) =>
           [0, -1, 1, 1, 2, 2, 3, i16::max_value() / 2]);
test_bops!(i32x4[i32; 4] | shr_s[i32x4_shr_s_test]:
           ([0, -1, 2, 3], 1) => [0, -1, 1, 1]);
test_bops!(i64x2[i64; 2] | shr_s[i64x2_shr_s_test]:
           ([0, -1], 1) => [0, -1]);

test_bops!(i8x16[i8; 16] | shr_u[i8x16_uhr_u_test]:
           ([0, -1, 2, 3, 4, 5, 6, i8::max_value(), 1, 1, 1, 1, 1, 1, 1, 1], 1) =>
           [0, i8::max_value(), 1, 1, 2, 2, 3, 63, 0, 0, 0, 0, 0, 0, 0, 0]);
test_bops!(i16x8[i16; 8] | shr_u[i16x8_uhr_u_test]:
           ([0, -1, 2, 3, 4, 5, 6, i16::max_value()], 1) =>
           [0, i16::max_value(), 1, 1, 2, 2, 3, i16::max_value() / 2]);
test_bops!(i32x4[i32; 4] | shr_u[i32x4_uhr_u_test]:
           ([0, -1, 2, 3], 1) => [0, i32::max_value(), 1, 1]);
test_bops!(i64x2[i64; 2] | shr_u[i64x2_uhr_u_test]:
           ([0, -1], 1) => [0, i64::max_value()]);

#[wasm_bindgen_test]
fn v128_bitwise_logical_ops() {
    unsafe {
        let a: [u32; 4] = [u32::max_value(), 0, u32::max_value(), 0];
        let b: [u32; 4] = [u32::max_value(); 4];
        let c: [u32; 4] = [0; 4];

        let vec_a: v128 = mem::transmute(a);
        let vec_b: v128 = mem::transmute(b);
        let vec_c: v128 = mem::transmute(c);

        let r: v128 = v128::and(vec_a, vec_a);
        compare_bytes(r, vec_a);
        let r: v128 = v128::and(vec_a, vec_b);
        compare_bytes(r, vec_a);
        let r: v128 = v128::or(vec_a, vec_b);
        compare_bytes(r, vec_b);
        let r: v128 = v128::not(vec_b);
        compare_bytes(r, vec_c);
        let r: v128 = v128::xor(vec_a, vec_c);
        compare_bytes(r, vec_a);

        let r: v128 = v128::bitselect(vec_b, vec_c, vec_b);
        compare_bytes(r, vec_b);
        let r: v128 = v128::bitselect(vec_b, vec_c, vec_c);
        compare_bytes(r, vec_c);
        let r: v128 = v128::bitselect(vec_b, vec_c, vec_a);
        compare_bytes(r, vec_a);
    }
}

macro_rules! test_bool_red {
    ($id:ident[$test_id:ident] | [$($true:expr),*] | [$($false:expr),*] | [$($alt:expr),*]) => {
        #[wasm_bindgen_test]
        fn $test_id() {
            unsafe {
                let vec_a: v128 = mem::transmute([$($true),*]); // true
                let vec_b: v128 = mem::transmute([$($false),*]); // false
                let vec_c: v128 = mem::transmute([$($alt),*]); // alternating

                assert_eq!($id::any_true(vec_a), 1);
                assert_eq!($id::any_true(vec_b), 0);
                assert_eq!($id::any_true(vec_c), 1);

                assert_eq!($id::all_true(vec_a), 1);
                assert_eq!($id::all_true(vec_b), 0);
                assert_eq!($id::all_true(vec_c), 0);
            }
        }
    }
}

test_bool_red!(
    i8x16[i8x16_boolean_reductions]
        | [1_i8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        | [0_i8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        | [1_i8, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
);
test_bool_red!(
    i16x8[i16x8_boolean_reductions]
        | [1_i16, 1, 1, 1, 1, 1, 1, 1]
        | [0_i16, 0, 0, 0, 0, 0, 0, 0]
        | [1_i16, 0, 1, 0, 1, 0, 1, 0]
);
test_bool_red!(
    i32x4[i32x4_boolean_reductions]
        | [1_i32, 1, 1, 1]
        | [0_i32, 0, 0, 0]
        | [1_i32, 0, 1, 0]
);
test_bool_red!(
    i64x2[i64x2_boolean_reductions] | [1_i64, 1] | [0_i64, 0] | [1_i64, 0]
);

test_bop!(i8x16[i8; 16] | eq[i8x16_eq_test]:
          ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
           [0, 2, 2, 4, 4, 6, 6, 7, 8, 10, 10, 12, 12, 14, 14, 15]) =>
          [-1, 0, -1, 0 ,-1, 0, -1, -1, -1, 0, -1, 0 ,-1, 0, -1, -1]);
test_bop!(i16x8[i16; 8] | eq[i16x8_eq_test]:
          ([0, 1, 2, 3, 4, 5, 6, 7], [0, 2, 2, 4, 4, 6, 6, 7]) =>
          [-1, 0, -1, 0 ,-1, 0, -1, -1]);
test_bop!(i32x4[i32; 4] | eq[i32x4_eq_test]:
          ([0, 1, 2, 3], [0, 2, 2, 4]) => [-1, 0, -1, 0]);
test_bop!(i64x2[i64; 2] | eq[i64x2_eq_test]: ([0, 1], [0, 2]) => [-1, 0]);
test_bop!(f32x4[f32; 4] => i32 | eq[f32x4_eq_test]:
          ([0., 1., 2., 3.], [0., 2., 2., 4.]) => [-1, 0, -1, 0]);
test_bop!(f64x2[f64; 2] => i64 | eq[f64x2_eq_test]: ([0., 1.], [0., 2.]) => [-1, 0]);

test_bop!(i8x16[i8; 16] | ne[i8x16_ne_test]:
          ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
           [0, 2, 2, 4, 4, 6, 6, 7, 8, 10, 10, 12, 12, 14, 14, 15]) =>
          [0, -1, 0, -1 ,0, -1, 0, 0, 0, -1, 0, -1 ,0, -1, 0, 0]);
test_bop!(i16x8[i16; 8] | ne[i16x8_ne_test]:
          ([0, 1, 2, 3, 4, 5, 6, 7], [0, 2, 2, 4, 4, 6, 6, 7]) =>
          [0, -1, 0, -1 ,0, -1, 0, 0]);
test_bop!(i32x4[i32; 4] | ne[i32x4_ne_test]:
          ([0, 1, 2, 3], [0, 2, 2, 4]) => [0, -1, 0, -1]);
test_bop!(i64x2[i64; 2] | ne[i64x2_ne_test]: ([0, 1], [0, 2]) => [0, -1]);
test_bop!(f32x4[f32; 4] => i32 | ne[f32x4_ne_test]:
          ([0., 1., 2., 3.], [0., 2., 2., 4.]) => [0, -1, 0, -1]);
test_bop!(f64x2[f64; 2] => i64 | ne[f64x2_ne_test]: ([0., 1.], [0., 2.]) => [0, -1]);

test_bop!(i8x16[i8; 16] | lt[i8x16_lt_test]:
          ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
           [0, 2, 2, 4, 4, 6, 6, 7, 8, 10, 10, 12, 12, 14, 14, 15]) =>
          [0, -1, 0, -1 ,0, -1, 0, 0, 0, -1, 0, -1 ,0, -1, 0, 0]);
test_bop!(i16x8[i16; 8] | lt[i16x8_lt_test]:
          ([0, 1, 2, 3, 4, 5, 6, 7], [0, 2, 2, 4, 4, 6, 6, 7]) =>
          [0, -1, 0, -1 ,0, -1, 0, 0]);
test_bop!(i32x4[i32; 4] | lt[i32x4_lt_test]:
          ([0, 1, 2, 3], [0, 2, 2, 4]) => [0, -1, 0, -1]);
test_bop!(i64x2[i64; 2] | lt[i64x2_lt_test]: ([0, 1], [0, 2]) => [0, -1]);
test_bop!(f32x4[f32; 4] => i32 | lt[f32x4_lt_test]:
          ([0., 1., 2., 3.], [0., 2., 2., 4.]) => [0, -1, 0, -1]);
test_bop!(f64x2[f64; 2] => i64 | lt[f64x2_lt_test]: ([0., 1.], [0., 2.]) => [0, -1]);

test_bop!(i8x16[i8; 16] | gt[i8x16_gt_test]:
          ([0, 2, 2, 4, 4, 6, 6, 7, 8, 10, 10, 12, 12, 14, 14, 15],
           [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]) =>
          [0, -1, 0, -1 ,0, -1, 0, 0, 0, -1, 0, -1 ,0, -1, 0, 0]);
test_bop!(i16x8[i16; 8] | gt[i16x8_gt_test]:
          ([0, 2, 2, 4, 4, 6, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7]) =>
          [0, -1, 0, -1 ,0, -1, 0, 0]);
test_bop!(i32x4[i32; 4] | gt[i32x4_gt_test]:
          ([0, 2, 2, 4], [0, 1, 2, 3]) => [0, -1, 0, -1]);
test_bop!(i64x2[i64; 2] | gt[i64x2_gt_test]: ([0, 2], [0, 1]) => [0, -1]);
test_bop!(f32x4[f32; 4] => i32 | gt[f32x4_gt_test]:
          ([0., 2., 2., 4.], [0., 1., 2., 3.]) => [0, -1, 0, -1]);
test_bop!(f64x2[f64; 2] => i64 | gt[f64x2_gt_test]: ([0., 2.], [0., 1.]) => [0, -1]);

test_bop!(i8x16[i8; 16] | ge[i8x16_ge_test]:
          ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
           [0, 2, 2, 4, 4, 6, 6, 7, 8, 10, 10, 12, 12, 14, 14, 15]) =>
          [-1, 0, -1, 0 ,-1, 0, -1, -1, -1, 0, -1, 0 ,-1, 0, -1, -1]);
test_bop!(i16x8[i16; 8] | ge[i16x8_ge_test]:
          ([0, 1, 2, 3, 4, 5, 6, 7], [0, 2, 2, 4, 4, 6, 6, 7]) =>
          [-1, 0, -1, 0 ,-1, 0, -1, -1]);
test_bop!(i32x4[i32; 4] | ge[i32x4_ge_test]:
          ([0, 1, 2, 3], [0, 2, 2, 4]) => [-1, 0, -1, 0]);
test_bop!(i64x2[i64; 2] | ge[i64x2_ge_test]: ([0, 1], [0, 2]) => [-1, 0]);
test_bop!(f32x4[f32; 4] => i32 | ge[f32x4_ge_test]:
          ([0., 1., 2., 3.], [0., 2., 2., 4.]) => [-1, 0, -1, 0]);
test_bop!(f64x2[f64; 2] => i64 | ge[f64x2_ge_test]: ([0., 1.], [0., 2.]) => [-1, 0]);

test_bop!(i8x16[i8; 16] | le[i8x16_le_test]:
          ([0, 2, 2, 4, 4, 6, 6, 7, 8, 10, 10, 12, 12, 14, 14, 15],
           [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
           ) =>
          [-1, 0, -1, 0 ,-1, 0, -1, -1, -1, 0, -1, 0 ,-1, 0, -1, -1]);
test_bop!(i16x8[i16; 8] | le[i16x8_le_test]:
          ([0, 2, 2, 4, 4, 6, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7]) =>
          [-1, 0, -1, 0 ,-1, 0, -1, -1]);
test_bop!(i32x4[i32; 4] | le[i32x4_le_test]:
          ([0, 2, 2, 4], [0, 1, 2, 3]) => [-1, 0, -1, 0]);
test_bop!(i64x2[i64; 2] | le[i64x2_le_test]: ([0, 2], [0, 1]) => [-1, 0]);
test_bop!(f32x4[f32; 4] => i32 | le[f32x4_le_test]:
          ([0., 2., 2., 4.], [0., 1., 2., 3.]) => [-1, 0, -1, -0]);
test_bop!(f64x2[f64; 2] => i64 | le[f64x2_le_test]: ([0., 2.], [0., 1.]) => [-1, 0]);

#[wasm_bindgen_test]
fn v128_bitwise_load_store() {
    unsafe {
        let mut arr: [i32; 4] = [0, 1, 2, 3];

        let vec = v128::load(arr.as_ptr() as *const v128);
        let vec = i32x4::add(vec, vec);
        v128::store(arr.as_mut_ptr() as *mut v128, vec);

        assert_eq!(arr, [0, 2, 4, 6]);
    }
}

test_uop!(f32x4[f32; 4] | neg[f32x4_neg_test]: [0., 1., 2., 3.] => [ 0., -1., -2., -3.]);
test_uop!(f32x4[f32; 4] | abs[f32x4_abs_test]: [0., -1., 2., -3.] => [ 0., 1., 2., 3.]);
test_bop!(f32x4[f32; 4] | min[f32x4_min_test]:
          ([0., -1., 7., 8.], [1., -3., -4., 10.]) => [0., -3., -4., 8.]);
test_bop!(f32x4[f32; 4] | min[f32x4_min_test_nan]:
          ([0., -1., 7., 8.], [1., -3., -4., std::f32::NAN])
          => [0., -3., -4., std::f32::NAN]);
test_bop!(f32x4[f32; 4] | max[f32x4_max_test]:
          ([0., -1., 7., 8.], [1., -3., -4., 10.]) => [1., -1., 7., 10.]);
test_bop!(f32x4[f32; 4] | max[f32x4_max_test_nan]:
          ([0., -1., 7., 8.], [1., -3., -4., std::f32::NAN])
          => [1., -1., 7., std::f32::NAN]);
test_bop!(f32x4[f32; 4] | add[f32x4_add_test]:
          ([0., -1., 7., 8.], [1., -3., -4., 10.]) => [1., -4., 3., 18.]);
test_bop!(f32x4[f32; 4] | sub[f32x4_sub_test]:
          ([0., -1., 7., 8.], [1., -3., -4., 10.]) => [-1., 2., 11., -2.]);
test_bop!(f32x4[f32; 4] | mul[f32x4_mul_test]:
          ([0., -1., 7., 8.], [1., -3., -4., 10.]) => [0., 3., -28., 80.]);
test_bop!(f32x4[f32; 4] | div[f32x4_div_test]:
          ([0., -8., 70., 8.], [1., 4., 10., 2.]) => [0., -2., 7., 4.]);

test_uop!(f64x2[f64; 2] | neg[f64x2_neg_test]: [0., 1.] => [ 0., -1.]);
test_uop!(f64x2[f64; 2] | abs[f64x2_abs_test]: [0., -1.] => [ 0., 1.]);
test_bop!(f64x2[f64; 2] | min[f64x2_min_test]:
          ([0., -1.], [1., -3.]) => [0., -3.]);
test_bop!(f64x2[f64; 2] | min[f64x2_min_test_nan]:
          ([7., 8.], [-4., std::f64::NAN])
          => [ -4., std::f64::NAN]);
test_bop!(f64x2[f64; 2] | max[f64x2_max_test]:
          ([0., -1.], [1., -3.]) => [1., -1.]);
test_bop!(f64x2[f64; 2] | max[f64x2_max_test_nan]:
          ([7., 8.], [ -4., std::f64::NAN])
          => [7., std::f64::NAN]);
test_bop!(f64x2[f64; 2] | add[f64x2_add_test]:
          ([0., -1.], [1., -3.]) => [1., -4.]);
test_bop!(f64x2[f64; 2] | sub[f64x2_sub_test]:
          ([0., -1.], [1., -3.]) => [-1., 2.]);
test_bop!(f64x2[f64; 2] | mul[f64x2_mul_test]:
          ([0., -1.], [1., -3.]) => [0., 3.]);
test_bop!(f64x2[f64; 2] | div[f64x2_div_test]:
          ([0., -8.], [1., 4.]) => [0., -2.]);

macro_rules! test_conv {
    ($test_id:ident | $conv_id:ident | $to_ty:ident | $from:expr,  $to:expr) => {
        #[wasm_bindgen_test]
        fn $test_id() {
            unsafe {
                let from: v128 = mem::transmute($from);
                let to: v128 = mem::transmute($to);

                let r: v128 = $to_ty::$conv_id(from);

                compare_bytes(r, to);
            }
        }
    };
}

test_conv!(
    f32x4_convert_s_i32x4 | convert_s_i32x4 | f32x4 | [1_i32, 2, 3, 4],
    [1_f32, 2., 3., 4.]
);
test_conv!(
    f32x4_convert_u_i32x4
        | convert_u_i32x4
        | f32x4
        | [u32::max_value(), 2, 3, 4],
    [u32::max_value() as f32, 2., 3., 4.]
);
test_conv!(
    f64x2_convert_s_i64x2 | convert_s_i64x2 | f64x2 | [1_i64, 2],
    [1_f64, 2.]
);
test_conv!(
    f64x2_convert_u_i64x2 | convert_u_i64x2 | f64x2 | [u64::max_value(), 2],
    [18446744073709552000.0, 2.]
);

// FIXME: this fails, and produces -2147483648 instead of saturating at
// i32::max_value() test_conv!(i32x4_trunc_s_f32x4_sat | trunc_s_f32x4_sat |
// i32x4 | [1_f32, 2., (i32::max_value() as f32 + 1.), 4.],
// [1_i32, 2, i32::max_value(), 4]); FIXME: add other saturating tests
