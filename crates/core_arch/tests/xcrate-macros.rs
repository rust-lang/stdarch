#![feature(stdsimd)]

#[test]
#[cfg(target_arch = "wasm32")]
fn wut() {
    unsafe {
        use core_arch::arch::wasm32;
        let a = wasm32::v128_const(0_u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b = wasm32::v128_const(
            16_u8, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        );

        let vec_r = wasm32::v8x16_shuffle!(
            a, b, 0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30,
        );

        let e = wasm32::v128_const(0_u8, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30);
        assert_eq!(wasm32::i8x16_all_true(wasm32::i8x16_eq(e, vec_r)), 1);
    }
}
