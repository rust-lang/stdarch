#![feature(stdsimd)]

#[macro_use]
extern crate coresimd;

use coresimd::simd::*;

#[test]
fn shuffle2() {
    let x = u8x2::new(3, 42);
    let e = u8x2::new(42, 3);
    let r = shuffle!(x, [1, 0]);
    assert_eq!(r, e);

    let y = u8x2::new(7, 12);
    let e = u8x2::new(42, 12);
    let r = shuffle!(x, y, [1, 3]);
    assert_eq!(r, e);

    let x = i16x4::new(1, 2, 3, 4);
    let e = i16x2::new(2, 4);
    let r = shuffle!(x, [1, 3]);
    assert_eq!(r, e);

    let y = i16x4::new(5, 6, 7, 8);
    let e = i16x2::new(2, 7);
    let r = shuffle!(x, y, [1, 6]);
    assert_eq!(r, e);
}

#[test]
fn shuffle4() {
    let x = u8x2::new(3, 42);
    let e = u8x4::new(42, 3, 42, 42);
    let r = shuffle!(x, [1, 0, 1, 1]);
    assert_eq!(r, e);

    let x = u32x4::new(1, 2, 3, 4);
    let e = u32x4::new(2, 4, 1, 3);
    let r = shuffle!(x, [1, 3, 0, 2]);
    assert_eq!(r, e);

    let y = u32x4::new(5, 6, 7, 8);
    let e = u32x4::new(3, 2, 6, 1);
    let r = shuffle!(x, y, [2, 1, 5, 0]);
    assert_eq!(r, e);

    let x = i32x8::new(1, 2, 3, 4, 7, 3, 2, 1);
    let e = i32x4::new(2, 7, 3, 3);
    let r = shuffle!(x, [1, 4, 5, 5]);
    assert_eq!(r, e);

    let y = i32x8::new(5, 6, 7, 8, 1, 5, 2, 3);
    let e = i32x4::new(3, 5, 7, 3);
    let r = shuffle!(x, y, [15, 13, 4, 5]);
    assert_eq!(r, e);
}

#[test]
fn shuffle8() {
    let x = f32x8::new(1., 2., 3., 4., 5., 6., 7., 8.);
    let e = f32x8::new(2., 8., 1., 3., 5., 2., 7., 4.);
    let r = shuffle!(x, [1, 7, 0, 2, 4, 1, 6, 3]);
    assert_eq!(r, e);

    let y = f32x8::new(51., 61., 71., 81., 11., 21., 31., 41.);
    let e = f32x8::new(2., 8., 51., 3., 71., 41., 7., 4.);
    let r = shuffle!(x, y, [1, 7, 8, 2, 10, 15, 6, 3]);
    assert_eq!(r, e);
}

#[test]
fn shuffle16() {
    #[cfg_attr(rustfmt, rustfmt_skip)]
    let x = u8x16::new(
        0, 1, 2, 3,
        4, 5, 6, 7,
        8, 9, 10, 11,
        12, 13, 14, 15,
    );
    #[cfg_attr(rustfmt, rustfmt_skip)]
    let y = u8x16::new(
        16, 17, 18, 19,
        20, 21, 22, 23,
        24, 25, 26, 27,
        28, 29, 30, 31
    );
    #[cfg_attr(rustfmt, rustfmt_skip)]
    let e = u8x16::new(
        0, 1, 2, 3,
        16, 17, 18, 19,
        8, 9, 10, 11,
        20, 21, 22, 23
    );
    #[cfg_attr(rustfmt, rustfmt_skip)]
    let r = shuffle!(
        x, y,
        [
            0, 1, 2, 3,
            16, 17, 18, 19,
            8, 9, 10, 11,
            20, 21, 22, 23
        ]
    );
    assert_eq!(r, e);
}

#[test]
fn shuffle32() {
    #[cfg_attr(rustfmt, rustfmt_skip)]
    let x = u8x32::new(
        0, 1, 2, 3,
        4, 5, 6, 7,
        8, 9, 10, 11,
        12, 13, 14, 15,
        16, 17, 18, 19,
        20, 21, 22, 23,
        24, 25, 26, 27,
        28, 29, 30, 31
    );
    #[cfg_attr(rustfmt, rustfmt_skip)]
    let y = u8x32::new(
        32, 33, 34, 35,
        36, 37, 38, 39,
        40, 41, 42, 43,
        44, 45, 46, 47,
        48, 49, 50, 51,
        52, 53, 54, 55,
        56, 57, 58, 59,
        60, 61, 62, 63,
    );
    #[cfg_attr(rustfmt, rustfmt_skip)]
    let e = u8x32::new(
        0, 1, 2, 3,
        32, 33, 34, 35,
        8, 9, 10, 11,
        36, 37, 38, 39,
        8, 9, 10, 11,
        40, 41, 42, 43,
        12, 13, 14, 15,
        44, 45, 46, 47
    );
    #[cfg_attr(rustfmt, rustfmt_skip)]
    let r = shuffle!(
        x, y,
        [
            0, 1, 2, 3,
            32, 33, 34, 35,
            8, 9, 10, 11,
            36, 37, 38, 39,
            8, 9, 10, 11,
            40, 41, 42, 43,
            12, 13, 14, 15,
            44, 45, 46, 47
        ]
    );
    assert_eq!(r, e);
}

#[test]
fn shuffle64() {
    #[cfg_attr(rustfmt, rustfmt_skip)]
    let x = u8x64::new(
        0, 1, 2, 3,
        4, 5, 6, 7,
        8, 9, 10, 11,
        12, 13, 14, 15,
        16, 17, 18, 19,
        20, 21, 22, 23,
        24, 25, 26, 27,
        28, 29, 30, 31,
        32, 33, 34, 35,
        36, 37, 38, 39,
        40, 41, 42, 43,
        44, 45, 46, 47,
        48, 49, 50, 51,
        52, 53, 54, 55,
        56, 57, 58, 59,
        60, 61, 62, 63,
    );
    #[cfg_attr(rustfmt, rustfmt_skip)]
    let y = u8x64::new(
        64, 65, 66, 67,
        68, 69, 70, 71,
        72, 73, 74, 75,
        76, 77, 78, 79,
        80, 81, 82, 83,
        84, 85, 86, 87,
        88, 89, 90, 91,
        92, 93, 94, 95,
        96, 97, 98, 99,
        100, 101, 102, 103,
        104, 105, 106, 107,
        108, 109, 110, 111,
        112, 113, 114, 115,
        116, 117, 118, 119,
        120, 121, 122, 123,
        124, 125, 126, 127,
    );
    #[cfg_attr(rustfmt, rustfmt_skip)]
    let e = u8x64::new(
        0, 1, 2, 3,
        64, 65, 66, 67,
        8, 9, 10, 11,
        68, 69, 70, 71,
        8, 9, 10, 11,
        72, 73, 74, 75,
        12, 13, 14, 15,
        76, 77, 78, 79,
        16, 17, 18, 19,
        80, 81, 82, 83,
        20, 21, 22, 23,
        84, 85, 86, 87,
        88, 89, 90, 91,
        24, 25, 26, 27,
        92, 93, 94, 95,
        28, 29, 30, 31
    );
    #[cfg_attr(rustfmt, rustfmt_skip)]
    let r = shuffle!(
        x, y,
        [
            0, 1, 2, 3,
            64, 65, 66, 67,
            8, 9, 10, 11,
            68, 69, 70, 71,
            8, 9, 10, 11,
            72, 73, 74, 75,
            12, 13, 14, 15,
            76, 77, 78, 79,
            16, 17, 18, 19,
            80, 81, 82, 83,
            20, 21, 22, 23,
            84, 85, 86, 87,
            88, 89, 90, 91,
            24, 25, 26, 27,
            92, 93, 94, 95,
            28, 29, 30, 31
        ]
    );
    assert_eq!(r, e);
}
