#![feature(stdsimd, avx512_target_feature)]

#[cfg(target_arch = "x86")]
use {core_arch::arch::x86::*};
#[cfg(target_arch = "x86_64")]
use {core_arch::arch::x86_64::*};


use rand::seq::SliceRandom;
use rand::thread_rng;
use rand::Rng;

use std::cmp;

use std::time::{Duration, Instant};

// types

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Color {
    Black,
    White,
    Empty,
}

type Square = i32; 
type Move = i32;
type Side = Color;
type Piece = Color;

// constants

const FILE_SIZE: i32 = 15;
const RANK_SIZE: i32 = 15;
const SQUARE_SIZE: i32 = (FILE_SIZE + 4) * (FILE_SIZE + 4 * 2 ) + 4;

const EVAL_INF: i32 = (FILE_SIZE * RANK_SIZE * 100);
const MOVE_NONE: Move = -1;
const SCORE_NONE: i32 = -EVAL_INF - 1;

const ENDCHECK: [[i32; 4]; 20] = [ [-4, -3, -2, -1],
                                   [-3, -2, -1,  1],
                                   [-2, -1,  1,  2],
                                   [-1,  1,  2,  3],
                                   [ 1,  2,  3,  4],

                             [1 * (-FILE_SIZE - 4), 2 * (-FILE_SIZE - 4), 3 * (-FILE_SIZE - 4), 4 * (-FILE_SIZE - 4)],
                             [1 * (-FILE_SIZE - 4), 2 * (-FILE_SIZE - 4), 3 * (-FILE_SIZE - 4), 1 * ( FILE_SIZE + 4)],
                             [1 * (-FILE_SIZE - 4), 2 * (-FILE_SIZE - 4), 1 * ( FILE_SIZE + 4), 2 * ( FILE_SIZE + 4)],
                             [1 * (-FILE_SIZE - 4), 1 * ( FILE_SIZE + 4), 2 * ( FILE_SIZE + 4), 3 * ( FILE_SIZE + 4)],
                             [1 * ( FILE_SIZE + 4), 2 * ( FILE_SIZE + 4), 3 * ( FILE_SIZE + 4), 4 * ( FILE_SIZE + 4)],

                             [1 * (-FILE_SIZE - 5), 2 * (-FILE_SIZE - 5), 3 * (-FILE_SIZE - 5), 4 * (-FILE_SIZE - 5)],
                             [1 * (-FILE_SIZE - 5), 2 * (-FILE_SIZE - 5), 3 * (-FILE_SIZE - 5), 1 * ( FILE_SIZE + 5)],
                             [1 * (-FILE_SIZE - 5), 2 * (-FILE_SIZE - 5), 1 * ( FILE_SIZE + 5), 2 * ( FILE_SIZE + 5)],
                             [1 * (-FILE_SIZE - 5), 1 * ( FILE_SIZE + 5), 2 * ( FILE_SIZE + 5), 3 * ( FILE_SIZE + 5)],
                             [1 * ( FILE_SIZE + 5), 2 * ( FILE_SIZE + 5), 3 * ( FILE_SIZE + 5), 4 * ( FILE_SIZE + 5)],

                             [1 * (-FILE_SIZE - 3), 2 * (-FILE_SIZE - 3), 3 * (-FILE_SIZE - 3), 4 * (-FILE_SIZE - 3)],
                             [1 * (-FILE_SIZE - 3), 2 * (-FILE_SIZE - 3), 3 * (-FILE_SIZE - 3), 1 * ( FILE_SIZE + 3)],
                             [1 * (-FILE_SIZE - 3), 2 * (-FILE_SIZE - 3), 1 * ( FILE_SIZE + 3), 2 * ( FILE_SIZE + 3)],
                             [1 * (-FILE_SIZE - 3), 1 * ( FILE_SIZE + 3), 2 * ( FILE_SIZE + 3), 3 * ( FILE_SIZE + 3)],
                             [1 * ( FILE_SIZE + 3), 2 * ( FILE_SIZE + 3), 3 * ( FILE_SIZE + 3), 4 * ( FILE_SIZE + 3)]
                           ];

const PATTERNFILE4: [i32; 7] = [0, 1, 2, 3, 4, 5, 6];
const PATTERNRANK4: [i32; 7] = [0, 1 * (FILE_SIZE + 4), 2 * (FILE_SIZE + 4), 3 * (FILE_SIZE + 4), 4 * (FILE_SIZE + 4), 5 * (FILE_SIZE + 4), 6 * (FILE_SIZE + 4)];
const PATTERNDIAL4: [i32; 7] = [0, 1 * (FILE_SIZE + 5), 2 * (FILE_SIZE + 5), 3 * (FILE_SIZE + 5), 4 * (FILE_SIZE + 5), 5 * (FILE_SIZE + 5), 6 * (FILE_SIZE + 5)];
const PATTERNDIAR4: [i32; 7] = [0, 1 * (FILE_SIZE + 3), 2 * (FILE_SIZE + 3), 3 * (FILE_SIZE + 3), 4 * (FILE_SIZE + 3), 5 * (FILE_SIZE + 3), 6 * (FILE_SIZE + 3)];

const MAPMOVEVALUE: [[i32; 367]; 4] = [ [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0, 0, 1<<31, 1<<30, 1<<29, 1<<28, 1<<27, 1<<26, 1<<25, 1<<24, 1<<23, 1<<22, 1<<21. 1<<20, 1<<19, 1<<18, 1<<17,
                                         0, 0, 0, 0, 1<<31, 1<<30, 1<<29, 1<<28, 1<<27, 1<<26, 1<<25, 1<<24, 1<<23, 1<<22, 1<<21. 1<<20, 1<<19, 1<<18, 1<<17,
                                         0, 0, 0, 0, 1<<31, 1<<30, 1<<29, 1<<28, 1<<27, 1<<26, 1<<25, 1<<24, 1<<23, 1<<22, 1<<21. 1<<20, 1<<19, 1<<18, 1<<17,
                                         0, 0, 0, 0, 1<<31, 1<<30, 1<<29, 1<<28, 1<<27, 1<<26, 1<<25, 1<<24, 1<<23, 1<<22, 1<<21. 1<<20, 1<<19, 1<<18, 1<<17,
                                         0, 0, 0, 0, 1<<31, 1<<30, 1<<29, 1<<28, 1<<27, 1<<26, 1<<25, 1<<24, 1<<23, 1<<22, 1<<21. 1<<20, 1<<19, 1<<18, 1<<17,
                                         0, 0, 0, 0, 1<<31, 1<<30, 1<<29, 1<<28, 1<<27, 1<<26, 1<<25, 1<<24, 1<<23, 1<<22, 1<<21. 1<<20, 1<<19, 1<<18, 1<<17,
                                         0, 0, 0, 0, 1<<31, 1<<30, 1<<29, 1<<28, 1<<27, 1<<26, 1<<25, 1<<24, 1<<23, 1<<22, 1<<21. 1<<20, 1<<19, 1<<18, 1<<17,
                                         0, 0, 0, 0, 1<<31, 1<<30, 1<<29, 1<<28, 1<<27, 1<<26, 1<<25, 1<<24, 1<<23, 1<<22, 1<<21. 1<<20, 1<<19, 1<<18, 1<<17,
                                         0, 0, 0, 0, 1<<31, 1<<30, 1<<29, 1<<28, 1<<27, 1<<26, 1<<25, 1<<24, 1<<23, 1<<22, 1<<21. 1<<20, 1<<19, 1<<18, 1<<17,
                                         0, 0, 0, 0, 1<<31, 1<<30, 1<<29, 1<<28, 1<<27, 1<<26, 1<<25, 1<<24, 1<<23, 1<<22, 1<<21. 1<<20, 1<<19, 1<<18, 1<<17,
                                         0, 0, 0, 0, 1<<31, 1<<30, 1<<29, 1<<28, 1<<27, 1<<26, 1<<25, 1<<24, 1<<23, 1<<22, 1<<21. 1<<20, 1<<19, 1<<18, 1<<17,
                                         0, 0, 0, 0, 1<<31, 1<<30, 1<<29, 1<<28, 1<<27, 1<<26, 1<<25, 1<<24, 1<<23, 1<<22, 1<<21. 1<<20, 1<<19, 1<<18, 1<<17,
                                         0, 0, 0, 0, 1<<31, 1<<30, 1<<29, 1<<28, 1<<27, 1<<26, 1<<25, 1<<24, 1<<23, 1<<22, 1<<21. 1<<20, 1<<19, 1<<18, 1<<17,
                                         0, 0, 0, 0, 1<<31, 1<<30, 1<<29, 1<<28, 1<<27, 1<<26, 1<<25, 1<<24, 1<<23, 1<<22, 1<<21. 1<<20, 1<<19, 1<<18, 1<<17,
                                         0, 0, 0, 0, 1<<31, 1<<30, 1<<29, 1<<28, 1<<27, 1<<26, 1<<25, 1<<24, 1<<23, 1<<22, 1<<21. 1<<20, 1<<19, 1<<18, 1<<17],

                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0, 0, 1<<31, 1<<31, 1<<31, 1<<31, 1<<31, 1<<31, 1<<31, 1<<31, 1<<31, 1<<31, 1<<31, 1<<31, 1<<31, 1<<31, 1<<31,
                                         0, 0, 0, 0, 1<<30, 1<<30, 1<<30, 1<<30, 1<<30, 1<<30, 1<<30, 1<<30, 1<<30, 1<<30, 1<<30, 1<<30, 1<<30, 1<<30, 1<<30,
                                         0, 0, 0, 0, 1<<29, 1<<29, 1<<29, 1<<29, 1<<29, 1<<29, 1<<29, 1<<29, 1<<29, 1<<29, 1<<29, 1<<29, 1<<29, 1<<29, 1<<29,
                                         0, 0, 0, 0, 1<<28, 1<<28, 1<<28, 1<<28, 1<<28, 1<<28, 1<<28, 1<<28, 1<<28, 1<<28, 1<<28, 1<<28, 1<<28, 1<<28, 1<<28,
                                         0, 0, 0, 0, 1<<27, 1<<27, 1<<27, 1<<27, 1<<27, 1<<27, 1<<27, 1<<27, 1<<27, 1<<27, 1<<27, 1<<27, 1<<27, 1<<27, 1<<27,
                                         0, 0, 0, 0, 1<<26, 1<<26, 1<<26, 1<<26, 1<<26, 1<<26, 1<<26, 1<<26, 1<<26, 1<<26, 1<<26, 1<<26, 1<<26, 1<<26, 1<<26,
                                         0, 0, 0, 0, 1<<25, 1<<25, 1<<25, 1<<25, 1<<25, 1<<25, 1<<25, 1<<25, 1<<25, 1<<25, 1<<25, 1<<25, 1<<25, 1<<25, 1<<25,
                                         0, 0, 0, 0, 1<<24, 1<<24, 1<<24, 1<<24, 1<<24, 1<<24, 1<<24, 1<<24, 1<<24, 1<<24, 1<<24, 1<<24, 1<<24, 1<<24, 1<<24,
                                         0, 0, 0, 0, 1<<23, 1<<23, 1<<23, 1<<23, 1<<23, 1<<23, 1<<23, 1<<23, 1<<23, 1<<23, 1<<23, 1<<23, 1<<23, 1<<23, 1<<23,
                                         0, 0, 0, 0, 1<<22, 1<<22, 1<<22, 1<<22, 1<<22, 1<<22, 1<<22, 1<<22, 1<<22, 1<<22, 1<<22, 1<<22, 1<<22, 1<<22, 1<<22,
                                         0, 0, 0, 0, 1<<21, 1<<21, 1<<21, 1<<21, 1<<21, 1<<21, 1<<21, 1<<21, 1<<21, 1<<21, 1<<21, 1<<21, 1<<21, 1<<21, 1<<21,
                                         0, 0, 0, 0, 1<<20, 1<<20, 1<<20, 1<<20, 1<<20, 1<<20, 1<<20, 1<<20, 1<<20, 1<<20, 1<<20, 1<<20, 1<<20, 1<<20, 1<<20,
                                         0, 0, 0, 0, 1<<19, 1<<19, 1<<19, 1<<19, 1<<19, 1<<19, 1<<19, 1<<19, 1<<19, 1<<19, 1<<19, 1<<19, 1<<19, 1<<19, 1<<19,
                                         0, 0, 0, 0, 1<<18, 1<<18, 1<<18, 1<<18, 1<<18, 1<<18, 1<<18, 1<<18, 1<<18, 1<<18, 1<<18, 1<<18, 1<<18, 1<<18, 1<<18,
                                         0, 0, 0, 0, 1<<17, 1<<17, 1<<17, 1<<17, 1<<17, 1<<17, 1<<17, 1<<17, 1<<17, 1<<17, 1<<17, 1<<17, 1<<17, 1<<17, 1<<17],

                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0, 0, 1<<31, 1<<31, 1<<31, 1<<31, 1<<31, 1<<31, 1<<31, 1<<31, 1<<31, 1<<31, 1<<31, 0,     0,     0,     0,
                                         0, 0, 0, 0, 1<<31, 1<<30, 1<<30, 1<<30, 1<<30, 1<<30, 1<<30, 1<<30, 1<<30, 1<<30, 1<<30, 1<<30, 0,     0,     0,
                                         0, 0, 0, 0, 1<<31, 1<<30, 1<<29, 1<<29, 1<<29, 1<<29, 1<<29, 1<<29, 1<<29, 1<<29, 1<<29, 1<<29, 1<<29, 0,     0,
                                         0, 0, 0, 0, 1<<31, 1<<30, 1<<29, 1<<28, 1<<28, 1<<28, 1<<28, 1<<28, 1<<28, 1<<28, 1<<28, 1<<28, 1<<28, 1<<28, 0,
                                         0, 0, 0, 0, 1<<31, 1<<30, 1<<29, 1<<28, 1<<27, 1<<27, 1<<27, 1<<27, 1<<27, 1<<27, 1<<27, 1<<27, 1<<27, 1<<27, 1<<27,
                                         0, 0, 0, 0, 1<<31, 1<<30, 1<<29, 1<<28, 1<<27, 1<<26, 1<<26, 1<<26, 1<<26, 1<<26, 1<<26, 1<<26, 1<<26, 1<<26, 1<<26,
                                         0, 0, 0, 0, 1<<31, 1<<30, 1<<29, 1<<28, 1<<27, 1<<26, 1<<25, 1<<25, 1<<25, 1<<25, 1<<25, 1<<25, 1<<25, 1<<25, 1<<25,
                                         0, 0, 0, 0, 1<<31, 1<<30, 1<<29, 1<<28, 1<<27, 1<<26, 1<<25, 1<<24, 1<<24, 1<<24, 1<<24, 1<<24, 1<<24, 1<<24, 1<<24,
                                         0, 0, 0, 0, 1<<31, 1<<30, 1<<29, 1<<28, 1<<27, 1<<26, 1<<25, 1<<24, 1<<23, 1<<23, 1<<23, 1<<23, 1<<23, 1<<23, 1<<23,
                                         0, 0, 0, 0, 1<<31, 1<<30, 1<<29, 1<<28, 1<<27, 1<<26, 1<<25, 1<<24, 1<<23, 1<<22, 1<<22, 1<<22, 1<<22, 1<<22, 1<<22,
                                         0, 0, 0, 0, 1<<31, 1<<30, 1<<29, 1<<28, 1<<27, 1<<26, 1<<25, 1<<24, 1<<23, 1<<22, 1<<21, 1<<21, 1<<21, 1<<21, 1<<21,
                                         0, 0, 0, 0, 0,     1<<30, 1<<29, 1<<28, 1<<27, 1<<26, 1<<25, 1<<24, 1<<23, 1<<22, 1<<21, 1<<20, 1<<20, 1<<20, 1<<20,
                                         0, 0, 0, 0, 0,     0,     1<<29, 1<<28, 1<<27, 1<<26, 1<<25, 1<<24, 1<<23, 1<<22, 1<<21, 1<<20, 1<<19, 1<<19, 1<<19,
                                         0, 0, 0, 0, 0,     0,     0,     1<<28, 1<<27, 1<<26, 1<<25, 1<<24, 1<<23, 1<<22, 1<<21, 1<<20, 1<<19, 1<<18, 1<<18,
                                         0, 0, 0, 0, 0,     0,     0,     0,     1<<27, 1<<26, 1<<25, 1<<24, 1<<23, 1<<22, 1<<21, 1<<20, 1<<19, 1<<18, 1<<17],

                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0, 0, 0,     0,     0,     0,     1<<31, 1<<31, 1<<31, 1<<31, 1<<31, 1<<31, 1<<31, 1<<31, 1<<31, 1<<31, 1<<31,
                                         0, 0, 0, 0, 0,     0,     0,     1<<30, 1<<30, 1<<30, 1<<30, 1<<30, 1<<30, 1<<30, 1<<30, 1<<30, 1<<30, 1<<30, 1<<31,
                                         0, 0, 0, 0, 0,     0,     1<<29, 1<<29, 1<<29, 1<<29, 1<<29, 1<<29, 1<<29, 1<<29, 1<<29, 1<<29, 1<<29, 1<<30, 1<<31,
                                         0, 0, 0, 0, 0,     1<<28, 1<<28, 1<<28, 1<<28, 1<<28, 1<<28, 1<<28, 1<<28, 1<<28, 1<<28, 1<<28, 1<<29, 1<<30, 1<<31,
                                         0, 0, 0, 0, 1<<27, 1<<27, 1<<27, 1<<27, 1<<27, 1<<27, 1<<27, 1<<27, 1<<27, 1<<27, 1<<27, 1<<28, 1<<29, 1<<30, 1<<31,
                                         0, 0, 0, 0, 1<<26, 1<<26, 1<<26, 1<<26, 1<<26, 1<<26, 1<<26, 1<<26, 1<<26, 1<<26, 1<<27, 1<<28, 1<<29, 1<<30, 1<<31,
                                         0, 0, 0, 0, 1<<25, 1<<25, 1<<25, 1<<25, 1<<25, 1<<25, 1<<25, 1<<25, 1<<25, 1<<26, 1<<27, 1<<28, 1<<29, 1<<30, 1<<31,
                                         0, 0, 0, 0, 1<<24, 1<<24, 1<<24, 1<<24, 1<<24, 1<<24, 1<<24, 1<<24, 1<<25, 1<<26, 1<<27, 1<<28, 1<<29, 1<<30, 1<<31,
                                         0, 0, 0, 0, 1<<23, 1<<23, 1<<23, 1<<23, 1<<23, 1<<23, 1<<23, 1<<24, 1<<25, 1<<26, 1<<27, 1<<28, 1<<29, 1<<30, 1<<31,
                                         0, 0, 0, 0, 1<<22, 1<<22, 1<<22, 1<<22, 1<<22, 1<<22, 1<<23, 1<<24, 1<<25, 1<<26, 1<<27, 1<<28, 1<<29, 1<<30, 1<<31,
                                         0, 0, 0, 0, 1<<21, 1<<21, 1<<21, 1<<21, 1<<21, 1<<22, 1<<23, 1<<24, 1<<25, 1<<26, 1<<27, 1<<28, 1<<29, 1<<30, 1<<31,
                                         0, 0, 0, 0, 1<<20, 1<<20, 1<<20, 1<<20, 1<<21, 1<<22, 1<<23, 1<<24, 1<<25, 1<<26, 1<<27, 1<<28, 1<<29, 1<<30, 0,
                                         0, 0, 0, 0, 1<<19, 1<<19, 1<<19, 1<<20, 1<<21, 1<<22, 1<<23, 1<<24, 1<<25, 1<<26, 1<<27, 1<<28, 1<<29, 0,     0,
                                         0, 0, 0, 0, 1<<18, 1<<18, 1<<19, 1<<20, 1<<21, 1<<22, 1<<23, 1<<24, 1<<25, 1<<26, 1<<27, 1<<28, 0,     0,     0,
                                         0, 0, 0, 0, 1<<17, 1<<18, 1<<19, 1<<20, 1<<21, 1<<22, 1<<23, 1<<24, 1<<25, 1<<26, 1<<27, 0,     0,     0,     0] 
                                        ];

const MAPMOVEIDX: [[i32; 367]; 4] = [ [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0,    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                       0, 0, 0, 0,    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                       0, 0, 0, 0,    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                                       0, 0, 0, 0,    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                                       0, 0, 0, 0,    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
                                       0, 0, 0, 0,    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                                       0, 0, 0, 0,    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                                       0, 0, 0, 0,    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
                                       0, 0, 0, 0,    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
                                       0, 0, 0, 0,    10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
                                       0, 0, 0, 0,    11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
                                       0, 0, 0, 0,    12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
                                       0, 0, 0, 0,    13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
                                       0, 0, 0, 0,    14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14],

                                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0,    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                                       0, 0, 0, 0,    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                                       0, 0, 0, 0,    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                                       0, 0, 0, 0,    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                                       0, 0, 0, 0,    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                                       0, 0, 0, 0,    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                                       0, 0, 0, 0,    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                                       0, 0, 0, 0,    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                                       0, 0, 0, 0,    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                                       0, 0, 0, 0,    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                                       0, 0, 0, 0,    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                                       0, 0, 0, 0,    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                                       0, 0, 0, 0,    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                                       0, 0, 0, 0,    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                                       0, 0, 0, 0,    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],

                                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0,    10,  9, 8,  7,  6,  5,  4,  3,  2,  1,  0,  0,  0,  0,  0,
                                       0, 0, 0, 0,    11, 10, 9,  8,  7,  6,  5,  4,  3,  2,  1,  0,  0,  0,  0,
                                       0, 0, 0, 0,    12, 11, 10, 9,  8,  7,  6,  5,  4,  3,  2,  1,  0,  0,  0,
                                       0, 0, 0, 0,    13, 12, 11, 10, 9,  8,  7,  6,  5,  4,  3,  2,  1,  0,  0,
                                       0, 0, 0, 0,    14, 13, 12, 11, 10, 9,  8,  7,  6,  5,  4,  3,  2,  1,  0,
                                       0, 0, 0, 0,    15, 14, 13, 6,  11, 10, 9,  8,  7,  6,  5,  4,  3,  2,  1,
                                       0, 0, 0, 0,    16, 15, 14, 13, 12, 11, 10, 9,  8,  7,  6,  5,  4,  3,  2,
                                       0, 0, 0, 0,    17, 16, 15, 14, 13, 12, 11, 10, 9,  8,  7,  6,  5,  4,  3,
                                       0, 0, 0, 0,    18, 17, 16, 15, 14, 13, 12, 11, 10, 9,  8,  7,  6,  5,  4,
                                       0, 0, 0, 0,    19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9,  8,  7,  6,  5,
                                       0, 0, 0, 0,    20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9,  8,  7,  6,
                                       0, 0, 0, 0,    0,  20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9,  8,  7,
                                       0, 0, 0, 0,    0,  0,  20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9,  8,
                                       0, 0, 0, 0,    0,  0,  0,  20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9
                                       0, 0, 0, 0,    0,  0,  0,  0,  20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10],

                                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0,    0,  0,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                                       0, 0, 0, 0,    0,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                                       0, 0, 0, 0,    0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                       0, 0, 0, 0,    0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
                                       0, 0, 0, 0,    0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                                       0, 0, 0, 0,    1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                       0, 0, 0, 0,    2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
                                       0, 0, 0, 0,    3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
                                       0, 0, 0, 0,    4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18,
                                       0, 0, 0, 0,    5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                                       0, 0, 0, 0,    6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                       0, 0, 0, 0,    7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,  0,
                                       0, 0, 0, 0,    8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,  0,  0,
                                       0, 0, 0, 0,    9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,  0,  0,  0,
                                       0, 0, 0, 0,    10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,  0,  0,  0,  0]
                                ];

// variables

static mut Endgame: bool = false;

// structures

pub struct Pos { // position
    state: [Color; SQUARE_SIZE as usize],
    p_turn: Side,
    p_last: Move,

    bitboard: [[[i32; 20]; 4]; 3], 

}

impl Pos {

    pub fn init(&mut self) { // starting position
        for i in 0..SQUARE_SIZE as usize {
            self.state[i] = Color::Empty;
        }

        self.p_turn = Color::Black;
        self.p_last = square_make(0, 0);

        //--------------------------------------------

        for i in 0..4 {
            for j in 0..20 { 
                self.bitboard[Color::Black as usize][i][j] = 0; 
            }
        }

        for i in 0..4 {
            for j in 0..20 { 
                self.bitboard[Color::White as usize][i][j] = 0;
            }
        }

        for i in 0..2 {
            for j in 0..20 { 
                self.bitboard[Color::Empty as usize][i][j] = (1<<31)|(1<<30)|(1<<29)|(1<<28)|(1<<27)|(1<<26)|(1<<25)|(1<<24)|(1<<23)|(1<<22)|(1<<21)|(1<<20)|(1<<19)|(1<<18)|(1<<17); 
            }
        }
        
        self.bitboard[Color::Empty as usize][2][0]  = (1<<31)|(1<<30)|(1<<29)|(1<<28)|(1<<27);
        self.bitboard[Color::Empty as usize][2][1]  = (1<<31)|(1<<30)|(1<<29)|(1<<28)|(1<<27)|(1<<26);
        self.bitboard[Color::Empty as usize][2][2]  = (1<<31)|(1<<30)|(1<<29)|(1<<28)|(1<<27)|(1<<26)|(1<<25);
        self.bitboard[Color::Empty as usize][2][3]  = (1<<31)|(1<<30)|(1<<29)|(1<<28)|(1<<27)|(1<<26)|(1<<25)|(1<<24);
        self.bitboard[Color::Empty as usize][2][4]  = (1<<31)|(1<<30)|(1<<29)|(1<<28)|(1<<27)|(1<<26)|(1<<25)|(1<<24)|(1<<23);
        self.bitboard[Color::Empty as usize][2][5]  = (1<<31)|(1<<30)|(1<<29)|(1<<28)|(1<<27)|(1<<26)|(1<<25)|(1<<24)|(1<<23)|(1<<22);
        self.bitboard[Color::Empty as usize][2][6]  = (1<<31)|(1<<30)|(1<<29)|(1<<28)|(1<<27)|(1<<26)|(1<<25)|(1<<24)|(1<<23)|(1<<22)|(1<<21);
        self.bitboard[Color::Empty as usize][2][7]  = (1<<31)|(1<<30)|(1<<29)|(1<<28)|(1<<27)|(1<<26)|(1<<25)|(1<<24)|(1<<23)|(1<<22)|(1<<21)|(1<<20);
        self.bitboard[Color::Empty as usize][2][8]  = (1<<31)|(1<<30)|(1<<29)|(1<<28)|(1<<27)|(1<<26)|(1<<25)|(1<<24)|(1<<23)|(1<<22)|(1<<21)|(1<<20)|(1<<19);
        self.bitboard[Color::Empty as usize][2][9]  = (1<<31)|(1<<30)|(1<<29)|(1<<28)|(1<<27)|(1<<26)|(1<<25)|(1<<24)|(1<<23)|(1<<22)|(1<<21)|(1<<20)|(1<<19)|(1<<18);
        self.bitboard[Color::Empty as usize][2][10] = (1<<31)|(1<<30)|(1<<29)|(1<<28)|(1<<27)|(1<<26)|(1<<25)|(1<<24)|(1<<23)|(1<<22)|(1<<21)|(1<<20)|(1<<19)|(1<<18)|(1<<17);
        self.bitboard[Color::Empty as usize][2][11] = (1<<31)|(1<<30)|(1<<29)|(1<<28)|(1<<27)|(1<<26)|(1<<25)|(1<<24)|(1<<23)|(1<<22)|(1<<21)|(1<<20)|(1<<19)|(1<<18);
        self.bitboard[Color::Empty as usize][2][12] = (1<<31)|(1<<30)|(1<<29)|(1<<28)|(1<<27)|(1<<26)|(1<<25)|(1<<24)|(1<<23)|(1<<22)|(1<<21)|(1<<20);
        self.bitboard[Color::Empty as usize][2][13] = (1<<31)|(1<<30)|(1<<29)|(1<<28)|(1<<27)|(1<<26)|(1<<25)|(1<<24)|(1<<23)|(1<<22)|(1<<21);
        self.bitboard[Color::Empty as usize][2][14] = (1<<31)|(1<<30)|(1<<29)|(1<<28)|(1<<27)|(1<<26)|(1<<25)|(1<<24)|(1<<23)|(1<<22);
        self.bitboard[Color::Empty as usize][2][15] = (1<<31)|(1<<30)|(1<<29)|(1<<28)|(1<<27)|(1<<26)|(1<<25)|(1<<24)|(1<<23);
        self.bitboard[Color::Empty as usize][2][16] = (1<<31)|(1<<30)|(1<<29)|(1<<28)|(1<<27)|(1<<26)|(1<<25)|(1<<24);
        self.bitboard[Color::Empty as usize][2][17] = (1<<31)|(1<<30)|(1<<29)|(1<<28)|(1<<27)|(1<<26)|(1<<25);
        self.bitboard[Color::Empty as usize][2][18] = (1<<31)|(1<<30)|(1<<29)|(1<<28)|(1<<27)|(1<<26);
        self.bitboard[Color::Empty as usize][2][19] = (1<<31)|(1<<30)|(1<<29)|(1<<28)|(1<<27);

        self.bitboard[Color::Empty as usize][3][0]  = (1<<31)|(1<<30)|(1<<29)|(1<<28)|(1<<27);
        self.bitboard[Color::Empty as usize][3][1]  = (1<<31)|(1<<30)|(1<<29)|(1<<28)|(1<<27)|(1<<26);
        self.bitboard[Color::Empty as usize][3][2]  = (1<<31)|(1<<30)|(1<<29)|(1<<28)|(1<<27)|(1<<26)|(1<<25);
        self.bitboard[Color::Empty as usize][3][3]  = (1<<31)|(1<<30)|(1<<29)|(1<<28)|(1<<27)|(1<<26)|(1<<25)|(1<<24);
        self.bitboard[Color::Empty as usize][3][4]  = (1<<31)|(1<<30)|(1<<29)|(1<<28)|(1<<27)|(1<<26)|(1<<25)|(1<<24)|(1<<23);
        self.bitboard[Color::Empty as usize][3][5]  = (1<<31)|(1<<30)|(1<<29)|(1<<28)|(1<<27)|(1<<26)|(1<<25)|(1<<24)|(1<<23)|(1<<22);
        self.bitboard[Color::Empty as usize][3][6]  = (1<<31)|(1<<30)|(1<<29)|(1<<28)|(1<<27)|(1<<26)|(1<<25)|(1<<24)|(1<<23)|(1<<22)|(1<<21);
        self.bitboard[Color::Empty as usize][3][7]  = (1<<31)|(1<<30)|(1<<29)|(1<<28)|(1<<27)|(1<<26)|(1<<25)|(1<<24)|(1<<23)|(1<<22)|(1<<21)|(1<<20);
        self.bitboard[Color::Empty as usize][3][8]  = (1<<31)|(1<<30)|(1<<29)|(1<<28)|(1<<27)|(1<<26)|(1<<25)|(1<<24)|(1<<23)|(1<<22)|(1<<21)|(1<<20)|(1<<19);
        self.bitboard[Color::Empty as usize][3][9]  = (1<<31)|(1<<30)|(1<<29)|(1<<28)|(1<<27)|(1<<26)|(1<<25)|(1<<24)|(1<<23)|(1<<22)|(1<<21)|(1<<20)|(1<<19)|(1<<18);
        self.bitboard[Color::Empty as usize][3][10] = (1<<31)|(1<<30)|(1<<29)|(1<<28)|(1<<27)|(1<<26)|(1<<25)|(1<<24)|(1<<23)|(1<<22)|(1<<21)|(1<<20)|(1<<19)|(1<<18)|(1<<17);
        self.bitboard[Color::Empty as usize][3][11] = (1<<31)|(1<<30)|(1<<29)|(1<<28)|(1<<27)|(1<<26)|(1<<25)|(1<<24)|(1<<23)|(1<<22)|(1<<21)|(1<<20)|(1<<19)|(1<<18);
        self.bitboard[Color::Empty as usize][3][12] = (1<<31)|(1<<30)|(1<<29)|(1<<28)|(1<<27)|(1<<26)|(1<<25)|(1<<24)|(1<<23)|(1<<22)|(1<<21)|(1<<20);
        self.bitboard[Color::Empty as usize][3][13] = (1<<31)|(1<<30)|(1<<29)|(1<<28)|(1<<27)|(1<<26)|(1<<25)|(1<<24)|(1<<23)|(1<<22)|(1<<21);
        self.bitboard[Color::Empty as usize][3][14] = (1<<31)|(1<<30)|(1<<29)|(1<<28)|(1<<27)|(1<<26)|(1<<25)|(1<<24)|(1<<23)|(1<<22);
        self.bitboard[Color::Empty as usize][3][15] = (1<<31)|(1<<30)|(1<<29)|(1<<28)|(1<<27)|(1<<26)|(1<<25)|(1<<24)|(1<<23);
        self.bitboard[Color::Empty as usize][3][16] = (1<<31)|(1<<30)|(1<<29)|(1<<28)|(1<<27)|(1<<26)|(1<<25)|(1<<24);
        self.bitboard[Color::Empty as usize][3][17] = (1<<31)|(1<<30)|(1<<29)|(1<<28)|(1<<27)|(1<<26)|(1<<25);
        self.bitboard[Color::Empty as usize][3][18] = (1<<31)|(1<<30)|(1<<29)|(1<<28)|(1<<27)|(1<<26);
        self.bitboard[Color::Empty as usize][3][19] = (1<<31)|(1<<30)|(1<<29)|(1<<28)|(1<<27);
    } 

    pub fn do_move(&mut self, mv: Move) {

        let atk: Side = self.p_turn;
        let def: Side = side_opp(atk);

        match self.p_turn {
            Color::Black => { self.state[mv as usize] = Color::Black;

                              for i in 0..4 {
                                  self.bitboard[Color::Black as usize][i][ MAPMOVEIDX[i][mv as usize] as usize ] |= MAPMOVEVALUE[i][mv as usize];
                                  self.bitboard[Color::Empty as usize][i][ MAPMOVEIDX[i][mv as usize] as usize ] ^= MAPMOVEVALUE[i][mv as usize];
                              }
            },

            Color::White => { self.state[mv as usize] = Color::White;

                              for i in 0..4 {
                                  self.bitboard[Color::White as usize][i][ MAPMOVEIDX[i][mv as usize] as usize ] |= MAPMOVEVALUE[i][mv as usize];
                                  self.bitboard[Color::Empty as usize][i][ MAPMOVEIDX[i][mv as usize] as usize ] ^= MAPMOVEVALUE[i][mv as usize];
                              }
            },

            Color::Empty => {},
        }

        self.p_last = mv;

        self.p_turn = def; 
    }

    fn turn(&self) -> Side {
        self.p_turn
    }

    pub fn can_play(&self, from: Square) -> bool {

        if self.state[from as usize] == Color::Empty { true } else { false }
    }

    pub fn count(&self, pc: Piece) -> i32 {
        
        let mut n: i32 = 0;

        for rk in 0..RANK_SIZE {
            for fl in 0..FILE_SIZE {
                let sq: Square = square_make(fl, rk);
                if self.state[sq as usize] == pc { n += 1; }
            }
        }
        n
    }
}

pub struct List {  // legal move list
    p_move: [Move; (FILE_SIZE * RANK_SIZE) as usize],
    p_size: i32,
}

impl List {
    
    pub fn clear(&mut self) {
        self.p_size = 0;
    }

    pub fn add(&mut self, mv: Move) {
        self.p_move[self.p_size as usize] = mv;
        self.p_size += 1;
    }

    pub fn size(&self) -> i32 {
        self.p_size
    }

    pub fn shuffle(&mut self) {

        let mut rng = thread_rng();

        let num = self.p_size;
        
        let mut new_move: Vec<Move> = vec![];

        for x in 0..(num as usize) {
            new_move.push(self.p_move[x]);
        }

        new_move.shuffle(&mut rng);

        for x in 0..(self.p_size as usize) {
            self.p_move[x] = new_move[x];
        }
    }

    //pub fn move(&self, i: i32) -> Move {
    //    self.p_move[i as usize]
    //}
}

// functions
//
fn square_make(fl: i32, rk: i32) -> Square {
    (rk + 4) * (FILE_SIZE + 4) + (fl + 4)
}

fn square_file(sq: Square) -> i32 {
    sq % (FILE_SIZE + 4) - 4
}

fn square_rank(sq: Square) -> i32 {
    sq / (FILE_SIZE + 4) - 4
}

fn side_opp(sd: Side) -> Side {

    let mut out: Side; 

    match sd {
        Side::White => out = Side::Black,
        Side::Black => out = Side::White,
        Side::Empty => panic!(""),
    }

    out
}

fn pos_is_winner(pos : &Pos) -> bool {

    let current_side = side_opp(pos.p_turn);

    let mut found : bool = true;
    
    for x in 0..20 {
        for y in 0..4 {

            found = true;

            let adj = pos.p_last + ENDCHECK[x][y];
      
            if pos.state[adj as usize] != current_side { found = false; break }
        }
        if found == true { break; } 
    }

    found
}

fn pos_is_winner_scan(pos : &Pos) -> bool {

   let current_side = side_opp(pos.p_turn);

   if check_patternfile5(&pos, current_side) ||
      check_patternrank5(&pos, current_side) ||
      check_patterndial5(&pos, current_side) ||
      check_patterndiar5(&pos, current_side) { return true }
    
   false
}

fn pos_is_draw(pos : &Pos) -> bool {

    
    let mut found : bool = true;
        
    for rk in 0..RANK_SIZE {
        for fl in 0..FILE_SIZE {

            let sq: Square = square_make(fl, rk);
            if  pos.can_play(sq) {
                found = false;
                break;
            }

        if found == false { break;}
        }
    }
/*
    
    let mut test: bool = false;

    if pos.bitboard[Color::Empty as usize][0][0] == 0 &&
       pos.bitboard[Color::Empty as usize][0][1] == 0 &&
       pos.bitboard[Color::Empty as usize][0][2] == 0 &&
       pos.bitboard[Color::Empty as usize][0][3] == 0 &&
       pos.bitboard[Color::Empty as usize][0][4] == 0 &&
       pos.bitboard[Color::Empty as usize][0][5] == 0 &&
       pos.bitboard[Color::Empty as usize][0][6] == 0 &&
       pos.bitboard[Color::Empty as usize][0][7] == 0 &&
       pos.bitboard[Color::Empty as usize][0][8] == 0 { test = true; } else { test = false; }
*/
   //if test != found { println!("bitboard!!!!!!!!!!!!!!!!!!!! pos_is_draw wrong!!!!!!!!!!!!!!"); }

    let mut out: bool = false;

    //if test && unsafe {!pos_is_winner_avx512(pos)} { out = true; }
    if found == true && !pos_is_winner_scan(pos) { out = true; }

    out
}

fn pos_is_end(pos : &Pos) -> bool {

    if pos_is_winner_scan(pos) || pos_is_draw(pos) { 
        true 
    } else {
        false
    }
}

fn pos_disp(pos: &Pos) {

    for rk in 0..RANK_SIZE {
        for fl in 0..FILE_SIZE {

            let sq: Square = square_make(fl, rk);

            match pos.state[sq as usize] {
                Color::Black => print!("# "),
                Color::White => print!("O "),
                Color::Empty => print!("- "),
            }
        }

        println!("");    
    }

    match pos.turn() {
        Color::Black => println!("black to play"),
        Color::White => println!("white to play"),
        _ => (),
    }
}

fn gen_moves(list : &mut List, pos: &Pos) {

    list.clear();

    for rk in 0..RANK_SIZE {
        for fl in 0..FILE_SIZE {
            let sq : Square = square_make(fl, rk);
            if pos.can_play(sq) { list.add(sq); }
        }
    }

}

fn search(pos : &Pos, depth: i32, endgame: i32) -> Move {

    //println!("call search");

    let mut new_depth = depth;

    let empties: i32 = pos.count(Color::Empty);
    if (empties <= endgame || new_depth > empties ) { new_depth = empties; }

    if(new_depth == empties) { unsafe { Endgame = true; } }

    search_real(pos, -EVAL_INF, EVAL_INF, new_depth, 0)

}

fn search_real(pos: &Pos, alpha: i32, beta: i32, depth: i32, ply: i32) -> i32 {


    assert!(-EVAL_INF <= alpha && alpha < beta && beta <= EVAL_INF);
    //println!("call search_real");
    //println!("depth = {}", depth);
    //println!("ply   = {}", ply);
    // leaf?

    //if unsafe { pos_is_winner_avx512(&pos) } != pos_is_winner(&pos) { println!("avx512 wrong!!!!!!!!!!!!!!!!!!!!!!!!!!"); }
    if pos_is_winner_scan(&pos) { return -EVAL_INF + ply }
    //if unsafe { pos_is_winner_avx512(&pos) } { return -EVAL_INF + ply }


    if pos_is_draw(&pos) { return 0 }

    if depth == 0 {
         return eval(&pos, ply)
    }

    let p_move_new : [Move; (FILE_SIZE * RANK_SIZE) as usize] = [0; (FILE_SIZE * RANK_SIZE) as usize];

    let mut list = List {
    p_move: p_move_new,
    p_size: 0,
    };

    let mut bm: Move = MOVE_NONE;
    let mut bs: i32  = SCORE_NONE;

    gen_moves(&mut list, &pos);

    // move loop

    if ply == 0 { list.shuffle(); }

    for i in 0..list.size() {

        if bs < beta { 

        let mv: Move = list.p_move[i as usize];

        let mut new_pos = Pos {
            state: pos.state,
            p_turn: pos.p_turn,
            p_last: pos.p_last,

            bitboard: pos.bitboard,
        };

        //println!("p_last = {}", new_pos.p_last);

        new_pos.do_move(mv);

        //println!("After do _move p_last = {}", new_pos.p_last);

        let sc: i32 = -search_real(&new_pos, -beta, -cmp::max(alpha, bs), depth - 1, ply + 1);

        
        //if sc >= 410 || sc <= -410 {
        //println!("sc = {} depth = {}-------------------------------", sc, depth);

        //pos_disp(&new_pos);
        //}
        

        if sc > bs { bm = mv; bs = sc; }

        }
    }

    assert!(bm != MOVE_NONE);
    assert!(bs >= -EVAL_INF && bs <= EVAL_INF);

    if ply == 0 { bm } else { bs } //best move at the root node, best score elsewhere
    //bs
}

fn result(pos: &Pos) -> i32 {

   if(pos_is_winner_scan(pos)) {
       -(FILE_SIZE*RANK_SIZE*100)
   } else {
       0
   }
}


fn eval(pos: &Pos, ply: i32) -> i32 {

    let atk: Side = pos.turn();
    let def: Side = side_opp(atk);

    //let mut sc: i32 = 0;

    let check_live4: Side = def; 
    let check_live4_opp: Side = atk; 

    //if ply % 2 == 1 { check_live4 = def; check_live4_opp = atk; } else { check_live4 = atk; check_live4_opp = def; }
    //if ply % 2 == 0 { check_live4 = def; check_live4_opp = atk; } else { check_live4 = atk; check_live4_opp = def; }
/*
    if unsafe { check_pattern4_once_avx512(&pos, check_live4) } != (check_patternfile4_once(&pos, check_live4) || check_patternrank4_once(&pos, check_live4) || check_patterndial4_once(&pos, check_live4) || check_patterndiar4_once(&pos, check_live4) ) { println!("avx512 wrong!!!!!!!!!!!!!!!!!!!!!!!!!! file4_once !!!!!! self ");  pos_disp(&pos); }
    if unsafe { check_pattern4_once_avx512(&pos, check_live4_opp) } != (check_patternfile4_once(&pos, check_live4_opp) || check_patternrank4_once(&pos, check_live4_opp) || check_patterndial4_once(&pos, check_live4_opp) || check_patterndiar4_once(&pos, check_live4_opp) ) { println!("avx512 wrong!!!!!!!!!!!!!!!!!!!!!!!!!! file4_once !!!!!! opp ");  pos_disp(&pos); }

    #[target_feature(enable = "avx512f")]
    unsafe { 
              let result = check_pattern4_dead_avx512(&pos, check_live4_opp); 

              let mut temp_check: [__mmask16; 5] = [0; 5];

              for i in 0..5 {

              let check_mask0 = _kor_mask16(result[i][0][0], result[i][0][1]);
              let check_mask1 = _kor_mask16(result[i][0][2], result[i][0][3]);
              let check_mask2 = _kor_mask16(result[i][0][4], result[i][1][0]);
              let check_mask3 = _kor_mask16(result[i][1][1], result[i][1][2]);
              let check_mask4 = _kor_mask16(result[i][1][3], result[i][1][4]);
              let check_mask5 = _kor_mask16(result[i][2][0], result[i][2][1]);
              let check_mask6 = _kor_mask16(result[i][2][2], result[i][2][3]);
              let check_mask7 = _kor_mask16(result[i][2][4], result[i][3][0]);
              let check_mask8 = _kor_mask16(result[i][3][1], result[i][3][2]);
              let check_mask9 = _kor_mask16(result[i][3][3], result[i][3][4]);

              let check_mask10 = _kor_mask16(check_mask0, check_mask1);
              let check_mask11 = _kor_mask16(check_mask2, check_mask3);
              let check_mask12 = _kor_mask16(check_mask4, check_mask5);
              let check_mask13 = _kor_mask16(check_mask6, check_mask7);
              let check_mask14 = _kor_mask16(check_mask8, check_mask9);

              let check_mask16 = _kor_mask16(check_mask10, check_mask11);
              let check_mask17 = _kor_mask16(check_mask12, check_mask13);
              let check_mask18 = _kor_mask16(check_mask16, check_mask17);
              temp_check[i] = _kor_mask16(check_mask18, check_mask14);

              }

              let check_mask0 = _kor_mask16(temp_check[0], temp_check[1]);
              let check_mask1 = _kor_mask16(temp_check[2], temp_check[3]);
              let check_mask2 = _kor_mask16(check_mask0, check_mask1);
              let check_mask3 = _kor_mask16(check_mask2, temp_check[4]);

              let test1: bool = check_patternfile4_dead(&pos, check_live4_opp) || check_patternrank4_dead(&pos, check_live4_opp) || check_patterndial4_dead(&pos, check_live4_opp) || check_patterndiar4_dead(&pos, check_live4_opp);

              let mut test2: bool = true;
              
              if check_mask3 > 0 { test2 = true; } else { test2 = false; }

              if test1 != test2 { println!("avx512 wrong!!!!!!!!!!!!!!!!!!!!!!!!!! file4_dead !!!!!! opp ");  pos_disp(&pos); }
    } 

    #[target_feature(enable = "avx512f")]
    unsafe { 
              let result = check_pattern4_dead_avx512(&pos, check_live4); 

              let mut count: i32 = 0;

              for i in 0..5 {
                  for j in 0..4 {
                      for k in 0..5 {
                          count += _popcnt32(result[i][j][k] as i32);
                      }
                  }
              }

              let c4f: i32  = check_patternfile4_dead_n(&pos, check_live4);
              let c4r: i32  = check_patternrank4_dead_n(&pos, check_live4);
              let c4dl: i32 = check_patterndial4_dead_n(&pos, check_live4);
              let c4dr: i32 = check_patterndiar4_dead_n(&pos, check_live4);


              if (c4f+c4r+c4dl+c4dr) != count { println!("avx512 wrong!!!!!!!!!!!!!!!!!!!!!!!!!! file4_dead_count !!!!!! opp org = {}, new = {}", c4f+c4r+c4dl+c4dr, count);  pos_disp(&pos); }
    }

    #[target_feature(enable = "avx512f")]
    unsafe { 
              let result = check_pattern3_live_avx512(&pos, check_live4); 

              let mut count: i32 = 0;

              for i in 0..3 {
                  for j in 0..4 {
                     for k in 0..5 {
                        count += _popcnt32(result[i][j][k] as i32);
                     }
                  }
              }

              let c3f: i32  = check_patternfile3_live_n(&pos, check_live4);
              let c3r: i32  = check_patternrank3_live_n(&pos, check_live4);
              let c3dl: i32 = check_patterndial3_live_n(&pos, check_live4);
              let c3dr: i32 = check_patterndiar3_live_n(&pos, check_live4);

              let mut count1: i32 = 0;

              count1 = c3f+c3r+c3dl+c3dr;

              if count != count1 { println!("avx512 wrong!!!!!!!!!!!!!!!!!!!!!!!!!! live3_dead !!!!!! self org = {}, new = {}", count1, count);  pos_disp(&pos); }
    } 
*/

    if check_patternfile4_once(&pos, check_live4) || 
       check_patternrank4_once(&pos, check_live4) ||
       check_patterndial4_once(&pos, check_live4) ||
       check_patterndiar4_once(&pos, check_live4) { return -4096 }

    //if unsafe { check_pattern4_once_avx512(&pos, check_live4) } { return -4096 }
    
    if check_patternfile4_once(&pos, check_live4_opp) || 
       check_patternrank4_once(&pos, check_live4_opp) ||
       check_patterndial4_once(&pos, check_live4_opp) ||
       check_patterndiar4_once(&pos, check_live4_opp) { return 2560 }

    //if unsafe { check_pattern4_once_avx512(&pos, check_live4_opp) } { return 2560 }

    if check_patternfile4_dead(&pos, check_live4_opp) || 
       check_patternrank4_dead(&pos, check_live4_opp) ||
       check_patterndial4_dead(&pos, check_live4_opp) ||
       check_patterndiar4_dead(&pos, check_live4_opp) { return 2560 }

    /*#[target_feature(enable = "avx512f")]
    unsafe { 
              let result = check_pattern4_dead_avx512(&pos, check_live4_opp); 

              let mut temp_check: [__mmask16; 5] = [0; 5];

              for i in 0..5 {
                  let check_mask0 = _kor_mask16(result[i][0][0], result[i][0][1]);
                  let check_mask1 = _kor_mask16(result[i][0][2], result[i][0][3]);
                  let check_mask2 = _kor_mask16(result[i][0][4], result[i][1][0]);
                  let check_mask3 = _kor_mask16(result[i][1][1], result[i][1][2]);
                  let check_mask4 = _kor_mask16(result[i][1][3], result[i][1][4]);
                  let check_mask5 = _kor_mask16(result[i][2][0], result[i][2][1]);
                  let check_mask6 = _kor_mask16(result[i][2][2], result[i][2][3]);
                  let check_mask7 = _kor_mask16(result[i][2][4], result[i][3][0]);
                  let check_mask8 = _kor_mask16(result[i][3][1], result[i][3][2]);
                  let check_mask9 = _kor_mask16(result[i][3][3], result[i][3][4]);

                  let check_mask10 = _kor_mask16(check_mask0, check_mask1);
                  let check_mask11 = _kor_mask16(check_mask2, check_mask3);
                  let check_mask12 = _kor_mask16(check_mask4, check_mask5);
                  let check_mask13 = _kor_mask16(check_mask6, check_mask7);
                  let check_mask14 = _kor_mask16(check_mask8, check_mask9);

                  let check_mask16 = _kor_mask16(check_mask10, check_mask11);
                  let check_mask17 = _kor_mask16(check_mask12, check_mask13);
                  let check_mask18 = _kor_mask16(check_mask16, check_mask17);
                  temp_check[i] = _kor_mask16(check_mask18, check_mask14);
              }

              let check_mask0 = _kor_mask16(temp_check[0], temp_check[1]);
              let check_mask1 = _kor_mask16(temp_check[2], temp_check[3]);
              let check_mask2 = _kor_mask16(check_mask0, check_mask1);
              let check_mask3 = _kor_mask16(check_mask2, temp_check[4]);

              if check_mask3 > 0 { return 2560 }
    } 
*/
    // 4,3
    let c4f: i32  = check_patternfile4_dead_n(&pos, check_live4);
    let c4r: i32  = check_patternrank4_dead_n(&pos, check_live4);
    let c4dl: i32 = check_patterndial4_dead_n(&pos, check_live4);
    let c4dr: i32 = check_patterndiar4_dead_n(&pos, check_live4);

    let c3f: i32 = check_patternfile3_live_n(&pos, check_live4);
    let c3r: i32 = check_patternrank3_live_n(&pos, check_live4);
    let c3dl: i32 = check_patterndial3_live_n(&pos, check_live4);
    let c3dr: i32 = check_patterndiar3_live_n(&pos, check_live4);

    let n_c4: i32 = c4f + c4r + c4dl + c4dr;

    if n_c4 > 1 { return -2048 }

    if n_c4 == 1 && ( c3f+c3r+c3dl+c3dr > 0 ) { return -3048 }

/*
    #[target_feature(enable = "avx512f")]
    unsafe { 
              let result = check_pattern4_dead_avx512(&pos, check_live4); 

              let mut count4: i32 = 0;

              for i in 0..5 {
                  for j in 0..4 {
                      for k in 0..5 {
                          count4 += _popcnt32(result[i][j][k] as i32);
                      }
                  }
              }

              if count4 > 1 { return -2048 }
              else if count4 == 1 {

              let result = check_pattern3_live_avx512(&pos, check_live4); 

              let mut count3: i32 = 0;

              for i in 0..3 {
                  for j in 0..4 {
                     for k in 0..5 {
                        count3 += _popcnt32(result[i][j][k] as i32);
                     }
                  }
              }

              if count3 > 0 { return -3048 }
              } 
    }
  */  
    //---------------------------------------------------------------------------
    
    let c3f_opp = check_patternfile3_live_n(&pos, check_live4_opp);
    let c3r_opp = check_patternrank3_live_n(&pos, check_live4_opp);
    let c3dl_opp = check_patterndial3_live_n(&pos, check_live4_opp);
    let c3dr_opp = check_patterndiar3_live_n(&pos, check_live4_opp);
    if c3f_opp + c3r_opp + c3dl_opp + c3dr_opp > 1 { return 2560 }

    if c3f + c3r + c3dl + c3dr > 1 { return -2048 }
  /* 
    #[target_feature(enable = "avx512f")]
    unsafe { 
              let result = check_pattern3_live_avx512(&pos, check_live4_opp); 

              let mut count: i32 = 0;

              for i in 0..3 {
                  for j in 0..4 {
                     for k in 0..5 {
                        count += _popcnt32(result[i][j][k] as i32);
                     }
                  }
              }

              let c3f: i32  = check_patternfile3_live_n(&pos, check_live4_opp);
              let c3r: i32  = check_patternrank3_live_n(&pos, check_live4_opp);
              let c3dl: i32 = check_patterndial3_live_n(&pos, check_live4_opp);
              let c3dr: i32 = check_patterndiar3_live_n(&pos, check_live4_opp);

              let mut count1: i32 = 0;

              count1 = c3f+c3r+c3dl+c3dr;

              if count1 > 1 { return -2048 }
    } 
*/
    0 
}


fn check_patternfile4_once(pos: &Pos, sd: Side) -> bool {

    for rk in 0..RANK_SIZE {
        for fl in 0..(FILE_SIZE - 5) {
            let sq : Square = square_make(fl, rk);

            let idx0 = sq + PATTERNFILE4[0];
            let idx1 = sq + PATTERNFILE4[1];
            let idx2 = sq + PATTERNFILE4[2];
            let idx3 = sq + PATTERNFILE4[3];
            let idx4 = sq + PATTERNFILE4[4];
            let idx5 = sq + PATTERNFILE4[5];

            let val0 = pos.state[idx0 as usize];
            let val1 = pos.state[idx1 as usize];
            let val2 = pos.state[idx2 as usize];
            let val3 = pos.state[idx3 as usize];
            let val4 = pos.state[idx4 as usize];
            let val5 = pos.state[idx5 as usize];

            if val0 == Color::Empty && val1 == sd && val2 == sd && val3 == sd && val4 == sd && val5 == Color::Empty { return true }
        }  
    } 

    false 
}

fn check_patternrank4_once(pos: &Pos, sd: Side) -> bool {

    for rk in 0..(RANK_SIZE - 5) {
        for fl in 0..FILE_SIZE {
            let sq : Square = square_make(fl, rk);

            let idx0 = sq + PATTERNRANK4[0];
            let idx1 = sq + PATTERNRANK4[1];
            let idx2 = sq + PATTERNRANK4[2];
            let idx3 = sq + PATTERNRANK4[3];
            let idx4 = sq + PATTERNRANK4[4];
            let idx5 = sq + PATTERNRANK4[5];

            let val0 = pos.state[idx0 as usize];
            let val1 = pos.state[idx1 as usize];
            let val2 = pos.state[idx2 as usize];
            let val3 = pos.state[idx3 as usize];
            let val4 = pos.state[idx4 as usize];
            let val5 = pos.state[idx5 as usize];

            if val0 == Color::Empty && val1 == sd && val2 == sd && val3 == sd && val4 == sd && val5 == Color::Empty { return true }
        }  
    } 

    false 
}

fn check_patterndial4_once(pos: &Pos, sd : Side) -> bool {

    for rk in 0..(RANK_SIZE - 5) {
        for fl in 0..(FILE_SIZE - 5) {
            let sq : Square = square_make(fl, rk);

            let idx0 = sq + PATTERNDIAL4[0];
            let idx1 = sq + PATTERNDIAL4[1];
            let idx2 = sq + PATTERNDIAL4[2];
            let idx3 = sq + PATTERNDIAL4[3];
            let idx4 = sq + PATTERNDIAL4[4];
            let idx5 = sq + PATTERNDIAL4[5];

            let val0 = pos.state[idx0 as usize];
            let val1 = pos.state[idx1 as usize];
            let val2 = pos.state[idx2 as usize];
            let val3 = pos.state[idx3 as usize];
            let val4 = pos.state[idx4 as usize];
            let val5 = pos.state[idx5 as usize];

            if val0 == Color::Empty && val1 == sd && val2 == sd && val3 == sd && val4 == sd && val5 == Color::Empty { return true }
        }  
    } 

    false 
}

fn check_patterndiar4_once(pos: &Pos, sd: Side) -> bool {

    for rk in 0..(RANK_SIZE - 5) {
        for fl in 5..FILE_SIZE {
            let sq : Square = square_make(fl, rk);

            let idx0 = sq + PATTERNDIAR4[0];
            let idx1 = sq + PATTERNDIAR4[1];
            let idx2 = sq + PATTERNDIAR4[2];
            let idx3 = sq + PATTERNDIAR4[3];
            let idx4 = sq + PATTERNDIAR4[4];
            let idx5 = sq + PATTERNDIAR4[5];

            let val0 = pos.state[idx0 as usize];
            let val1 = pos.state[idx1 as usize];
            let val2 = pos.state[idx2 as usize];
            let val3 = pos.state[idx3 as usize];
            let val4 = pos.state[idx4 as usize];
            let val5 = pos.state[idx5 as usize];

            if val0 == Color::Empty && val1 == sd && val2 == sd && val3 == sd && val4 == sd && val5 == Color::Empty { return true }
        }  
    } 

    false 
}

fn check_patternfile4_dead(pos: &Pos, sd: Side) -> bool {

    for rk in 0..RANK_SIZE {
        for fl in 0..(FILE_SIZE - 4) {
            let sq : Square = square_make(fl, rk);

            let idx0 = sq + PATTERNFILE4[0];
            let idx1 = sq + PATTERNFILE4[1];
            let idx2 = sq + PATTERNFILE4[2];
            let idx3 = sq + PATTERNFILE4[3];
            let idx4 = sq + PATTERNFILE4[4];

            let val0 = pos.state[idx0 as usize];
            let val1 = pos.state[idx1 as usize];
            let val2 = pos.state[idx2 as usize];
            let val3 = pos.state[idx3 as usize];
            let val4 = pos.state[idx4 as usize];

            if val0 == sd && val1 == sd && val2 == sd && val3 == sd && val4 == Color::Empty { return true }
            if val0 == sd && val1 == sd && val2 == sd && val3 == Color::Empty && val4 == sd { return true }
            if val0 == sd && val1 == sd && val2 == Color::Empty && val3 == sd && val4 == sd { return true }
            if val0 == sd && val1 == Color::Empty && val2 == sd && val3 == sd && val4 == sd { return true }
            if val0 == Color::Empty && val1 == sd && val2 == sd && val3 == sd && val4 == sd { return true }
        }  
    } 

    false
}

fn check_patternrank4_dead(pos: &Pos, sd: Side) -> bool {

    for rk in 0..(RANK_SIZE - 4) {
        for fl in 0..FILE_SIZE {
            let sq : Square = square_make(fl, rk);

            let idx0 = sq + PATTERNRANK4[0];
            let idx1 = sq + PATTERNRANK4[1];
            let idx2 = sq + PATTERNRANK4[2];
            let idx3 = sq + PATTERNRANK4[3];
            let idx4 = sq + PATTERNRANK4[4];

            let val0 = pos.state[idx0 as usize];
            let val1 = pos.state[idx1 as usize];
            let val2 = pos.state[idx2 as usize];
            let val3 = pos.state[idx3 as usize];
            let val4 = pos.state[idx4 as usize];

            if val0 == sd && val1 == sd && val2 == sd && val3 == sd && val4 == Color::Empty { return true }
            if val0 == sd && val1 == sd && val2 == sd && val3 == Color::Empty && val4 == sd { return true }
            if val0 == sd && val1 == sd && val2 == Color::Empty && val3 == sd && val4 == sd { return true }
            if val0 == sd && val1 == Color::Empty && val2 == sd && val3 == sd && val4 == sd { return true }
            if val0 == Color::Empty && val1 == sd && val2 == sd && val3 == sd && val4 == sd { return true }
        }  
    } 

    false
}

fn check_patterndial4_dead(pos: &Pos, sd: Side) -> bool {

    for rk in 0..(RANK_SIZE - 4) {
        for fl in 0..(FILE_SIZE - 4) {
            let sq : Square = square_make(fl, rk);

            let idx0 = sq + PATTERNDIAL4[0];
            let idx1 = sq + PATTERNDIAL4[1];
            let idx2 = sq + PATTERNDIAL4[2];
            let idx3 = sq + PATTERNDIAL4[3];
            let idx4 = sq + PATTERNDIAL4[4];

            let val0 = pos.state[idx0 as usize];
            let val1 = pos.state[idx1 as usize];
            let val2 = pos.state[idx2 as usize];
            let val3 = pos.state[idx3 as usize];
            let val4 = pos.state[idx4 as usize];

            if val0 == sd && val1 == sd && val2 == sd && val3 == sd && val4 == Color::Empty { return true }
            if val0 == sd && val1 == sd && val2 == sd && val3 == Color::Empty && val4 == sd { return true }
            if val0 == sd && val1 == sd && val2 == Color::Empty && val3 == sd && val4 == sd { return true }
            if val0 == sd && val1 == Color::Empty && val2 == sd && val3 == sd && val4 == sd { return true }
            if val0 == Color::Empty && val1 == sd && val2 == sd && val3 == sd && val4 == sd { return true }
        }  
    } 

    false
}

fn check_patterndiar4_dead(pos: &Pos, sd: Side) -> bool {

    for rk in 0..(RANK_SIZE - 4) {
        for fl in 4..FILE_SIZE {
            let sq : Square = square_make(fl, rk);

            let idx0 = sq + PATTERNDIAR4[0];
            let idx1 = sq + PATTERNDIAR4[1];
            let idx2 = sq + PATTERNDIAR4[2];
            let idx3 = sq + PATTERNDIAR4[3];
            let idx4 = sq + PATTERNDIAR4[4];

            let val0 = pos.state[idx0 as usize];
            let val1 = pos.state[idx1 as usize];
            let val2 = pos.state[idx2 as usize];
            let val3 = pos.state[idx3 as usize];
            let val4 = pos.state[idx4 as usize];

            if val0 == sd && val1 == sd && val2 == sd && val3 == sd && val4 == Color::Empty { return true }
            if val0 == sd && val1 == sd && val2 == sd && val3 == Color::Empty && val4 == sd { return true }
            if val0 == sd && val1 == sd && val2 == Color::Empty && val3 == sd && val4 == sd { return true }
            if val0 == sd && val1 == Color::Empty && val2 == sd && val3 == sd && val4 == sd { return true }
            if val0 == Color::Empty && val1 == sd && val2 == sd && val3 == sd && val4 == sd { return true }
        }  
    } 

    false 
}


fn check_patternfile4_dead_n(pos: &Pos, sd: Side) -> i32 {

    let mut n: i32 = 0;

    for rk in 0..RANK_SIZE {
        for fl in 0..(FILE_SIZE - 4) {
            let sq : Square = square_make(fl, rk);

            let idx0 = sq + PATTERNFILE4[0];
            let idx1 = sq + PATTERNFILE4[1];
            let idx2 = sq + PATTERNFILE4[2];
            let idx3 = sq + PATTERNFILE4[3];
            let idx4 = sq + PATTERNFILE4[4];

            let val0 = pos.state[idx0 as usize];
            let val1 = pos.state[idx1 as usize];
            let val2 = pos.state[idx2 as usize];
            let val3 = pos.state[idx3 as usize];
            let val4 = pos.state[idx4 as usize];

            if val0 == sd && val1 == sd && val2 == sd && val3 == sd && val4 == Color::Empty { n += 1; }
            if val0 == sd && val1 == sd && val2 == sd && val3 == Color::Empty && val4 == sd { n += 1; }
            if val0 == sd && val1 == sd && val2 == Color::Empty && val3 == sd && val4 == sd { n += 1; }
            if val0 == sd && val1 == Color::Empty && val2 == sd && val3 == sd && val4 == sd { n += 1; }
            if val0 == Color::Empty && val1 == sd && val2 == sd && val3 == sd && val4 == sd { n += 1; }
        }  
    } 

    n
}

fn check_patternrank4_dead_n(pos: &Pos, sd: Side) -> i32 {

    let mut n: i32 = 0;

    for rk in 0..(RANK_SIZE - 4) {
        for fl in 0..FILE_SIZE {
            let sq : Square = square_make(fl, rk);

            let idx0 = sq + PATTERNRANK4[0];
            let idx1 = sq + PATTERNRANK4[1];
            let idx2 = sq + PATTERNRANK4[2];
            let idx3 = sq + PATTERNRANK4[3];
            let idx4 = sq + PATTERNRANK4[4];

            let val0 = pos.state[idx0 as usize];
            let val1 = pos.state[idx1 as usize];
            let val2 = pos.state[idx2 as usize];
            let val3 = pos.state[idx3 as usize];
            let val4 = pos.state[idx4 as usize];

            if val0 == sd && val1 == sd && val2 == sd && val3 == sd && val4 == Color::Empty { n += 1; }
            if val0 == sd && val1 == sd && val2 == sd && val3 == Color::Empty && val4 == sd { n += 1; }
            if val0 == sd && val1 == sd && val2 == Color::Empty && val3 == sd && val4 == sd { n += 1; }
            if val0 == sd && val1 == Color::Empty && val2 == sd && val3 == sd && val4 == sd { n += 1; }
            if val0 == Color::Empty && val1 == sd && val2 == sd && val3 == sd && val4 == sd { n += 1; }
        }  
    } 

    n
}

fn check_patterndial4_dead_n(pos: &Pos, sd: Side) -> i32 {

    let mut n: i32 = 0;

    for rk in 0..(RANK_SIZE - 4) {
        for fl in 0..(FILE_SIZE - 4) {
            let sq : Square = square_make(fl, rk);

            let idx0 = sq + PATTERNDIAL4[0];
            let idx1 = sq + PATTERNDIAL4[1];
            let idx2 = sq + PATTERNDIAL4[2];
            let idx3 = sq + PATTERNDIAL4[3];
            let idx4 = sq + PATTERNDIAL4[4];

            let val0 = pos.state[idx0 as usize];
            let val1 = pos.state[idx1 as usize];
            let val2 = pos.state[idx2 as usize];
            let val3 = pos.state[idx3 as usize];
            let val4 = pos.state[idx4 as usize];

            if val0 == sd && val1 == sd && val2 == sd && val3 == sd && val4 == Color::Empty { n += 1; }
            if val0 == sd && val1 == sd && val2 == sd && val3 == Color::Empty && val4 == sd { n += 1; }
            if val0 == sd && val1 == sd && val2 == Color::Empty && val3 == sd && val4 == sd { n += 1; }
            if val0 == sd && val1 == Color::Empty && val2 == sd && val3 == sd && val4 == sd { n += 1; }
            if val0 == Color::Empty && val1 == sd && val2 == sd && val3 == sd && val4 == sd { n += 1; }
        }  
    } 

    n
}

fn check_patterndiar4_dead_n(pos: &Pos, sd: Side) -> i32 {

    let mut n: i32 = 0;

    for rk in 0..(RANK_SIZE - 4) {
        for fl in 4..FILE_SIZE {
            let sq : Square = square_make(fl, rk);

            let idx0 = sq + PATTERNDIAR4[0];
            let idx1 = sq + PATTERNDIAR4[1];
            let idx2 = sq + PATTERNDIAR4[2];
            let idx3 = sq + PATTERNDIAR4[3];
            let idx4 = sq + PATTERNDIAR4[4];

            let val0 = pos.state[idx0 as usize];
            let val1 = pos.state[idx1 as usize];
            let val2 = pos.state[idx2 as usize];
            let val3 = pos.state[idx3 as usize];
            let val4 = pos.state[idx4 as usize];

            if val0 == sd && val1 == sd && val2 == sd && val3 == sd && val4 == Color::Empty { n += 1; }
            if val0 == sd && val1 == sd && val2 == sd && val3 == Color::Empty && val4 == sd { n += 1; }
            if val0 == sd && val1 == sd && val2 == Color::Empty && val3 == sd && val4 == sd { n += 1; }
            if val0 == sd && val1 == Color::Empty && val2 == sd && val3 == sd && val4 == sd { n += 1; }
            if val0 == Color::Empty && val1 == sd && val2 == sd && val3 == sd && val4 == sd { n += 1; }
        }  
    } 

    n
}


/*fn check_patternfile3_live(pos: &Pos, sd: Side) -> bool {

    let last_move: Move = pos.p_last;

    let mut n: i32 = 0;

    for rk in 0..RANK_SIZE {
        for fl in 0..(FILE_SIZE - 4) {
            let sq : Square = square_make(fl, rk);

            let idx0 = sq + PATTERNFILE4[0];
            let idx1 = sq + PATTERNFILE4[1];
            let idx2 = sq + PATTERNFILE4[2];
            let idx3 = sq + PATTERNFILE4[3];
            let idx4 = sq + PATTERNFILE4[4];

            let val0 = pos.state[idx0 as usize];
            let val1 = pos.state[idx1 as usize];
            let val2 = pos.state[idx2 as usize];
            let val3 = pos.state[idx3 as usize];
            let val4 = pos.state[idx4 as usize];

            if val0 == Color::Empty && val1 == sd && val2 == sd && val3 == sd && val4 == Color::Empty { return true }
        }  
    } 

    for rk in 0..RANK_SIZE {
        for fl in 0..(FILE_SIZE - 5) {
            let sq : Square = square_make(fl, rk);

            let idx0 = sq + PATTERNFILE4[0];
            let idx1 = sq + PATTERNFILE4[1];
            let idx2 = sq + PATTERNFILE4[2];
            let idx3 = sq + PATTERNFILE4[3];
            let idx4 = sq + PATTERNFILE4[4];
            let idx5 = sq + PATTERNFILE4[5];

            let val0 = pos.state[idx0 as usize];
            let val1 = pos.state[idx1 as usize];
            let val2 = pos.state[idx2 as usize];
            let val3 = pos.state[idx3 as usize];
            let val4 = pos.state[idx4 as usize];
            let val5 = pos.state[idx5 as usize];

            if val0 == Color::Empty && val1 == sd && val2 == sd && val3 == Color::Empty && val4 == sd && val5 == Color::Empty { return true }
            if val0 == Color::Empty && val1 == sd && val2 == Color::Empty && val3 == sd && val4 == sd && val5 == Color::Empty { return true }
        }  
    } 

    false 
}

fn check_patternrank3_live(pos: &Pos, sd: Side) -> bool {

    let last_move: Move = pos.p_last;

    // let mut n: i32 = 0;

    for rk in 0..(RANK_SIZE - 4) {
        for fl in 0..FILE_SIZE {
            let sq : Square = square_make(fl, rk);

            let idx0 = sq + PATTERNRANK4[0];
            let idx1 = sq + PATTERNRANK4[1];
            let idx2 = sq + PATTERNRANK4[2];
            let idx3 = sq + PATTERNRANK4[3];
            let idx4 = sq + PATTERNRANK4[4];

            let val0 = pos.state[idx0 as usize];
            let val1 = pos.state[idx1 as usize];
            let val2 = pos.state[idx2 as usize];
            let val3 = pos.state[idx3 as usize];
            let val4 = pos.state[idx4 as usize];

            if val0 == Color::Empty && val1 == sd && val2 == sd && val3 == sd && val4 == Color::Empty { return true }
        }  
    } 

    for rk in 0..(RANK_SIZE - 5) {
        for fl in 0..FILE_SIZE {
            let sq : Square = square_make(fl, rk);

            let idx0 = sq + PATTERNRANK4[0];
            let idx1 = sq + PATTERNRANK4[1];
            let idx2 = sq + PATTERNRANK4[2];
            let idx3 = sq + PATTERNRANK4[3];
            let idx4 = sq + PATTERNRANK4[4];
            let idx5 = sq + PATTERNRANK4[5];

            let val0 = pos.state[idx0 as usize];
            let val1 = pos.state[idx1 as usize];
            let val2 = pos.state[idx2 as usize];
            let val3 = pos.state[idx3 as usize];
            let val4 = pos.state[idx4 as usize];
            let val5 = pos.state[idx5 as usize];

            if val0 == Color::Empty && val1 == sd && val2 == sd && val3 == Color::Empty && val4 == sd && val5 == Color::Empty { return true }
            if val0 == Color::Empty && val1 == sd && val2 == Color::Empty && val3 == sd && val4 == sd && val5 == Color::Empty { return true }
        }  
    } 

    false 
}

fn check_patterndial3_live(pos: &Pos, sd: Side) -> bool {

    let last_move: Move = pos.p_last;
 
    //let mut n: i32 = 0;

    for rk in 0..(RANK_SIZE - 4) {
        for fl in 0..(FILE_SIZE - 4) {
            let sq : Square = square_make(fl, rk);

            let idx0 = sq + PATTERNDIAL4[0];
            let idx1 = sq + PATTERNDIAL4[1];
            let idx2 = sq + PATTERNDIAL4[2];
            let idx3 = sq + PATTERNDIAL4[3];
            let idx4 = sq + PATTERNDIAL4[4];

            let val0 = pos.state[idx0 as usize];
            let val1 = pos.state[idx1 as usize];
            let val2 = pos.state[idx2 as usize];
            let val3 = pos.state[idx3 as usize];
            let val4 = pos.state[idx4 as usize];

            if val0 == Color::Empty && val1 == sd && val2 == sd && val3 == sd && val4 == Color::Empty { return true }
        }  
    } 

    for rk in 0..(RANK_SIZE - 5) {
        for fl in 0..(FILE_SIZE - 5) {
            let sq : Square = square_make(fl, rk);

            let idx0 = sq + PATTERNDIAL4[0];
            let idx1 = sq + PATTERNDIAL4[1];
            let idx2 = sq + PATTERNDIAL4[2];
            let idx3 = sq + PATTERNDIAL4[3];
            let idx4 = sq + PATTERNDIAL4[4];
            let idx5 = sq + PATTERNDIAL4[5];

            let val0 = pos.state[idx0 as usize];
            let val1 = pos.state[idx1 as usize];
            let val2 = pos.state[idx2 as usize];
            let val3 = pos.state[idx3 as usize];
            let val4 = pos.state[idx4 as usize];
            let val5 = pos.state[idx5 as usize];

            if val0 == Color::Empty && val1 == sd && val2 == sd && val3 == Color::Empty && val4 == sd && val5 == Color::Empty { return true }
            if val0 == Color::Empty && val1 == sd && val2 == Color::Empty && val3 == sd && val4 == sd && val5 == Color::Empty { return true }
        }  
    } 

    false 
}

fn check_patterndiar3_live(pos: &Pos, sd: Side) -> bool {

    let last_move: Move = pos.p_last;

    //let mut n: i32 = 0;

    for rk in 0..(RANK_SIZE - 4) {
        for fl in 4..FILE_SIZE {
            let sq : Square = square_make(fl, rk);

            let idx0 = sq + PATTERNDIAR4[0];
            let idx1 = sq + PATTERNDIAR4[1];
            let idx2 = sq + PATTERNDIAR4[2];
            let idx3 = sq + PATTERNDIAR4[3];
            let idx4 = sq + PATTERNDIAR4[4];

            let val0 = pos.state[idx0 as usize];
            let val1 = pos.state[idx1 as usize];
            let val2 = pos.state[idx2 as usize];
            let val3 = pos.state[idx3 as usize];
            let val4 = pos.state[idx4 as usize];

            if val0 == Color::Empty && val1 == sd && val2 == sd && val3 == sd && val4 == Color::Empty { return true }
        }  
    } 

    for rk in 0..(RANK_SIZE - 5) {
        for fl in 5..FILE_SIZE {
            let sq : Square = square_make(fl, rk);

            let idx0 = sq + PATTERNDIAR4[0];
            let idx1 = sq + PATTERNDIAR4[1];
            let idx2 = sq + PATTERNDIAR4[2];
            let idx3 = sq + PATTERNDIAR4[3];
            let idx4 = sq + PATTERNDIAR4[4];
            let idx5 = sq + PATTERNDIAR4[5];

            let val0 = pos.state[idx0 as usize];
            let val1 = pos.state[idx1 as usize];
            let val2 = pos.state[idx2 as usize];
            let val3 = pos.state[idx3 as usize];
            let val4 = pos.state[idx4 as usize];
            let val5 = pos.state[idx5 as usize];

            if val0 == Color::Empty && val1 == sd && val2 == sd && val3 == Color::Empty && val4 == sd && val5 == Color::Empty { return true }
            if val0 == Color::Empty && val1 == sd && val2 == Color::Empty && val3 == sd && val4 == sd && val5 == Color::Empty { return true }
        }  
    } 

    false 
}
*/

fn check_patternfile3_live_n(pos: &Pos, sd: Side) -> i32 {

    let last_move: Move = pos.p_last;

    let mut n: i32 = 0;

    for rk in 0..RANK_SIZE {
        for fl in 0..(FILE_SIZE - 4) {
            let sq : Square = square_make(fl, rk);

            let idx0 = sq + PATTERNFILE4[0];
            let idx1 = sq + PATTERNFILE4[1];
            let idx2 = sq + PATTERNFILE4[2];
            let idx3 = sq + PATTERNFILE4[3];
            let idx4 = sq + PATTERNFILE4[4];

            let val0 = pos.state[idx0 as usize];
            let val1 = pos.state[idx1 as usize];
            let val2 = pos.state[idx2 as usize];
            let val3 = pos.state[idx3 as usize];
            let val4 = pos.state[idx4 as usize];

            if val0 == Color::Empty && val1 == sd && val2 == sd && val3 == sd && val4 == Color::Empty { n +=1 ; }
        }  
    } 

    for rk in 0..RANK_SIZE {
        for fl in 0..(FILE_SIZE - 5) {
            let sq : Square = square_make(fl, rk);

            let idx0 = sq + PATTERNFILE4[0];
            let idx1 = sq + PATTERNFILE4[1];
            let idx2 = sq + PATTERNFILE4[2];
            let idx3 = sq + PATTERNFILE4[3];
            let idx4 = sq + PATTERNFILE4[4];
            let idx5 = sq + PATTERNFILE4[5];

            let val0 = pos.state[idx0 as usize];
            let val1 = pos.state[idx1 as usize];
            let val2 = pos.state[idx2 as usize];
            let val3 = pos.state[idx3 as usize];
            let val4 = pos.state[idx4 as usize];
            let val5 = pos.state[idx5 as usize];

            if val0 == Color::Empty && val1 == sd && val2 == sd && val3 == Color::Empty && val4 == sd && val5 == Color::Empty { n += 1; }
            if val0 == Color::Empty && val1 == sd && val2 == Color::Empty && val3 == sd && val4 == sd && val5 == Color::Empty { n += 1; }
        }  
    } 

    n
}

fn check_patternrank3_live_n(pos: &Pos, sd: Side) -> i32 {

    let last_move: Move = pos.p_last;

    let mut n: i32 = 0;

    for rk in 0..(RANK_SIZE - 4) {
        for fl in 0..FILE_SIZE {
            let sq : Square = square_make(fl, rk);

            let idx0 = sq + PATTERNRANK4[0];
            let idx1 = sq + PATTERNRANK4[1];
            let idx2 = sq + PATTERNRANK4[2];
            let idx3 = sq + PATTERNRANK4[3];
            let idx4 = sq + PATTERNRANK4[4];

            let val0 = pos.state[idx0 as usize];
            let val1 = pos.state[idx1 as usize];
            let val2 = pos.state[idx2 as usize];
            let val3 = pos.state[idx3 as usize];
            let val4 = pos.state[idx4 as usize];

            if val0 == Color::Empty && val1 == sd && val2 == sd && val3 == sd && val4 == Color::Empty { n += 1; }
        }  
    } 

    for rk in 0..(RANK_SIZE - 5) {
        for fl in 0..FILE_SIZE {
            let sq : Square = square_make(fl, rk);

            let idx0 = sq + PATTERNRANK4[0];
            let idx1 = sq + PATTERNRANK4[1];
            let idx2 = sq + PATTERNRANK4[2];
            let idx3 = sq + PATTERNRANK4[3];
            let idx4 = sq + PATTERNRANK4[4];
            let idx5 = sq + PATTERNRANK4[5];

            let val0 = pos.state[idx0 as usize];
            let val1 = pos.state[idx1 as usize];
            let val2 = pos.state[idx2 as usize];
            let val3 = pos.state[idx3 as usize];
            let val4 = pos.state[idx4 as usize];
            let val5 = pos.state[idx5 as usize];

            if val0 == Color::Empty && val1 == sd && val2 == sd && val3 == Color::Empty && val4 == sd && val5 == Color::Empty { n += 1; }
            if val0 == Color::Empty && val1 == sd && val2 == Color::Empty && val3 == sd && val4 == sd && val5 == Color::Empty { n += 1; }
        }  
    } 

    n
}

fn check_patterndial3_live_n(pos: &Pos, sd: Side) -> i32 {

    let last_move: Move = pos.p_last;
 
    let mut n: i32 = 0;

    for rk in 0..(RANK_SIZE - 4) {
        for fl in 0..(FILE_SIZE - 4) {
            let sq : Square = square_make(fl, rk);

            let idx0 = sq + PATTERNDIAL4[0];
            let idx1 = sq + PATTERNDIAL4[1];
            let idx2 = sq + PATTERNDIAL4[2];
            let idx3 = sq + PATTERNDIAL4[3];
            let idx4 = sq + PATTERNDIAL4[4];

            let val0 = pos.state[idx0 as usize];
            let val1 = pos.state[idx1 as usize];
            let val2 = pos.state[idx2 as usize];
            let val3 = pos.state[idx3 as usize];
            let val4 = pos.state[idx4 as usize];

            if val0 == Color::Empty && val1 == sd && val2 == sd && val3 == sd && val4 == Color::Empty { n += 1; }
        }  
    } 

    for rk in 0..(RANK_SIZE - 5) {
        for fl in 0..(FILE_SIZE - 5) {
            let sq : Square = square_make(fl, rk);

            let idx0 = sq + PATTERNDIAL4[0];
            let idx1 = sq + PATTERNDIAL4[1];
            let idx2 = sq + PATTERNDIAL4[2];
            let idx3 = sq + PATTERNDIAL4[3];
            let idx4 = sq + PATTERNDIAL4[4];
            let idx5 = sq + PATTERNDIAL4[5];

            let val0 = pos.state[idx0 as usize];
            let val1 = pos.state[idx1 as usize];
            let val2 = pos.state[idx2 as usize];
            let val3 = pos.state[idx3 as usize];
            let val4 = pos.state[idx4 as usize];
            let val5 = pos.state[idx5 as usize];

            if val0 == Color::Empty && val1 == sd && val2 == sd && val3 == Color::Empty && val4 == sd && val5 == Color::Empty { n += 1; }
            if val0 == Color::Empty && val1 == sd && val2 == Color::Empty && val3 == sd && val4 == sd && val5 == Color::Empty { n += 1; }
        }  
    } 

    n
}

fn check_patterndiar3_live_n(pos: &Pos, sd: Side) -> i32 {

    let last_move: Move = pos.p_last;

    let mut n: i32 = 0;

    for rk in 0..(RANK_SIZE - 4) {
        for fl in 4..FILE_SIZE {
            let sq : Square = square_make(fl, rk);

            let idx0 = sq + PATTERNDIAR4[0];
            let idx1 = sq + PATTERNDIAR4[1];
            let idx2 = sq + PATTERNDIAR4[2];
            let idx3 = sq + PATTERNDIAR4[3];
            let idx4 = sq + PATTERNDIAR4[4];

            let val0 = pos.state[idx0 as usize];
            let val1 = pos.state[idx1 as usize];
            let val2 = pos.state[idx2 as usize];
            let val3 = pos.state[idx3 as usize];
            let val4 = pos.state[idx4 as usize];

            if val0 == Color::Empty && val1 == sd && val2 == sd && val3 == sd && val4 == Color::Empty { n += 1; }
        }  
    } 

    for rk in 0..(RANK_SIZE - 5) {
        for fl in 5..FILE_SIZE {
            let sq : Square = square_make(fl, rk);

            let idx0 = sq + PATTERNDIAR4[0];
            let idx1 = sq + PATTERNDIAR4[1];
            let idx2 = sq + PATTERNDIAR4[2];
            let idx3 = sq + PATTERNDIAR4[3];
            let idx4 = sq + PATTERNDIAR4[4];
            let idx5 = sq + PATTERNDIAR4[5];

            let val0 = pos.state[idx0 as usize];
            let val1 = pos.state[idx1 as usize];
            let val2 = pos.state[idx2 as usize];
            let val3 = pos.state[idx3 as usize];
            let val4 = pos.state[idx4 as usize];
            let val5 = pos.state[idx5 as usize];

            if val0 == Color::Empty && val1 == sd && val2 == sd && val3 == Color::Empty && val4 == sd && val5 == Color::Empty { n += 1; }
            if val0 == Color::Empty && val1 == sd && val2 == Color::Empty && val3 == sd && val4 == sd && val5 == Color::Empty { n += 1; }
        }  
    } 

    n
}

#[target_feature(enable = "avx512f")]
unsafe fn pos_is_winner_avx512(pos : &Pos) -> bool {

    let current_side = side_opp(pos.p_turn);

    let answer = _mm512_set1_epi32( (1<<31)|(1<<30)|(1<<29)|(1<<28)|(1<<27) );

    let answer_mask: __mmask16 = 0b01111111_11111111;

    let coloridx = current_side as usize;

    let mut temp_mask: [[__mmask16; 5]; 4] = [[0; 5]; 4];

    for dir in 0..4 {
        let board0 = _mm512_set_epi32(0, pos.bitboard[coloridx][dir][14], pos.bitboard[coloridx][dir][13], pos.bitboard[coloridx][dir][12], pos.bitboard[coloridx][dir][11], pos.bitboard[coloridx][dir][10], pos.bitboard[coloridx][dir][9], pos.bitboard[coloridx][dir][8], pos.bitboard[coloridx][dir][7], pos.bitboard[coloridx][dir][6], pos.bitboard[coloridx][dir][5], pos.bitboard[coloridx][dir][4], pos.bitboard[coloridx][dir][3], pos.bitboard[coloridx][dir][2], pos.bitboard[coloridx][dir][1], pos.bitboard[coloridx][dir][0]);

        let boardf = _mm512_and_epi32(answer, board0);

        temp_mask[dir][0] = _mm512_mask_cmp_epi32_mask(answer_mask, answer, boardf, 0);//no need answer_mask, because and above

        for i in 1..5 {

            let board1 = _mm512_rol_epi32(board0, i);

            let boardf = _mm512_and_epi32(answer, board1);

            temp_mask[dir][i as usize] = _mm512_mask_cmp_epi32_mask(answer_mask, answer, boardf, 0);//no need answer_mask, because and above
        }
    }

    let check_mask0: __mmask16 = _kor_mask16(temp_mask[0][0], temp_mask[0][1]); 
    let check_mask1: __mmask16 = _kor_mask16(temp_mask[0][2], temp_mask[0][3]); 
    let check_mask2: __mmask16 = _kor_mask16(temp_mask[0][4], temp_mask[1][0]); 
    let check_mask3: __mmask16 = _kor_mask16(temp_mask[1][1], temp_mask[1][2]); 
    let check_mask4: __mmask16 = _kor_mask16(temp_mask[1][3], temp_mask[1][4]); 
    let check_mask5: __mmask16 = _kor_mask16(temp_mask[2][0], temp_mask[2][1]); 
    let check_mask6: __mmask16 = _kor_mask16(temp_mask[2][2], temp_mask[2][3]); 
    let check_mask7: __mmask16 = _kor_mask16(temp_mask[2][4], temp_mask[3][0]); 
    let check_mask8: __mmask16 = _kor_mask16(temp_mask[3][1], temp_mask[3][2]); 
    let check_mask9: __mmask16 = _kor_mask16(temp_mask[3][3], temp_mask[3][4]); 

    let check_mask10: __mmask16 = _kor_mask16(check_mask0, check_mask1); 
    let check_mask11: __mmask16 = _kor_mask16(check_mask2, check_mask3); 
    let check_mask12: __mmask16 = _kor_mask16(check_mask4, check_mask5); 
    let check_mask13: __mmask16 = _kor_mask16(check_mask6, check_mask7); 
    let check_mask14: __mmask16 = _kor_mask16(check_mask8, check_mask9); 

    let check_mask16: __mmask16 = _kor_mask16(check_mask10, check_mask11); 
    let check_mask17: __mmask16 = _kor_mask16(check_mask12, check_mask13); 
    let check_mask18: __mmask16 = _kor_mask16(check_mask16, check_mask17); 
    let check_mask19: __mmask16 = _kor_mask16(check_mask18, check_mask14); 

    if check_mask19 > 0 { return true } else { return false }
}

#[target_feature(enable = "avx512f")]
unsafe fn check_pattern4_once_avx512(pos : &Pos, sd: Side) -> bool {

    //let current_side = side_opp(sd);

    let answer_color = _mm512_set1_epi32(         (1<<30)|(1<<29)|(1<<28)|(1<<27) );
    let answer_empty = _mm512_set1_epi32( (1<<31)|                               (1<<26) );
    let answer       = _mm512_set1_epi32( (1<<31)|(1<<30)|(1<<29)|(1<<28)|(1<<27)|(1<<26));

    let answer_mask: __mmask16 = 0b00000001_11111111;

    //let coloridx = current_side as usize;
    let coloridx = sd as usize;
    let emptyidx = Color::Empty as usize;

    let mut temp_mask: [[__mmask16; 4]; 4] = [[0; 4]; 4];

    for dir in 0..4 {

       let board0 = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, pos.bitboard[coloridx][dir][8], pos.bitboard[coloridx][dir][7], pos.bitboard[coloridx][dir][6], pos.bitboard[coloridx][dir][5], pos.bitboard[coloridx][dir][4], pos.bitboard[coloridx][dir][3], pos.bitboard[coloridx][dir][2], pos.bitboard[coloridx][dir][1], pos.bitboard[coloridx][dir][0]);

       let  board1 = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, pos.bitboard[emptyidx][dir][8], pos.bitboard[emptyidx][dir][7], pos.bitboard[emptyidx][dir][6], pos.bitboard[emptyidx][dir][5], pos.bitboard[emptyidx][dir][4], pos.bitboard[emptyidx][dir][3], pos.bitboard[emptyidx][dir][2], pos.bitboard[emptyidx][dir][1], pos.bitboard[emptyidx][dir][0]);

        let boardf1 = _mm512_and_epi32(answer_color, board0);// check sd
        let boardf2 = _mm512_and_epi32(answer_empty, board1);// check empty
        let boardf  = _mm512_or_epi32(boardf1, boardf2);

        temp_mask[dir][0] = _mm512_mask_cmp_epi32_mask(answer_mask, answer, boardf, 0);//no need answer_mask, because and above

        for i in 1..4 { //only move 3 times

            let board2 = _mm512_rol_epi32(board0, i);//rot sd
            let board3 = _mm512_rol_epi32(board1, i);//rot empty

            let boardf1 = _mm512_and_epi32(answer_color, board2);
            let boardf2 = _mm512_and_epi32(answer_empty, board3);
            let boardf  = _mm512_or_epi32(boardf1, boardf2);

            temp_mask[dir][i as usize] = _mm512_mask_cmp_epi32_mask(answer_mask, answer, boardf, 0);//no need answer_mask, because and above
        }
    }

    let check_mask0: __mmask16 = _kor_mask16(temp_mask[0][0], temp_mask[0][1]); 
    let check_mask1: __mmask16 = _kor_mask16(temp_mask[0][2], temp_mask[0][3]); 
    let check_mask2: __mmask16 = _kor_mask16(temp_mask[1][0], temp_mask[1][1]); 
    let check_mask3: __mmask16 = _kor_mask16(temp_mask[1][2], temp_mask[1][3]); 
    let check_mask4: __mmask16 = _kor_mask16(temp_mask[2][0], temp_mask[2][1]); 
    let check_mask5: __mmask16 = _kor_mask16(temp_mask[2][2], temp_mask[2][3]); 
    let check_mask6: __mmask16 = _kor_mask16(temp_mask[3][0], temp_mask[3][1]); 
    let check_mask7: __mmask16 = _kor_mask16(temp_mask[3][2], temp_mask[3][3]); 

    let check_mask10: __mmask16 = _kor_mask16(check_mask0, check_mask1); 
    let check_mask11: __mmask16 = _kor_mask16(check_mask2, check_mask3); 
    let check_mask12: __mmask16 = _kor_mask16(check_mask4, check_mask5); 
    let check_mask13: __mmask16 = _kor_mask16(check_mask6, check_mask7); 

    let check_mask16: __mmask16 = _kor_mask16(check_mask10, check_mask11); 
    let check_mask17: __mmask16 = _kor_mask16(check_mask12, check_mask13); 
    let check_mask19: __mmask16 = _kor_mask16(check_mask16, check_mask17); 

    if check_mask19 > 0 { return true } else { return false }
}

#[target_feature(enable = "avx512f")]
unsafe fn check_pattern4_dead_avx512(pos : &Pos, sd: Side) -> [[[__mmask16; 5]; 4]; 5] {

    //let current_side = side_opp(sd);

    let answer_color: [__m512i; 5] = [_mm512_set1_epi32(         (1<<30)|(1<<29)|(1<<28)|(1<<27) ),
                                      _mm512_set1_epi32( (1<<31)|        (1<<29)|(1<<28)|(1<<27) ),
                                      _mm512_set1_epi32( (1<<31)|(1<<30)        |(1<<28)|(1<<27) ),
                                      _mm512_set1_epi32( (1<<31)|(1<<30)|(1<<29)        |(1<<27) ),
                                      _mm512_set1_epi32( (1<<31)|(1<<30)|(1<<29)|(1<<28)         )];

    let answer_empty: [__m512i; 5]= [_mm512_set1_epi32( (1<<31) ),
                                     _mm512_set1_epi32(         (1<<30) ),
                                     _mm512_set1_epi32(         (1<<29) ),
                                     _mm512_set1_epi32(         (1<<28) ),
                                     _mm512_set1_epi32(         (1<<27) )];

    let answer       = _mm512_set1_epi32( (1<<31)|(1<<30)|(1<<29)|(1<<28)|(1<<27));

    let answer_mask: __mmask16 = 0b00000001_11111111;

    //let coloridx = current_side as usize;
    let coloridx = sd as usize;
    let emptyidx = Color::Empty as usize;

    let mut temp_mask: [[[__mmask16; 5]; 4]; 5] = [[[0; 5]; 4]; 5];

    for pattern in 0..5 {

        for dir in 0..4 {

            let board0 = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, pos.bitboard[coloridx][dir][8], pos.bitboard[coloridx][dir][7], pos.bitboard[coloridx][dir][6], pos.bitboard[coloridx][dir][5], pos.bitboard[coloridx][dir][4], pos.bitboard[coloridx][dir][3], pos.bitboard[coloridx][dir][2], pos.bitboard[coloridx][dir][1], pos.bitboard[coloridx][dir][0]);

           let  board1 = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, pos.bitboard[emptyidx][dir][8], pos.bitboard[emptyidx][dir][7], pos.bitboard[emptyidx][dir][6], pos.bitboard[emptyidx][dir][5], pos.bitboard[emptyidx][dir][4], pos.bitboard[emptyidx][dir][3], pos.bitboard[emptyidx][dir][2], pos.bitboard[emptyidx][dir][1], pos.bitboard[emptyidx][dir][0]);

           let boardf1 = _mm512_and_epi32(answer_color[pattern], board0);// check sd
           let boardf2 = _mm512_and_epi32(answer_empty[pattern], board1);// check empty
           let boardf  = _mm512_or_epi32(boardf1, boardf2);

           temp_mask[pattern][dir][0] = _mm512_mask_cmp_epi32_mask(answer_mask, answer, boardf, 0);//no need answer_mask, because and above

           for i in 1..5 { //only move 4 times

               let board2 = _mm512_rol_epi32(board0, i);//rot sd
               let board3 = _mm512_rol_epi32(board1, i);//rot empty

               let boardf1 = _mm512_and_epi32(answer_color[pattern], board2);
               let boardf2 = _mm512_and_epi32(answer_empty[pattern], board3);
               let boardf  = _mm512_or_epi32(boardf1, boardf2);

               temp_mask[pattern][dir][i as usize] = _mm512_mask_cmp_epi32_mask(answer_mask, answer, boardf, 0);//no need answer_mask, because and above
               }
          }
    }

    temp_mask 
}


#[target_feature(enable = "avx512f")]
unsafe fn check_pattern3_live_avx512(pos : &Pos, sd: Side) -> [[[__mmask16; 5]; 4]; 3] {

    //let current_side = side_opp(sd);

    let answer_color: [__m512i; 3] = [_mm512_set1_epi32(         (1<<30)|(1<<29)|(1<<28)         ),
                                      _mm512_set1_epi32(         (1<<30)|        (1<<28)|(1<<27) ),
                                      _mm512_set1_epi32(         (1<<30)|(1<<29)        |(1<<27) )];

    let answer_empty: [__m512i; 3]= [_mm512_set1_epi32( (1<<31)|                         (1<<27) ),
                                     _mm512_set1_epi32( (1<<31)|         (1<<29)|                (1<<26) ),
                                     _mm512_set1_epi32( (1<<31)|                 (1<<28)|        (1<<26) )];

    //let answer       = _mm512_set1_epi32( (1<<31)|(1<<30)|(1<<29)|(1<<28)|(1<<27));
    let answer: [__m512i; 3]       = [_mm512_set1_epi32( (1<<31)|(1<<30)|(1<<29)|(1<<28)|(1<<27) ),
                                      _mm512_set1_epi32( (1<<31)|(1<<30)|(1<<29)|(1<<28)|(1<<27)|(1<<26) ),
                                      _mm512_set1_epi32( (1<<31)|(1<<30)|(1<<29)|(1<<28)|(1<<27)|(1<<26) )];

    let answer_mask: __mmask16 = 0b00000001_11111111;

    //let coloridx = current_side as usize;
    let coloridx = sd as usize;
    let emptyidx = Color::Empty as usize;

    let mut temp_mask: [[[__mmask16; 5]; 4]; 3] = [[[0; 5]; 4]; 3];

    for pattern in 0..3 {

        for dir in 0..4 {

            let board0 = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, pos.bitboard[coloridx][dir][8], pos.bitboard[coloridx][dir][7], pos.bitboard[coloridx][dir][6], pos.bitboard[coloridx][dir][5], pos.bitboard[coloridx][dir][4], pos.bitboard[coloridx][dir][3], pos.bitboard[coloridx][dir][2], pos.bitboard[coloridx][dir][1], pos.bitboard[coloridx][dir][0]);

           let  board1 = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, pos.bitboard[emptyidx][dir][8], pos.bitboard[emptyidx][dir][7], pos.bitboard[emptyidx][dir][6], pos.bitboard[emptyidx][dir][5], pos.bitboard[emptyidx][dir][4], pos.bitboard[emptyidx][dir][3], pos.bitboard[emptyidx][dir][2], pos.bitboard[emptyidx][dir][1], pos.bitboard[emptyidx][dir][0]);

           let boardf1 = _mm512_and_epi32(answer_color[pattern], board0);// check sd
           let boardf2 = _mm512_and_epi32(answer_empty[pattern], board1);// check empty
           let boardf  = _mm512_or_epi32(boardf1, boardf2);

           temp_mask[pattern][dir][0] = _mm512_mask_cmp_epi32_mask(answer_mask, answer[pattern], boardf, 0);//no need answer_mask, because and above

           for i in 1..5 { //only move 4 times

               let board2 = _mm512_rol_epi32(board0, i);//rot sd
               let board3 = _mm512_rol_epi32(board1, i);//rot empty

               let boardf1 = _mm512_and_epi32(answer_color[pattern], board2);
               let boardf2 = _mm512_and_epi32(answer_empty[pattern], board3);
               let boardf  = _mm512_or_epi32(boardf1, boardf2);

               temp_mask[pattern][dir][i as usize] = _mm512_mask_cmp_epi32_mask(answer_mask, answer[pattern], boardf, 0);//no need answer_mask, because and above
               }
          }
    }

    temp_mask 
}

fn check_patternfile5(pos: &Pos, sd: Side) -> bool {

    for rk in 0..RANK_SIZE {
        for fl in 0..(FILE_SIZE - 4) {
            let sq : Square = square_make(fl, rk);

            let idx0 = sq + PATTERNFILE4[0];
            let idx1 = sq + PATTERNFILE4[1];
            let idx2 = sq + PATTERNFILE4[2];
            let idx3 = sq + PATTERNFILE4[3];
            let idx4 = sq + PATTERNFILE4[4];

            let val0 = pos.state[idx0 as usize];
            let val1 = pos.state[idx1 as usize];
            let val2 = pos.state[idx2 as usize];
            let val3 = pos.state[idx3 as usize];
            let val4 = pos.state[idx4 as usize];

            if val0 == sd && val1 == sd && val2 == sd && val3 == sd && val4 == sd { return true }
        }  
    } 

    false
}

fn check_patternrank5(pos: &Pos, sd: Side) -> bool {

    for rk in 0..(RANK_SIZE - 4) {
        for fl in 0..FILE_SIZE {
            let sq : Square = square_make(fl, rk);

            let idx0 = sq + PATTERNRANK4[0];
            let idx1 = sq + PATTERNRANK4[1];
            let idx2 = sq + PATTERNRANK4[2];
            let idx3 = sq + PATTERNRANK4[3];
            let idx4 = sq + PATTERNRANK4[4];

            let val0 = pos.state[idx0 as usize];
            let val1 = pos.state[idx1 as usize];
            let val2 = pos.state[idx2 as usize];
            let val3 = pos.state[idx3 as usize];
            let val4 = pos.state[idx4 as usize];

            if val0 == sd && val1 == sd && val2 == sd && val3 == sd && val4 == sd { return true }
        }  
    } 

    false
}

fn check_patterndial5(pos: &Pos, sd: Side) -> bool {

    for rk in 0..(RANK_SIZE - 4) {
        for fl in 0..(FILE_SIZE - 4) {
            let sq : Square = square_make(fl, rk);

            let idx0 = sq + PATTERNDIAL4[0];
            let idx1 = sq + PATTERNDIAL4[1];
            let idx2 = sq + PATTERNDIAL4[2];
            let idx3 = sq + PATTERNDIAL4[3];
            let idx4 = sq + PATTERNDIAL4[4];

            let val0 = pos.state[idx0 as usize];
            let val1 = pos.state[idx1 as usize];
            let val2 = pos.state[idx2 as usize];
            let val3 = pos.state[idx3 as usize];
            let val4 = pos.state[idx4 as usize];

            if val0 == sd && val1 == sd && val2 == sd && val3 == sd && val4 == sd { return true }
        }  
    } 

    false
}

fn check_patterndiar5(pos: &Pos, sd: Side) -> bool {

    for rk in 0..(RANK_SIZE - 4) {
        for fl in 4..FILE_SIZE {
            let sq : Square = square_make(fl, rk);

            let idx0 = sq + PATTERNDIAR4[0];
            let idx1 = sq + PATTERNDIAR4[1];
            let idx2 = sq + PATTERNDIAR4[2];
            let idx3 = sq + PATTERNDIAR4[3];
            let idx4 = sq + PATTERNDIAR4[4];

            let val0 = pos.state[idx0 as usize];
            let val1 = pos.state[idx1 as usize];
            let val2 = pos.state[idx2 as usize];
            let val3 = pos.state[idx3 as usize];
            let val4 = pos.state[idx4 as usize];

            if val0 == sd && val1 == sd && val2 == sd && val3 == sd && val4 == sd { return true }
        }  
    } 

    false 
}

fn main() {

    loop
    {

    let start = Instant::now();

    println!("Hello, this is connect 6!");

    //unsafe { test_avx512(); }

    let test_state: [Color; SQUARE_SIZE as usize] = [Color::Empty; SQUARE_SIZE as usize];

    let test_bitboard: [[[i32; FILE_SIZE as usize]; 4]; 3] = [[[0; FILE_SIZE as usize]; 4]; 3];

    let mut test1 = Pos {
        state: test_state,
        p_turn: Color::Black,
        p_last: square_make(5,5),

        bitboard: test_bitboard,
    };

    test1.init();

    //pos_disp(&test1);

    for i in 0..(FILE_SIZE*RANK_SIZE) {

     //   println!("----------------------------------------\n\n\n\n");
      //  println!("MOVE {}!!!!\n\n\n\n", i);


    let mut d = 2;
    let mut e = 4;

    //if i < 6 { d = 1; e = 2; }

    let next_move: Move = search(&test1, d, e);
    //println!("next move is {}", next_move);
    //println!("file is {}",  square_file(next_move));
    //println!("rank is {}",  square_rank(next_move));

    test1.do_move(next_move);

    //pos_disp(&test1);

    if pos_is_end(&test1) { 
        
        println!("Game over!!!!!!");
        println!("MOVE {}!!!!\n", i);
        //pos_disp(&test1);
        
        break; }
    }


    let duration = start.elapsed();

    println!("Time elapsed in expensive_function() is: {:?}", duration);
    }


}
