//! <b>Outer-Open Gomoku</b> is a board game which is a enhanced version of connect5 (Gomoku).\
//! The game is a two-player game which played on a 15x15 Go board.\
//! Two players take turns placing a move on an empty intersection in this board.\
//! The winner is the first player to form an unbroken chain of five moves horizontally, vertically, or diagonally.\
//! Unlike Gomoku, the first move is required to be placed at the two outer rows or columns of this board.\
//! This program provides an AI playing with Minimax search with alpha-beta pruning.\
//! The avx512f intrinsic version can do 16 pattern matching at one time.\
//!
//! On Intel i7-7800x using singe core with fixed AVX-512 clock at 4.0GHz, the avx512f version is speed up about 4.3x.\
//! The average time for each move in the avx512f version is around 31.17s.
//! In the future, avx512bw can do 64 pattern matching at one time. It might speed up more.\
//!
//! //! You can test out this program via:
//!
//!     cargo +nightly run --release --bin connect5
//!
//! and you should see a game self-playing. In the end of the game, it shows the average time for
//! each move.

#![feature(stdsimd, avx512_target_feature)]
#![feature(stmt_expr_attributes)]

use rand::seq::SliceRandom;
use rand::thread_rng;

use std::cmp;
use std::time::Instant;

#[cfg(target_arch = "x86")]
use {core_arch::arch::x86::*, std_detect::is_x86_feature_detected};
#[cfg(target_arch = "x86_64")]
use {core_arch::arch::x86_64::*, std_detect::is_x86_feature_detected};

// types

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Color {
    Black = 0,
    White = 1,
    Empty = 2,
    Border = 3,
}

type Square = i32;
type Move = i32;
type Side = Color;

// constants

const FILE_SIZE: i32 = 15;
const RANK_SIZE: i32 = 15;
const SQUARE_SIZE: i32 = (FILE_SIZE + 1) * (FILE_SIZE + 4) + 16 + 4;

const EVAL_INF: i32 = FILE_SIZE * RANK_SIZE * 100;
const MOVE_NONE: Move = -1;
const SCORE_NONE: i32 = -EVAL_INF - 1;

/// DIRECTION 0: left to right\
/// DIRECTION 1: top to bottom\
/// DIRECTION 2: top left to bottom right\
/// DIRECTION 3: top right to bottom left
#[rustfmt::skip]
const DIRECTION: [[i32; 5]; 4] = [ [1, 2, 3, 4, 5],
                                   [1 * (FILE_SIZE + 1), 2 * (FILE_SIZE + 1), 3 * (FILE_SIZE + 1), 4 * (FILE_SIZE + 1), 5 * (FILE_SIZE + 1)],
                                   [1 * (FILE_SIZE + 2), 2 * (FILE_SIZE + 2), 3 * (FILE_SIZE + 2), 4 * (FILE_SIZE + 2), 5 * (FILE_SIZE + 2)],
                                   [1 * (FILE_SIZE + 0), 2 * (FILE_SIZE + 0), 3 * (FILE_SIZE + 0), 4 * (FILE_SIZE + 0), 5 * (FILE_SIZE + 0)]];

/// A table to encode each location to a value in bit 31-0 in the bitboard for 4 direction
#[rustfmt::skip]
const MAPMOVEVALUE: [[i32; 239]; 4] = [ [// Direction 0 
                                         1<<31, 1<<30, 1<<29, 1<<28, 1<<27, 1<<26, 1<<25, 1<<24, 1<<23, 1<<22, 1<<21, 1<<20, 1<<19, 1<<18, 1<<17, 0,
                                         1<<31, 1<<30, 1<<29, 1<<28, 1<<27, 1<<26, 1<<25, 1<<24, 1<<23, 1<<22, 1<<21, 1<<20, 1<<19, 1<<18, 1<<17, 0,
                                         1<<31, 1<<30, 1<<29, 1<<28, 1<<27, 1<<26, 1<<25, 1<<24, 1<<23, 1<<22, 1<<21, 1<<20, 1<<19, 1<<18, 1<<17, 0,
                                         1<<31, 1<<30, 1<<29, 1<<28, 1<<27, 1<<26, 1<<25, 1<<24, 1<<23, 1<<22, 1<<21, 1<<20, 1<<19, 1<<18, 1<<17, 0,
                                         1<<31, 1<<30, 1<<29, 1<<28, 1<<27, 1<<26, 1<<25, 1<<24, 1<<23, 1<<22, 1<<21, 1<<20, 1<<19, 1<<18, 1<<17, 0,
                                         1<<31, 1<<30, 1<<29, 1<<28, 1<<27, 1<<26, 1<<25, 1<<24, 1<<23, 1<<22, 1<<21, 1<<20, 1<<19, 1<<18, 1<<17, 0,
                                         1<<31, 1<<30, 1<<29, 1<<28, 1<<27, 1<<26, 1<<25, 1<<24, 1<<23, 1<<22, 1<<21, 1<<20, 1<<19, 1<<18, 1<<17, 0,
                                         1<<31, 1<<30, 1<<29, 1<<28, 1<<27, 1<<26, 1<<25, 1<<24, 1<<23, 1<<22, 1<<21, 1<<20, 1<<19, 1<<18, 1<<17, 0,
                                         1<<31, 1<<30, 1<<29, 1<<28, 1<<27, 1<<26, 1<<25, 1<<24, 1<<23, 1<<22, 1<<21, 1<<20, 1<<19, 1<<18, 1<<17, 0,
                                         1<<31, 1<<30, 1<<29, 1<<28, 1<<27, 1<<26, 1<<25, 1<<24, 1<<23, 1<<22, 1<<21, 1<<20, 1<<19, 1<<18, 1<<17, 0,
                                         1<<31, 1<<30, 1<<29, 1<<28, 1<<27, 1<<26, 1<<25, 1<<24, 1<<23, 1<<22, 1<<21, 1<<20, 1<<19, 1<<18, 1<<17, 0,
                                         1<<31, 1<<30, 1<<29, 1<<28, 1<<27, 1<<26, 1<<25, 1<<24, 1<<23, 1<<22, 1<<21, 1<<20, 1<<19, 1<<18, 1<<17, 0,
                                         1<<31, 1<<30, 1<<29, 1<<28, 1<<27, 1<<26, 1<<25, 1<<24, 1<<23, 1<<22, 1<<21, 1<<20, 1<<19, 1<<18, 1<<17, 0,
                                         1<<31, 1<<30, 1<<29, 1<<28, 1<<27, 1<<26, 1<<25, 1<<24, 1<<23, 1<<22, 1<<21, 1<<20, 1<<19, 1<<18, 1<<17, 0,
                                         1<<31, 1<<30, 1<<29, 1<<28, 1<<27, 1<<26, 1<<25, 1<<24, 1<<23, 1<<22, 1<<21, 1<<20, 1<<19, 1<<18, 1<<17],
                                        [// Direction 1
                                         1<<31, 1<<31, 1<<31, 1<<31, 1<<31, 1<<31, 1<<31, 1<<31, 1<<31, 1<<31, 1<<31, 1<<31, 1<<31, 1<<31, 1<<31, 0,
                                         1<<30, 1<<30, 1<<30, 1<<30, 1<<30, 1<<30, 1<<30, 1<<30, 1<<30, 1<<30, 1<<30, 1<<30, 1<<30, 1<<30, 1<<30, 0,
                                         1<<29, 1<<29, 1<<29, 1<<29, 1<<29, 1<<29, 1<<29, 1<<29, 1<<29, 1<<29, 1<<29, 1<<29, 1<<29, 1<<29, 1<<29, 0,
                                         1<<28, 1<<28, 1<<28, 1<<28, 1<<28, 1<<28, 1<<28, 1<<28, 1<<28, 1<<28, 1<<28, 1<<28, 1<<28, 1<<28, 1<<28, 0,
                                         1<<27, 1<<27, 1<<27, 1<<27, 1<<27, 1<<27, 1<<27, 1<<27, 1<<27, 1<<27, 1<<27, 1<<27, 1<<27, 1<<27, 1<<27, 0,
                                         1<<26, 1<<26, 1<<26, 1<<26, 1<<26, 1<<26, 1<<26, 1<<26, 1<<26, 1<<26, 1<<26, 1<<26, 1<<26, 1<<26, 1<<26, 0,
                                         1<<25, 1<<25, 1<<25, 1<<25, 1<<25, 1<<25, 1<<25, 1<<25, 1<<25, 1<<25, 1<<25, 1<<25, 1<<25, 1<<25, 1<<25, 0,
                                         1<<24, 1<<24, 1<<24, 1<<24, 1<<24, 1<<24, 1<<24, 1<<24, 1<<24, 1<<24, 1<<24, 1<<24, 1<<24, 1<<24, 1<<24, 0,
                                         1<<23, 1<<23, 1<<23, 1<<23, 1<<23, 1<<23, 1<<23, 1<<23, 1<<23, 1<<23, 1<<23, 1<<23, 1<<23, 1<<23, 1<<23, 0,
                                         1<<22, 1<<22, 1<<22, 1<<22, 1<<22, 1<<22, 1<<22, 1<<22, 1<<22, 1<<22, 1<<22, 1<<22, 1<<22, 1<<22, 1<<22, 0,
                                         1<<21, 1<<21, 1<<21, 1<<21, 1<<21, 1<<21, 1<<21, 1<<21, 1<<21, 1<<21, 1<<21, 1<<21, 1<<21, 1<<21, 1<<21, 0,
                                         1<<20, 1<<20, 1<<20, 1<<20, 1<<20, 1<<20, 1<<20, 1<<20, 1<<20, 1<<20, 1<<20, 1<<20, 1<<20, 1<<20, 1<<20, 0,
                                         1<<19, 1<<19, 1<<19, 1<<19, 1<<19, 1<<19, 1<<19, 1<<19, 1<<19, 1<<19, 1<<19, 1<<19, 1<<19, 1<<19, 1<<19, 0,
                                         1<<18, 1<<18, 1<<18, 1<<18, 1<<18, 1<<18, 1<<18, 1<<18, 1<<18, 1<<18, 1<<18, 1<<18, 1<<18, 1<<18, 1<<18, 0,
                                         1<<17, 1<<17, 1<<17, 1<<17, 1<<17, 1<<17, 1<<17, 1<<17, 1<<17, 1<<17, 1<<17, 1<<17, 1<<17, 1<<17, 1<<17],
                                        [// Direction 2 
                                         1<<15, 1<<15, 1<<15, 1<<15, 1<<15, 1<<15, 1<<15, 1<<15, 1<<15, 1<<15, 1<<15, 0,     0,     0,     0,     0,
                                         1<<15, 1<<14, 1<<14, 1<<14, 1<<14, 1<<14, 1<<14, 1<<14, 1<<14, 1<<14, 1<<14, 1<<14, 0,     0,     0,     0,
                                         1<<15, 1<<14, 1<<13, 1<<13, 1<<13, 1<<13, 1<<13, 1<<13, 1<<13, 1<<13, 1<<13, 1<<13, 1<<13, 0,     0,     0,
                                         1<<15, 1<<14, 1<<13, 1<<12, 1<<12, 1<<12, 1<<12, 1<<12, 1<<12, 1<<12, 1<<12, 1<<12, 1<<12, 1<<12, 0,     0,
                                         1<<15, 1<<14, 1<<13, 1<<12, 1<<11, 1<<11, 1<<11, 1<<11, 1<<11, 1<<11, 1<<11, 1<<11, 1<<11, 1<<11, 1<<11, 0,
                                         1<<15, 1<<14, 1<<13, 1<<12, 1<<11, 1<<10, 1<<10, 1<<10, 1<<10, 1<<10, 1<<10, 1<<10, 1<<10, 1<<10, 1<<10, 0,
                                         1<<9,  1<<14, 1<<13, 1<<12, 1<<11, 1<<10, 1<<9,  1<<9,  1<<9,  1<<9,  1<<9,  1<<9,  1<<9,  1<<9,  1<<9,  0,
                                         1<<8,  1<<8,  1<<13, 1<<12, 1<<11, 1<<10, 1<<9,  1<<8,  1<<8,  1<<8,  1<<8,  1<<8,  1<<8,  1<<8,  1<<8,  0,
                                         1<<7,  1<<7,  1<<7,  1<<12, 1<<11, 1<<10, 1<<9,  1<<8,  1<<7,  1<<7,  1<<7,  1<<7,  1<<7,  1<<7,  1<<7,  0,
                                         1<<6,  1<<6,  1<<6,  1<<6,  1<<11, 1<<10, 1<<9,  1<<8,  1<<7,  1<<6,  1<<6,  1<<6,  1<<6,  1<<6,  1<<6,  0,
                                         1<<5,  1<<5,  1<<5,  1<<5,  1<<5,  1<<10, 1<<9,  1<<8,  1<<7,  1<<6,  1<<5,  1<<5,  1<<5,  1<<5,  1<<5,  0,
                                         0,     1<<4,  1<<4,  1<<4,  1<<4,  1<<4,  1<<9,  1<<8,  1<<7,  1<<6,  1<<5,  1<<4,  1<<4,  1<<4,  1<<4,  0,
                                         0,     0,     1<<3,  1<<3,  1<<3,  1<<3,  1<<3,  1<<8,  1<<7,  1<<6,  1<<5,  1<<4,  1<<3,  1<<3,  1<<3,  0,
                                         0,     0,     0,     1<<2,  1<<2,  1<<2,  1<<2,  1<<2,  1<<7,  1<<6,  1<<5,  1<<4,  1<<3,  1<<2,  1<<2,  0,
                                         0,     0,     0,     0,     1<<1,  1<<1,  1<<1,  1<<1,  1<<1,  1<<6,  1<<5,  1<<4,  1<<3,  1<<2,  1<<1],
                                        [// Direction 3
                                         0,     0,     0,     0,     1<<15, 1<<15, 1<<15, 1<<15, 1<<15, 1<<15, 1<<15, 1<<15, 1<<15, 1<<15, 1<<15, 0,
                                         0,     0,     0,     1<<14, 1<<14, 1<<14, 1<<14, 1<<14, 1<<14, 1<<14, 1<<14, 1<<14, 1<<14, 1<<14, 1<<15, 0,
                                         0,     0,     1<<13, 1<<13, 1<<13, 1<<13, 1<<13, 1<<13, 1<<13, 1<<13, 1<<13, 1<<13, 1<<13, 1<<14, 1<<15, 0,
                                         0,     1<<12, 1<<12, 1<<12, 1<<12, 1<<12, 1<<12, 1<<12, 1<<12, 1<<12, 1<<12, 1<<12, 1<<13, 1<<14, 1<<15, 0,
                                         1<<11, 1<<11, 1<<11, 1<<11, 1<<11, 1<<11, 1<<11, 1<<11, 1<<11, 1<<11, 1<<11, 1<<12, 1<<13, 1<<14, 1<<15, 0,
                                         1<<10, 1<<10, 1<<10, 1<<10, 1<<10, 1<<10, 1<<10, 1<<10, 1<<10, 1<<10, 1<<11, 1<<12, 1<<13, 1<<14, 1<<15, 0,
                                         1<<9,  1<<9,  1<<9,  1<<9,  1<<9,  1<<9,  1<<9,  1<<9,  1<<9,  1<<10, 1<<11, 1<<12, 1<<13, 1<<14, 1<<9,  0,
                                         1<<8,  1<<8,  1<<8,  1<<8,  1<<8,  1<<8,  1<<8,  1<<8,  1<<9,  1<<10, 1<<11, 1<<12, 1<<13, 1<<8,  1<<8,  0,
                                         1<<7,  1<<7,  1<<7,  1<<7,  1<<7,  1<<7,  1<<7,  1<<8,  1<<9,  1<<10, 1<<11, 1<<12, 1<<7,  1<<7,  1<<7,  0,
                                         1<<6,  1<<6,  1<<6,  1<<6,  1<<6,  1<<6,  1<<7,  1<<8,  1<<9,  1<<10, 1<<11, 1<<6,  1<<6,  1<<6,  1<<6,  0,
                                         1<<5,  1<<5,  1<<5,  1<<5,  1<<5,  1<<6,  1<<7,  1<<8,  1<<9,  1<<10, 1<<5,  1<<5,  1<<5,  1<<5,  1<<5,  0,
                                         1<<4,  1<<4,  1<<4,  1<<4,  1<<5,  1<<6,  1<<7,  1<<8,  1<<9,  1<<4,  1<<4,  1<<4,  1<<4,  1<<4,  0,     0,
                                         1<<3,  1<<3,  1<<3,  1<<4,  1<<5,  1<<6,  1<<7,  1<<8,  1<<3,  1<<3,  1<<3,  1<<3,  1<<3,  0,     0,     0,
                                         1<<2,  1<<2,  1<<3,  1<<4,  1<<5,  1<<6,  1<<7,  1<<2,  1<<2,  1<<2,  1<<2,  1<<2,  0,     0,     0,     0,
                                         1<<1,  1<<2,  1<<3,  1<<4,  1<<5,  1<<6,  1<<1,  1<<1,  1<<1,  1<<1,  1<<1,  0,     0,     0,     0]
                                        ];

/// A table to encode each location to an index in the bitboard for 4 direction 
#[rustfmt::skip]
const MAPMOVEIDX: [[i32; 239]; 4] = [ [// Direction 0 
                                       0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                                       1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0,
                                       2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  0,
                                       3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  0,
                                       4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  0,
                                       5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  0,
                                       6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  0,
                                       7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  0,
                                       8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  0,
                                       9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  0,
                                       10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 0,
                                       11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 0,
                                       12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 0,
                                       13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 0,
                                       14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14],
                                      [// Direction 1 
                                       0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0,
                                       0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0,
                                       0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0,
                                       0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0,
                                       0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0,
                                       0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0,
                                       0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0,
                                       0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0,
                                       0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0,
                                       0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0,
                                       0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0,
                                       0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0,
                                       0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0,
                                       0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0,
                                       0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
                                      [// Direction 2
                                       10, 9,  8,   7,  6,  5,  4,  3,  2,  1,  0,  0,  0,  0,  0,  0,
                                       11, 10, 9,   8,  7,  6,  5,  4,  3,  2,  1,  0,  0,  0,  0,  0,
                                       12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1,  0,  0,  0,  0,
                                       13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1,  0,  0,  0,
                                       14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1,  0,  0,
                                       15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1,  0,
                                        1, 15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  0,
                                        2,  1, 15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  0,
                                        3,  2,  1, 15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  0,
                                        4,  3,  2,  1, 15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  0,
                                        5,  4,  3,  2,  1, 15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  0,
                                        0,  5,  4,  3,  2,  1, 15, 14, 13, 12, 11, 10,  9,  8,  7,  0,
                                        0,  0,  5,  4,  3,  2,  1, 15, 14, 13, 12, 11, 10,  9,  8,  0,
                                        0,  0,  0,  5,  4,  3,  2,  1, 15, 14, 13, 12, 11, 10,  9,  0,
                                        0,  0,  0,  0,  5,  4,  3,  2,  1, 15, 14, 13, 12, 11, 10],
                                      [// Direction 3
                                       0,  0,  0,  0,   0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10,  0,
                                       0,  0,  0,  0,   1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11,  0,
                                       0,  0,  0,  1,   2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12,  0,
                                       0,  0,  1,  2,   3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13,  0,
                                       0,  1,  2,  3,   4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,  0,
                                       1,  2,  3,  4,   5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,  0,
                                       2,  3,  4,  5,   6,  7,  8,  9, 10, 11, 12, 13, 14, 15,  1,  0,
                                       3,  4,  5,  6,   7,  8,  9, 10, 11, 12, 13, 14, 15,  1,  2,  0,
                                       4,  5,  6,  7,   8,  9, 10, 11, 12, 13, 14, 15,  1,  2,  3,  0,
                                       5,  6,  7,  8,   9, 10, 11, 12, 13, 14, 15,  1,  2,  3,  4,  0,
                                       6,  7,  8,  9,  10, 11, 12, 13, 14, 15,  1,  2,  3,  4,  5,  0,
                                       7,  8,  9,  10, 11, 12, 13, 14, 15,  1,  2,  3,  4,  5,  0,  0,
                                       8,  9,  10, 11, 12, 13, 14, 15,  1,  2,  3,  4,  5,  0,  0,  0,
                                       9,  10, 11, 12, 13, 14, 15,  1,  2,  3,  4,  5,  0,  0,  0,  0,
                                       10, 11, 12, 13, 14, 15,  1,  2,  3,  4,  5,  0,  0,  0,  0]
                                ];

// structures

/// Use one-dimensional array to store the board state. The location 0 is top left.\
/// 0   1   2   3   4   5   6   7   8   9   10  11  12  13  14  <b>15</b>\
/// 16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  <b>31</b>\
/// ... \
/// position 15, 31, ... are Borders.\
/// position 0 is file 0, rank 0.\
/// position 17 is file 1, rank 1.\
///
/// Use a three-dimensional array to store the bitboard.\
/// The first dimension is color: Black, White and Empty.\
/// The second and third one are 2 x 512-bit. Direction 0 and 2 use the first 512-bit. Direction 1 and
/// 3 use the second 512-bit.\
/// Each 512-bit is a 32-bit x 16 array. Direction 0 and 1 store at bit 31-16 and Direction 2 and 3 store at bit 15-0.  

pub struct Pos {
    // position
    state: [Color; SQUARE_SIZE as usize],
    p_turn: Side,
    bitboard: [[[i32; 16]; 2]; 3],
}

impl Pos {
    pub fn init(&mut self) {
        // starting position
        // Set up the Border
        for i in 0..SQUARE_SIZE as usize {
            self.state[i] = Color::Border;
        }

        // In the beginning, all is Empty
        for rk in 0..RANK_SIZE {
            for fl in 0..FILE_SIZE {
                let sq: Square = square_make(fl, rk);
                self.state[sq as usize] = Color::Empty;
            }
        }

        // first move is Black
        self.p_turn = Color::Black;

        let black = Color::Black as usize;
        let white = Color::White as usize;
        let empty = Color::Empty as usize;

        // set up the corresponding bitboard
        for i in 0..2 {
            for j in 0..16 {
                self.bitboard[black][i][j] = 0;
                self.bitboard[white][i][j] = 0;
                self.bitboard[empty][i][j] = 0;
            }
        }

        for i in 0..2 {
            // use bit 31-16 to store direction 0 and 1
            #[rustfmt::skip]
            for j in 0..FILE_SIZE as usize {
                self.bitboard[empty][i][j] = (1<<31)|(1<<30)|(1<<29)|(1<<28)|(1<<27)|(1<<26)|(1<<25)|(1<<24)|(1<<23)|(1<<22)|(1<<21)|(1<<20)|(1<<19)|(1<<18)|(1<<17);
            }
        }

        // use bit 15-0 to store direction 2 and 3. There are 21 for each one. We combine row1 and row16, row2 and row17, row3 and row18, row4 and row19, and row 5 and row20
        #[rustfmt::skip]
        for i in 0..2 {
            self.bitboard[empty][i][0]  |= (1<<15)|(1<<14)|(1<<13)|(1<<12)|(1<<11); //row 0
            self.bitboard[empty][i][1]  |= (1<<15)|(1<<14)|(1<<13)|(1<<12)|(1<<11)|(1<<10)/*row1*/|(1<<9)|(1<<8)|(1<<7)|(1<<6)|(1<<5)|(1<<4)|(1<<3)|(1<<2)|(1<<1);//row16
            self.bitboard[empty][i][2]  |= (1<<15)|(1<<14)|(1<<13)|(1<<12)|(1<<11)|(1<<10)|(1<<9)/*row2*/|(1<<8)|(1<<7)|(1<<6)|(1<<5)|(1<<4)|(1<<3)|(1<<2)|(1<<1);//row17
            self.bitboard[empty][i][3]  |= (1<<15)|(1<<14)|(1<<13)|(1<<12)|(1<<11)|(1<<10)|(1<<9)|(1<<8)/*row3*/|(1<<7)|(1<<6)|(1<<5)|(1<<4)|(1<<3)|(1<<2)|(1<<1);//row18
            self.bitboard[empty][i][4]  |= (1<<15)|(1<<14)|(1<<13)|(1<<12)|(1<<11)|(1<<10)|(1<<9)|(1<<8)|(1<<7)/*row4*/|(1<<6)|(1<<5)|(1<<4)|(1<<3)|(1<<2)|(1<<1);//row19
            self.bitboard[empty][i][5]  |= (1<<15)|(1<<14)|(1<<13)|(1<<12)|(1<<11)|(1<<10)|(1<<9)|(1<<8)|(1<<7)|(1<<6)/*row5*/|(1<<5)|(1<<4)|(1<<3)|(1<<2)|(1<<1);//row20
            self.bitboard[empty][i][6]  |= (1<<15)|(1<<14)|(1<<13)|(1<<12)|(1<<11)|(1<<10)|(1<<9)|(1<<8)|(1<<7)|(1<<6)|(1<<5);//row6
            self.bitboard[empty][i][7]  |= (1<<15)|(1<<14)|(1<<13)|(1<<12)|(1<<11)|(1<<10)|(1<<9)|(1<<8)|(1<<7)|(1<<6)|(1<<5)|(1<<4);//row7
            self.bitboard[empty][i][8]  |= (1<<15)|(1<<14)|(1<<13)|(1<<12)|(1<<11)|(1<<10)|(1<<9)|(1<<8)|(1<<7)|(1<<6)|(1<<5)|(1<<4)|(1<<3);//row8
            self.bitboard[empty][i][9]  |= (1<<15)|(1<<14)|(1<<13)|(1<<12)|(1<<11)|(1<<10)|(1<<9)|(1<<8)|(1<<7)|(1<<6)|(1<<5)|(1<<4)|(1<<3)|(1<<2);//row9
            self.bitboard[empty][i][10] |= (1<<15)|(1<<14)|(1<<13)|(1<<12)|(1<<11)|(1<<10)|(1<<9)|(1<<8)|(1<<7)|(1<<6)|(1<<5)|(1<<4)|(1<<3)|(1<<2)|(1<<1);//row10
            self.bitboard[empty][i][11] |= (1<<15)|(1<<14)|(1<<13)|(1<<12)|(1<<11)|(1<<10)|(1<<9)|(1<<8)|(1<<7)|(1<<6)|(1<<5)|(1<<4)|(1<<3)|(1<<2);//row11
            self.bitboard[empty][i][12] |= (1<<15)|(1<<14)|(1<<13)|(1<<12)|(1<<11)|(1<<10)|(1<<9)|(1<<8)|(1<<7)|(1<<6)|(1<<5)|(1<<4)|(1<<3);//row12
            self.bitboard[empty][i][13] |= (1<<15)|(1<<14)|(1<<13)|(1<<12)|(1<<11)|(1<<10)|(1<<9)|(1<<8)|(1<<7)|(1<<6)|(1<<5)|(1<<4);//row13
            self.bitboard[empty][i][14] |= (1<<15)|(1<<14)|(1<<13)|(1<<12)|(1<<11)|(1<<10)|(1<<9)|(1<<8)|(1<<7)|(1<<6)|(1<<5);//row14
            self.bitboard[empty][i][15] |= (1<<15)|(1<<14)|(1<<13)|(1<<12)|(1<<11)|(1<<10)|(1<<9)|(1<<8)|(1<<7)|(1<<6);//row15
        }
    }

    pub fn do_move(&mut self, mv: Move) {
        let atk: Side = self.p_turn;
        let def: Side = side_opp(atk);

        let mv = mv as usize;
        let black = Color::Black as usize;
        let white = Color::White as usize;
        let empty = Color::Empty as usize;

        match self.p_turn {
            Color::Black => {
                self.state[mv as usize] = Color::Black;
                // update black move and remove empty move in bitboard
                self.bitboard[black][0][MAPMOVEIDX[0][mv] as usize] |= MAPMOVEVALUE[0][mv];
                self.bitboard[empty][0][MAPMOVEIDX[0][mv] as usize] ^= MAPMOVEVALUE[0][mv];
                self.bitboard[black][1][MAPMOVEIDX[1][mv] as usize] |= MAPMOVEVALUE[1][mv];
                self.bitboard[empty][1][MAPMOVEIDX[1][mv] as usize] ^= MAPMOVEVALUE[1][mv];
                self.bitboard[black][0][MAPMOVEIDX[2][mv] as usize] |= MAPMOVEVALUE[2][mv];
                self.bitboard[empty][0][MAPMOVEIDX[2][mv] as usize] ^= MAPMOVEVALUE[2][mv];
                self.bitboard[black][1][MAPMOVEIDX[3][mv] as usize] |= MAPMOVEVALUE[3][mv];
                self.bitboard[empty][1][MAPMOVEIDX[3][mv] as usize] ^= MAPMOVEVALUE[3][mv];
            }
            Color::White => {
                self.state[mv as usize] = Color::White;
                // update white move and remove empty move in bitboard
                self.bitboard[white][0][MAPMOVEIDX[0][mv] as usize] |= MAPMOVEVALUE[0][mv];
                self.bitboard[empty][0][MAPMOVEIDX[0][mv] as usize] ^= MAPMOVEVALUE[0][mv];
                self.bitboard[white][1][MAPMOVEIDX[1][mv] as usize] |= MAPMOVEVALUE[1][mv];
                self.bitboard[empty][1][MAPMOVEIDX[1][mv] as usize] ^= MAPMOVEVALUE[1][mv];
                self.bitboard[white][0][MAPMOVEIDX[2][mv] as usize] |= MAPMOVEVALUE[2][mv];
                self.bitboard[empty][0][MAPMOVEIDX[2][mv] as usize] ^= MAPMOVEVALUE[2][mv];
                self.bitboard[white][1][MAPMOVEIDX[3][mv] as usize] |= MAPMOVEVALUE[3][mv];
                self.bitboard[empty][1][MAPMOVEIDX[3][mv] as usize] ^= MAPMOVEVALUE[3][mv];
            }
            _ => panic! {},
        }

        self.p_turn = def;
    }

    fn turn(&self) -> Side {
        self.p_turn
    }

    pub fn can_play(&self, from: Square) -> bool {
        if self.state[from as usize] == Color::Empty {
            true
        } else {
            false
        }
    }
}

pub struct List {
    // legal move list
    p_move: [Move; (FILE_SIZE * RANK_SIZE) as usize],
    p_size: i32,
}

/// Use List to store legal moves.
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
}

// functions

fn square_make(fl: i32, rk: i32) -> Square {
    rk * (FILE_SIZE + 1) + fl
}

fn side_opp(sd: Side) -> Side {
    match sd {
        Side::White => Side::Black,
        Side::Black => Side::White,
        _ => panic!(""),
    }
}

fn pos_is_winner(pos: &Pos) -> bool {
    let current_side = side_opp(pos.p_turn);
    check_pattern5(&pos, current_side)
}

fn pos_is_draw(pos: &Pos) -> bool {
    let mut found: bool = true;

    for rk in 0..RANK_SIZE {
        for fl in 0..FILE_SIZE {
            let sq: Square = square_make(fl, rk);
            if pos.can_play(sq) {
                found = false;
                break;
            }

            if found == false {
                break;
            }
        }
    }

    let mut out: bool = false;
    if found == true && !pos_is_winner(pos) {
        out = true;
    }

    out
}

#[target_feature(enable = "avx512f")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe fn pos_is_draw_avx512f(pos: &Pos) -> bool {
    let empty = Color::Empty as usize;

    let board0org = _mm512_loadu_epi32(&pos.bitboard[empty][0][0]);

    let answer = _mm512_set1_epi32(0);

    // if all empty is 0, all board is filled.
    let temp_mask = _mm512_mask_cmp_epi32_mask(0b11111111_11111111, answer, board0org, 0);

    #[rustfmt::skip]
    if temp_mask == 1<<15|1<<14|1<<13|1<<12|1<<11|1<<10|1<<9|1<<8|1<<7|1<<6|1<<5|1<<4|1<<3|1<<2|1<<1|1<<0 && !pos_is_winner_avx512(pos) { return true } else { return false }
}

fn pos_is_end(pos: &Pos) -> bool {
    if pos_is_winner(pos) || pos_is_draw(pos) {
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
                Color::Border => print!("| "),
            }
        }

        println!("");
    }

    match pos.turn() {
        Color::Black => println!("black to play"),
        Color::White => println!("white to play"),
        _ => panic!(),
    }
}

fn gen_moves(list: &mut List, pos: &Pos) {
    list.clear();

    for rk in 0..RANK_SIZE {
        for fl in 0..FILE_SIZE {
            let sq: Square = square_make(fl, rk);
            if pos.can_play(sq) {
                list.add(sq);
            }
        }
    }
}

/// AI: use Minimax search with alpha-beta pruning
fn search(pos: &Pos, alpha: i32, beta: i32, depth: i32, _ply: i32) -> i32 {
    assert!(-EVAL_INF <= alpha && alpha < beta && beta <= EVAL_INF);
    // leaf?

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe {
                if pos_is_winner_avx512(&pos) {
                    return -EVAL_INF + _ply;
                }

                if pos_is_draw_avx512f(&pos) {
                    return 0;
                }
            }
        } else {
            if pos_is_winner(&pos) {
                return -EVAL_INF + _ply;
            }

            if pos_is_draw(&pos) {
                return 0;
            }
        }
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        if pos_is_winner(&pos) {
            return -EVAL_INF + _ply;
        }

        if pos_is_draw(&pos) {
            return 0;
        }
    }

    if depth == 0 {
        return eval(&pos, _ply);
    }

    let p_move_new: [Move; (FILE_SIZE * RANK_SIZE) as usize] =
        [0; (FILE_SIZE * RANK_SIZE) as usize];

    let mut list = List {
        p_move: p_move_new,
        p_size: 0,
    };

    let mut bm: Move = MOVE_NONE;
    let mut bs: i32 = SCORE_NONE;

    gen_moves(&mut list, &pos);

    // move loop

    if _ply == 0 {
        list.shuffle();
    }

    for i in 0..list.size() {
        if bs < beta {
            let mv: Move = list.p_move[i as usize];

            let mut new_pos = Pos {
                state: pos.state,
                p_turn: pos.p_turn,
                bitboard: pos.bitboard,
            };

            new_pos.do_move(mv);

            let sc: i32 = -search(&new_pos, -beta, -cmp::max(alpha, bs), depth - 1, _ply + 1);

            if sc > bs {
                bm = mv;
                bs = sc;
            }
        }
    }

    assert!(bm != MOVE_NONE);
    assert!(bs >= -EVAL_INF && bs <= EVAL_INF);

    if _ply == 0 {
        bm
    } else {
        bs
    } //best move at the root node, best score elsewhere
}

/// Evaluation function: give different scores to different patterns after a fixed depth.
fn eval(pos: &Pos, _ply: i32) -> i32 {
    let atk: Side = pos.turn();
    let def: Side = side_opp(atk);

    // check if opp has live4 which will win playing next move
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe {
                if check_patternlive4_avx512(&pos, def) {
                    return -4096;
                }
            }
        } else {
            if check_patternlive4(&pos, def) {
                return -4096;
            }
        }
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        if check_patternlive4(&pos, def) {
            return -4096;
        }
    }

    // check if self has live4 which will win playing next move
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe {
                if check_patternlive4_avx512(&pos, atk) {
                    return 2560;
                }
            }
        } else {
            if check_patternlive4(&pos, atk) {
                return 2560;
            }
        }
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        if check_patternlive4(&pos, atk) {
            return 2560;
        }
    }

    // check if self has dead4 which will win playing next move
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe {
                if check_patterndead4_avx512(&pos, atk) > 0 {
                    return 2560;
                }
            }
        } else {
            if check_patterndead4(&pos, atk) > 0 {
                return 2560;
            }
        }
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        if check_patterndead4(&pos, atk) > 0 {
            return 2560;
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe {
                let n_c4: i32 = check_patterndead4_avx512(&pos, def);
                let n_c3: i32 = check_patternlive3_avx512(&pos, def);

                // check if opp has 2 dead4 which will win playing next move
                if n_c4 > 1 {
                    return -2048;
                }

                // check if opp has a dead 4 and live 3 which will win playing the next two move
                if n_c4 == 1 && n_c3 > 0 {
                    return -2048;
                }

                if check_patternlive3_avx512(&pos, atk) > 1 {
                    return 2560;
                }

                // check if opp has 2 live3 which will win playing the next two move
                if n_c3 > 1 {
                    return -2048;
                }
            }
        } else {
            let n_c4: i32 = check_patterndead4(&pos, def);
            let n_c3: i32 = check_patternlive3(&pos, def);

            // check if opp has 2 dead4 which will win playing next move
            if n_c4 > 1 {
                return -2048;
            }

            // check if opp has a dead 4 and live 3 which will win playing the next two move
            if n_c4 == 1 && n_c3 > 0 {
                return -2048;
            }

            // check if self has 2 live3 which will win playing the next two move
            if check_patternlive3(&pos, atk) > 1 {
                return 2560;
            }

            // check if opp has 2 live3 which will win playing the next two move
            if n_c3 > 1 {
                return -2048;
            }
        }
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        let n_c4: i32 = check_patterndead4(&pos, def);
        let n_c3: i32 = check_patternlive3(&pos, def);

        // check if opp has 2 dead4 which will win playing next move
        if n_c4 > 1 {
            return -2048;
        }

        // check if opp has a dead 4 and live 3 which will win playing the next two move
        if n_c4 == 1 && n_c3 > 0 {
            return -2048;
        }

        // check if self has 2 live3 which will win playing the next two move
        if check_patternlive3(&pos, atk) > 1 {
            return 2560;
        }

        // check if opp has 2 live3 which will win playing the next two move
        if n_c3 > 1 {
            return -2048;
        }
    }

    0
}

/// Check <b>OOOOO</b>
fn check_pattern5(pos: &Pos, sd: Side) -> bool {
    let mut n: i32 = 0;

    for rk in 0..RANK_SIZE {
        for fl in 0..FILE_SIZE {
            let sq: Square = square_make(fl, rk);

            for pat in 0..4 {
                let idx0 = sq;
                let idx1 = sq + DIRECTION[pat][0];
                let idx2 = sq + DIRECTION[pat][1];
                let idx3 = sq + DIRECTION[pat][2];
                let idx4 = sq + DIRECTION[pat][3];

                let val0 = pos.state[idx0 as usize];
                let val1 = pos.state[idx1 as usize];
                let val2 = pos.state[idx2 as usize];
                let val3 = pos.state[idx3 as usize];
                let val4 = pos.state[idx4 as usize];

                #[rustfmt::skip]
                if val0 == sd && val1 == sd && val2 == sd && val3 == sd && val4 == sd { n += 1; }
            }
        }
    }

    if n > 0 {
        true
    } else {
        false
    }
}

/// Check <b>-OOOO-</b>
fn check_patternlive4(pos: &Pos, sd: Side) -> bool {
    let mut n: i32 = 0;

    for rk in 0..RANK_SIZE {
        for fl in 0..FILE_SIZE {
            let sq: Square = square_make(fl, rk);

            for pat in 0..4 {
                let idx0 = sq;
                let idx1 = sq + DIRECTION[pat][0];
                let idx2 = sq + DIRECTION[pat][1];
                let idx3 = sq + DIRECTION[pat][2];
                let idx4 = sq + DIRECTION[pat][3];
                let idx5 = sq + DIRECTION[pat][4];

                let val0 = pos.state[idx0 as usize];
                let val1 = pos.state[idx1 as usize];
                let val2 = pos.state[idx2 as usize];
                let val3 = pos.state[idx3 as usize];
                let val4 = pos.state[idx4 as usize];
                let val5 = pos.state[idx5 as usize];

                #[rustfmt::skip]
                if val0 == Color::Empty && val1 == sd && val2 == sd && val3 == sd && val4 == sd && val5 == Color::Empty { n += 1; }
            }
        }
    }

    if n > 0 {
        true
    } else {
        false
    }
}

/// Check <b>OOOO-, OOO-O, OO-OO, O-OOO, -OOOO</b>
fn check_patterndead4(pos: &Pos, sd: Side) -> i32 {
    let mut n: i32 = 0;

    for rk in 0..RANK_SIZE {
        for fl in 0..FILE_SIZE {
            let sq: Square = square_make(fl, rk);

            for dir in 0..4 {
                let idx0 = sq;
                let idx1 = sq + DIRECTION[dir][0];
                let idx2 = sq + DIRECTION[dir][1];
                let idx3 = sq + DIRECTION[dir][2];
                let idx4 = sq + DIRECTION[dir][3];

                let val0 = pos.state[idx0 as usize];
                let val1 = pos.state[idx1 as usize];
                let val2 = pos.state[idx2 as usize];
                let val3 = pos.state[idx3 as usize];
                let val4 = pos.state[idx4 as usize];

                #[rustfmt::skip]
                if val0 == sd && val1 == sd && val2 == sd && val3 == sd && val4 == Color::Empty { n += 1; }
                #[rustfmt::skip]
                if val0 == sd && val1 == sd && val2 == sd && val3 == Color::Empty && val4 == sd { n += 1; }
                #[rustfmt::skip]
                if val0 == sd && val1 == sd && val2 == Color::Empty && val3 == sd && val4 == sd { n += 1; }
                #[rustfmt::skip]
                if val0 == sd && val1 == Color::Empty && val2 == sd && val3 == sd && val4 == sd { n += 1; }
                #[rustfmt::skip]
                if val0 == Color::Empty && val1 == sd && val2 == sd && val3 == sd && val4 == sd { n += 1; }
            }
        }
    }

    n
}

/// Check <b>-OOO-, -OO-O-, -O-OO-</br>
fn check_patternlive3(pos: &Pos, sd: Side) -> i32 {
    let mut n: i32 = 0;

    for rk in 0..RANK_SIZE {
        for fl in 0..FILE_SIZE {
            let sq: Square = square_make(fl, rk);

            for dir in 0..4 {
                let idx0 = sq;
                let idx1 = sq + DIRECTION[dir][0];
                let idx2 = sq + DIRECTION[dir][1];
                let idx3 = sq + DIRECTION[dir][2];
                let idx4 = sq + DIRECTION[dir][3];
                let idx5 = sq + DIRECTION[dir][4];

                let val0 = pos.state[idx0 as usize];
                let val1 = pos.state[idx1 as usize];
                let val2 = pos.state[idx2 as usize];
                let val3 = pos.state[idx3 as usize];
                let val4 = pos.state[idx4 as usize];
                let val5 = pos.state[idx5 as usize];

                #[rustfmt::skip]
                if val0 == Color::Empty && val1 == sd && val2 == sd && val3 == sd && val4 == Color::Empty { n +=1; }
                #[rustfmt::skip]
                if val0 == Color::Empty && val1 == sd && val2 == sd && val3 == Color::Empty && val4 == sd && val5 == Color::Empty { n += 1; }
                #[rustfmt::skip]
                if val0 == Color::Empty && val1 == sd && val2 == Color::Empty && val3 == sd && val4 == sd && val5 == Color::Empty { n += 1; }
            }
        }
    }

    n
}

#[target_feature(enable = "avx512f")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe fn pos_is_winner_avx512(pos: &Pos) -> bool {
    let current_side = side_opp(pos.p_turn);
    let coloridx = current_side as usize;

    let board0org: [__m512i; 2] = [
        _mm512_loadu_epi32(&pos.bitboard[coloridx][0][0]),
        _mm512_loadu_epi32(&pos.bitboard[coloridx][1][0]),
    ]; // load states from bitboard

    #[rustfmt::skip]
    let answer = _mm512_set1_epi32( (1<<31)|(1<<30)|(1<<29)|(1<<28)|(1<<27) ); // an unbroken chain of five moves

    #[rustfmt::skip]
    let answer_mask: [__mmask16; 11] = [0b11111111_11111111,
                                          0b11111111_11111111,
                                          0b11111111_11111101, // use mask 0 to pass the overlapping
                                          0b11111111_11111001,
                                          0b11111111_11110001,
                                          0b11111111_11100001,
                                          0b11111111_11000011, // row 16 starts here
                                          0b11111111_10000111,
                                          0b11111111_10001111,
                                          0b11111111_10011111,
                                          0b11111111_10111111,];

    let mut temp_mask: [[__mmask16; (11 + 11)]; 2] = [[0; (11 + 11)]; 2]; // total possible patterns

    for dir in 0..2 {
        // direction 0 and 1
        let mut board0 = board0org[dir];
        let boardf = _mm512_and_epi32(answer, board0);
        temp_mask[dir][0] = _mm512_mask_cmp_epi32_mask(0b01111111_11111111, answer, boardf, 0); // match OOOOO

        for i in 1..11 {
            // OOOOOOOOOOO----, the last 4 "-" cannot make an unbroken chain of five.
            board0 = _mm512_rol_epi32(board0, 1); // rotate one space left
            let boardf = _mm512_and_epi32(answer, board0); // filter out except the pattern
            temp_mask[dir][i as usize] =
                _mm512_mask_cmp_epi32_mask(0b01111111_11111111, answer, boardf, 0);
            // see if it matches the pattern
        }

        board0 = _mm512_rol_epi32(board0, 6); // whatever 11,12,13,14 are occupied, it cannot match an unbroken chain of five moves. Therefore, shift from direction 0 to 2.

        // direction 2 and 3
        let boardf = _mm512_and_epi32(answer, board0);
        temp_mask[dir][11] = _mm512_mask_cmp_epi32_mask(answer_mask[0], answer, boardf, 0);

        for i in 12..22 {
            let idx: i32 = i - 11;
            board0 = _mm512_rol_epi32(board0, 1);
            let boardf = _mm512_and_epi32(answer, board0);
            temp_mask[dir][i as usize] =
                _mm512_mask_cmp_epi32_mask(answer_mask[idx as usize], answer, boardf, 0);
        }
    }

    let mut n: i32 = 0;

    // calculate how many patterns matched
    for i in 0..2 {
        for j in 0..(11 + 11) {
            n += _popcnt32(temp_mask[i][j] as i32);
        }
    }

    if n > 0 {
        return true;
    } else {
        return false;
    }
}

#[target_feature(enable = "avx512f")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe fn check_patternlive4_avx512(pos: &Pos, sd: Side) -> bool {
    let coloridx = sd as usize;
    let emptyidx = Color::Empty as usize;

    #[rustfmt::skip]
    let answer_color = _mm512_set1_epi32(         (1<<30)|(1<<29)|(1<<28)|(1<<27)         );
    #[rustfmt::skip]
    let answer_empty = _mm512_set1_epi32( (1<<31)|                                (1<<26) );
    #[rustfmt::skip]
    let answer       = _mm512_set1_epi32( (1<<31)|(1<<30)|(1<<29)|(1<<28)|(1<<27)|(1<<26) );

    #[rustfmt::skip]
    let answer_mask: [__mmask16; 10] = [0b11111111_11111110, 
                                        0b11111111_11111100,
                                        0b11111111_11111000,
                                        0b11111111_11110000,
                                        0b11111111_11100000,
                                        0b01111111_11000000,
                                        0b00111111_10000010,
                                        0b00011111_00000110,
                                        0b00001110_00001110,
                                        0b00000100_00011110,];

    let board0org: [__m512i; 2] = [
        _mm512_loadu_epi32(&pos.bitboard[coloridx][0][0]),
        _mm512_loadu_epi32(&pos.bitboard[coloridx][1][0]),
    ];
    let board1org: [__m512i; 2] = [
        _mm512_loadu_epi32(&pos.bitboard[emptyidx][0][0]),
        _mm512_loadu_epi32(&pos.bitboard[emptyidx][1][0]),
    ];

    let mut temp_mask: [[__mmask16; 10 + 10]; 2] = [[0; 10 + 10]; 2];

    for dir in 0..2 {
        let mut board0 = board0org[dir];
        let mut board1 = board1org[dir];

        let boardf1 = _mm512_and_epi32(answer_color, board0);
        let boardf2 = _mm512_and_epi32(answer_empty, board1);
        let boardf = _mm512_or_epi32(boardf1, boardf2);

        temp_mask[dir][0] = _mm512_mask_cmp_epi32_mask(0b01111111_11111111, answer, boardf, 0);

        for i in 1..10 {
            board0 = _mm512_rol_epi32(board0, 1);
            board1 = _mm512_rol_epi32(board1, 1);

            let boardf1 = _mm512_and_epi32(answer_color, board0);
            let boardf2 = _mm512_and_epi32(answer_empty, board1);
            let boardf = _mm512_or_epi32(boardf1, boardf2);

            temp_mask[dir][i as usize] =
                _mm512_mask_cmp_epi32_mask(0b01111111_11111111, answer, boardf, 0);
        }

        board0 = _mm512_rol_epi32(board0, 7);
        board1 = _mm512_rol_epi32(board1, 7);

        let boardf1 = _mm512_and_epi32(answer_color, board0);
        let boardf2 = _mm512_and_epi32(answer_empty, board1);
        let boardf = _mm512_or_epi32(boardf1, boardf2);

        temp_mask[dir][10] = _mm512_mask_cmp_epi32_mask(answer_mask[0], answer, boardf, 0);

        for i in 11..20 {
            let idx: i32 = i - 10;

            board0 = _mm512_rol_epi32(board0, 1);
            board1 = _mm512_rol_epi32(board1, 1);

            let boardf1 = _mm512_and_epi32(answer_color, board0);
            let boardf2 = _mm512_and_epi32(answer_empty, board1);
            let boardf = _mm512_or_epi32(boardf1, boardf2);

            temp_mask[dir][i as usize] =
                _mm512_mask_cmp_epi32_mask(answer_mask[idx as usize], answer, boardf, 0);
        }
    }

    let mut n: i32 = 0;

    for i in 0..2 {
        for j in 0..20 {
            n += _popcnt32(temp_mask[i][j] as i32);
        }
    }

    if n > 0 {
        return true;
    } else {
        return false;
    }
}

#[target_feature(enable = "avx512f")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe fn check_patterndead4_avx512(pos: &Pos, sd: Side) -> i32 {
    let coloridx = sd as usize;
    let emptyidx = Color::Empty as usize;

    #[rustfmt::skip]
    let answer_color: [__m512i; 5] = [_mm512_set1_epi32(         (1<<30)|(1<<29)|(1<<28)|(1<<27) ),
                                      _mm512_set1_epi32( (1<<31)|        (1<<29)|(1<<28)|(1<<27) ),
                                      _mm512_set1_epi32( (1<<31)|(1<<30)        |(1<<28)|(1<<27) ),
                                      _mm512_set1_epi32( (1<<31)|(1<<30)|(1<<29)        |(1<<27) ),
                                      _mm512_set1_epi32( (1<<31)|(1<<30)|(1<<29)|(1<<28)         )];

    #[rustfmt::skip]
    let answer_empty: [__m512i; 5]= [_mm512_set1_epi32( 1<<31 ),
                                     _mm512_set1_epi32(          1<<30 ),
                                     _mm512_set1_epi32(                  1<<29 ),
                                     _mm512_set1_epi32(                          1<<28 ),
                                     _mm512_set1_epi32(                                  1<<27)];

    #[rustfmt::skip]
    let answer       = _mm512_set1_epi32( (1<<31)|(1<<30)|(1<<29)|(1<<28)|(1<<27));

    #[rustfmt::skip]
    let answer_mask: [__mmask16; 11] = [0b11111111_11111111,
                                        0b11111111_11111111,
                                        0b11111111_11111101,
                                        0b11111111_11111001,
                                        0b11111111_11110001,
                                        0b11111111_11100001,
                                        0b11111111_11000011,
                                        0b11111111_11000111,
                                        0b11111111_11001111,
                                        0b11111111_11011111,
                                        0b11111111_11111111,];

    let board0org: [__m512i; 2] = [
        _mm512_loadu_epi32(&pos.bitboard[coloridx][0][0]),
        _mm512_loadu_epi32(&pos.bitboard[coloridx][1][0]),
    ];
    let board1org: [__m512i; 2] = [
        _mm512_loadu_epi32(&pos.bitboard[emptyidx][0][0]),
        _mm512_loadu_epi32(&pos.bitboard[emptyidx][1][0]),
    ];

    let mut temp_mask: [[[__mmask16; 11 + 11]; 2]; 5] = [[[0; 11 + 11]; 2]; 5];

    for pattern in 0..5 {
        for dir in 0..2 {
            let mut board0 = board0org[dir];
            let mut board1 = board1org[dir];

            let boardf1 = _mm512_and_epi32(answer_color[pattern], board0);
            let boardf2 = _mm512_and_epi32(answer_empty[pattern], board1);
            let boardf = _mm512_or_epi32(boardf1, boardf2);

            temp_mask[pattern][dir][0] =
                _mm512_mask_cmp_epi32_mask(0b01111111_11111111, answer, boardf, 0);

            for i in 1..11 {
                board0 = _mm512_rol_epi32(board0, 1);
                board1 = _mm512_rol_epi32(board1, 1);

                let boardf1 = _mm512_and_epi32(answer_color[pattern], board0);
                let boardf2 = _mm512_and_epi32(answer_empty[pattern], board1);
                let boardf = _mm512_or_epi32(boardf1, boardf2);

                temp_mask[pattern][dir][i as usize] =
                    _mm512_mask_cmp_epi32_mask(0b01111111_11111111, answer, boardf, 0);
            }

            board0 = _mm512_rol_epi32(board0, 6);
            board1 = _mm512_rol_epi32(board1, 6);

            let boardf1 = _mm512_and_epi32(answer_color[pattern], board0);
            let boardf2 = _mm512_and_epi32(answer_empty[pattern], board1);
            let boardf = _mm512_or_epi32(boardf1, boardf2);

            temp_mask[pattern][dir][11] =
                _mm512_mask_cmp_epi32_mask(answer_mask[0], answer, boardf, 0);

            for i in 12..22 {
                let idx: i32 = i - 11;

                board0 = _mm512_rol_epi32(board0, 1);
                board1 = _mm512_rol_epi32(board1, 1);

                let boardf1 = _mm512_and_epi32(answer_color[pattern], board0);
                let boardf2 = _mm512_and_epi32(answer_empty[pattern], board1);
                let boardf = _mm512_or_epi32(boardf1, boardf2);

                temp_mask[pattern][dir][i as usize] =
                    _mm512_mask_cmp_epi32_mask(answer_mask[idx as usize], answer, boardf, 0);
            }
        }
    }

    let mut count: i32 = 0;

    for i in 0..5 {
        for j in 0..2 {
            for k in 0..22 {
                count += _popcnt32(temp_mask[i][j][k] as i32);
            }
        }
    }

    count
}

#[target_feature(enable = "avx512f")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe fn check_patternlive3_avx512(pos: &Pos, sd: Side) -> i32 {
    let coloridx = sd as usize;
    let emptyidx = Color::Empty as usize;

    #[rustfmt::skip]
    let board0org: [__m512i; 2]  = [_mm512_loadu_epi32(&pos.bitboard[coloridx][0][0]), _mm512_loadu_epi32(&pos.bitboard[coloridx][1][0])];
    #[rustfmt::skip]
    let board1org: [__m512i; 2]  = [_mm512_loadu_epi32(&pos.bitboard[emptyidx][0][0]), _mm512_loadu_epi32(&pos.bitboard[emptyidx][1][0])];

    #[rustfmt::skip]
    let answer_color: [__m512i; 1] = [_mm512_set1_epi32(         (1<<30)|(1<<29)|(1<<28)         )];
    #[rustfmt::skip]
    let answer_empty: [__m512i; 1] = [_mm512_set1_epi32( (1<<31)|                        (1<<27) )];
    #[rustfmt::skip]
    let answer: __m512i = _mm512_set1_epi32( (1<<31)|(1<<30)|(1<<29)|(1<<28)|(1<<27) );

    #[rustfmt::skip]
    let answer_mask: [__mmask16; 11] = [0b11111111_11111111,
                                        0b11111111_11111111,
                                        0b11111111_11111101,
                                        0b11111111_11111001,
                                        0b11111111_11110001,
                                        0b11111111_11100001,
                                        0b11111111_11000011,
                                        0b11111111_11000111,
                                        0b11111111_11001111,
                                        0b11111111_11011111,
                                        0b11111111_11111111,];

    let mut temp_mask: [[[__mmask16; 11 + 11]; 2]; 1] = [[[0; 11 + 11]; 2]; 1];

    for pattern in 0..1 {
        for dir in 0..2 {
            let mut board0 = board0org[dir];
            let mut board1 = board1org[dir];

            let boardf1 = _mm512_and_epi32(answer_color[pattern], board0);
            let boardf2 = _mm512_and_epi32(answer_empty[pattern], board1);
            let boardf = _mm512_or_epi32(boardf1, boardf2);

            temp_mask[pattern][dir][0] =
                _mm512_mask_cmp_epi32_mask(0b01111111_11111111, answer, boardf, 0);

            for i in 1..11 {
                board0 = _mm512_rol_epi32(board0, 1);
                board1 = _mm512_rol_epi32(board1, 1);

                let boardf1 = _mm512_and_epi32(answer_color[pattern], board0);
                let boardf2 = _mm512_and_epi32(answer_empty[pattern], board1);
                let boardf = _mm512_or_epi32(boardf1, boardf2);

                temp_mask[pattern][dir][i as usize] =
                    _mm512_mask_cmp_epi32_mask(0b01111111_11111111, answer, boardf, 0);
            }

            board0 = _mm512_rol_epi32(board0, 6);
            board1 = _mm512_rol_epi32(board1, 6);

            let boardf1 = _mm512_and_epi32(answer_color[pattern], board0);
            let boardf2 = _mm512_and_epi32(answer_empty[pattern], board1);
            let boardf = _mm512_or_epi32(boardf1, boardf2);

            temp_mask[pattern][dir][11] =
                _mm512_mask_cmp_epi32_mask(answer_mask[0], answer, boardf, 0);

            for i in 12..22 {
                let idx: i32 = i - 11;

                board0 = _mm512_rol_epi32(board0, 1);
                board1 = _mm512_rol_epi32(board1, 1);

                let boardf1 = _mm512_and_epi32(answer_color[pattern], board0);
                let boardf2 = _mm512_and_epi32(answer_empty[pattern], board1);
                let boardf = _mm512_or_epi32(boardf1, boardf2);

                temp_mask[pattern][dir][i as usize] =
                    _mm512_mask_cmp_epi32_mask(answer_mask[idx as usize], answer, boardf, 0);
            }
        }
    }

    let mut count: i32 = 0;

    for i in 0..1 {
        for j in 0..2 {
            for k in 0..22 {
                count += _popcnt32(temp_mask[i][j][k] as i32);
            }
        }
    }

    #[rustfmt::skip]
    let answer_color: [__m512i; 2] = [_mm512_set1_epi32(          (1<<30)|        (1<<28)|(1<<27) ),
                                      _mm512_set1_epi32(          (1<<30)|(1<<29)        |(1<<27) )];

    #[rustfmt::skip]
    let answer_empty: [__m512i; 2] = [_mm512_set1_epi32( (1<<31)|         (1<<29)|                (1<<26) ),
                                      _mm512_set1_epi32( (1<<31)|                 (1<<28)|        (1<<26) )];

    #[rustfmt::skip]
    let answer: __m512i = _mm512_set1_epi32( (1<<31)|(1<<30)|(1<<29)|(1<<28)|(1<<27)|(1<<26) );

    #[rustfmt::skip]
    let answer_mask: [__mmask16; 10] = [0b11111111_11111111,
                                        0b11111111_11111101,
                                        0b11111111_11111001,
                                        0b11111111_11110001,
                                        0b11111111_11100001,
                                        0b11111111_11000001,
                                        0b11111111_11000011,
                                        0b11111111_11000111,
                                        0b11111111_11001111,
                                        0b11111111_11011111,];

    let mut temp_mask: [[[__mmask16; 10 + 10]; 2]; 2] = [[[0; 10 + 10]; 2]; 2];

    for pattern in 0..2 {
        for dir in 0..2 {
            let mut board0 = board0org[dir];
            let mut board1 = board1org[dir];

            let boardf1 = _mm512_and_epi32(answer_color[pattern], board0);
            let boardf2 = _mm512_and_epi32(answer_empty[pattern], board1);
            let boardf = _mm512_or_epi32(boardf1, boardf2);

            temp_mask[pattern][dir][0] =
                _mm512_mask_cmp_epi32_mask(0b01111111_11111111, answer, boardf, 0);

            for i in 1..10 {
                board0 = _mm512_rol_epi32(board0, 1);
                board1 = _mm512_rol_epi32(board1, 1);

                let boardf1 = _mm512_and_epi32(answer_color[pattern], board0);
                let boardf2 = _mm512_and_epi32(answer_empty[pattern], board1);
                let boardf = _mm512_or_epi32(boardf1, boardf2);

                temp_mask[pattern][dir][i as usize] =
                    _mm512_mask_cmp_epi32_mask(0b01111111_11111111, answer, boardf, 0);
            }

            board0 = _mm512_rol_epi32(board0, 7);
            board1 = _mm512_rol_epi32(board1, 7);

            let boardf1 = _mm512_and_epi32(answer_color[pattern], board0);
            let boardf2 = _mm512_and_epi32(answer_empty[pattern], board1);
            let boardf = _mm512_or_epi32(boardf1, boardf2);

            temp_mask[pattern][dir][10] =
                _mm512_mask_cmp_epi32_mask(answer_mask[0], answer, boardf, 0);

            for i in 11..20 {
                let idx: i32 = i - 10;

                board0 = _mm512_rol_epi32(board0, 1);
                board1 = _mm512_rol_epi32(board1, 1);

                let boardf1 = _mm512_and_epi32(answer_color[pattern], board0);
                let boardf2 = _mm512_and_epi32(answer_empty[pattern], board1);
                let boardf = _mm512_or_epi32(boardf1, boardf2);

                temp_mask[pattern][dir][i as usize] =
                    _mm512_mask_cmp_epi32_mask(answer_mask[idx as usize], answer, boardf, 0);
            }
        }
    }

    for i in 0..2 {
        for j in 0..2 {
            for k in 0..20 {
                count += _popcnt32(temp_mask[i][j][k] as i32);
            }
        }
    }

    count
}

fn main() {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx512f") {
            println!("\n\nThe program is running with avx512f intrinsics\n\n");
        } else {
            println!("\n\nThe program is running with NO intrinsics.\n\n");
        }
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        println!("\n\nThe program is running with NO intrinsics.\n\n");
    }

    loop {
        let start = Instant::now();

        println!("Hello, this is Connect5 (Outer-Open Gomoku)!");
        println!("Self-playing with search depth = 4");

        let test_state: [Color; SQUARE_SIZE as usize] = [Color::Empty; SQUARE_SIZE as usize];
        let test_bitboard: [[[i32; 16]; 2]; 3] = [[[0; 16]; 2]; 3];

        let mut test1 = Pos {
            state: test_state,
            p_turn: Color::Black,
            bitboard: test_bitboard,
        };

        test1.init();

        let mut count: i32 = 0;

        for i in 0..(FILE_SIZE * RANK_SIZE) {
            let mut next_move: Move = square_make(1, 7); // set the first move is (1,7)

            if i > 0 {
                next_move = search(&test1, -EVAL_INF, EVAL_INF, 4, 0);
            } // search depth = 4

            test1.do_move(next_move);
            pos_disp(&test1);

            if pos_is_end(&test1) {
                println!("Game over!!!!!! at Move {}", i);
                count = i + 1;
                break;
            }
        }

        let duration = start.elapsed();

        println!(
            "Average time for each move is: {:?}",
            duration / count as u32
        );
    }
}
