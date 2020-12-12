//! <b>Outer-Open Gomoku</b> is a board game which is a enchanced version of connect5 (Gomoku).\
//! The game is a two-player game which played on a 15x15 Go board.\
//! Two players take turns placing a move on an empty intersection in this board.\
//! The winner is the first player to form an unbroken chain of five moves horizontally, vertically, or diagonally.\
//! Unlike Gomoku, the first move is required to be placed at the two outer rows or columns of this board.\
//! This program provides an AI playing with Minimax search with alpha-beta pruning.\
//! On a skylake-sp machine, with avx-512 4.0HZ. The avx512f version is faster than sequence
//! version 5 times.
//! This example, will show xxxxxx

#![feature(stdsimd, avx512_target_feature)]

use rand::seq::SliceRandom;
use rand::thread_rng;
use rand::Rng;

use std::cmp;
use std::time::{Duration, Instant};

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
type Piece = Color;

// constants

const FILE_SIZE: i32 = 15;
const RANK_SIZE: i32 = 15;
const SQUARE_SIZE: i32 = (FILE_SIZE + 1) * (FILE_SIZE + 4) + 16 + 4;

const EVAL_INF: i32 = (FILE_SIZE * RANK_SIZE * 100);
const MOVE_NONE: Move = -1;
const SCORE_NONE: i32 = -EVAL_INF - 1;

/// PATTERN 0: left to right\
/// PATTERN 1: top to bottom\
/// PATTERN 2: top left to bottom right\
/// PATTERN 3: top right to bottom left
const PATTERN: [[i32; 5]; 4] = [ [1, 2, 3, 4, 5],
                                 [1 * (FILE_SIZE + 1), 2 * (FILE_SIZE + 1), 3 * (FILE_SIZE + 1), 4 * (FILE_SIZE + 1), 5 * (FILE_SIZE + 1)],
                                 [1 * (FILE_SIZE + 2), 2 * (FILE_SIZE + 2), 3 * (FILE_SIZE + 2), 4 * (FILE_SIZE + 2), 5 * (FILE_SIZE + 2)],
                                 [1 * (FILE_SIZE + 0), 2 * (FILE_SIZE + 0), 3 * (FILE_SIZE + 0), 4 * (FILE_SIZE + 0), 5 * (FILE_SIZE + 0)]];

/// A table to encode each position to a value for 4 Patterns
const MAPMOVEVALUE: [[i32; 239]; 4] = [ [ 
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
                                        [ 
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
                                        [ 
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
                                        [ 
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

/// A table to encode each position to an index for 4 Patterns
const MAPMOVEIDX: [[i32; 239]; 4] = [ [ 
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

                                      [ 
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

                                      [ 
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

                                      [ 
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

pub struct Pos { // position
    state: [Color; SQUARE_SIZE as usize],
    p_turn: Side,

    bitboard: [[[i32; 16]; 2]; 3], 
}

impl Pos {

    pub fn init(&mut self) { // starting position
        for i in 0..SQUARE_SIZE as usize {
            self.state[i] = Color::Border;
        }

        for rk in 0..RANK_SIZE {
            for fl in 0..FILE_SIZE {
                let sq: Square = square_make(fl, rk);
                self.state[sq as usize] = Color::Empty;
            }
        }

        self.p_turn = Color::Black;

        //--------------------------------------------

        for i in 0..2 {
            for j in 0..16 { 
                self.bitboard[Color::Black as usize][i][j] = 0; 
            }
        }

        for i in 0..2 {
            for j in 0..16 { 
                self.bitboard[Color::White as usize][i][j] = 0;
            }
        }

        for i in 0..2 {
            for j in 0..16 { 
                self.bitboard[Color::Empty as usize][i][j] = 0;
            }
        }

        for i in 0..2 {
            for j in 0..FILE_SIZE as usize { 
                self.bitboard[Color::Empty as usize][i][j] = (1<<31)|(1<<30)|(1<<29)|(1<<28)|(1<<27)|(1<<26)|(1<<25)|(1<<24)|(1<<23)|(1<<22)|(1<<21)|(1<<20)|(1<<19)|(1<<18)|(1<<17); 
            }
        }
        
        self.bitboard[Color::Empty as usize][0][0]  |= (1<<15)|(1<<14)|(1<<13)|(1<<12)|(1<<11);
        self.bitboard[Color::Empty as usize][0][1]  |= (1<<15)|(1<<14)|(1<<13)|(1<<12)|(1<<11)|(1<<10)  |(1<<9)|(1<<8)|(1<<7)|(1<<6)|(1<<5)|(1<<4)|(1<<3)|(1<<2)|(1<<1);
        self.bitboard[Color::Empty as usize][0][2]  |= (1<<15)|(1<<14)|(1<<13)|(1<<12)|(1<<11)|(1<<10)|(1<<9)  |(1<<8)|(1<<7)|(1<<6)|(1<<5)|(1<<4)|(1<<3)|(1<<2)|(1<<1);
        self.bitboard[Color::Empty as usize][0][3]  |= (1<<15)|(1<<14)|(1<<13)|(1<<12)|(1<<11)|(1<<10)|(1<<9)|(1<<8)  |(1<<7)|(1<<6)|(1<<5)|(1<<4)|(1<<3)|(1<<2)|(1<<1);
        self.bitboard[Color::Empty as usize][0][4]  |= (1<<15)|(1<<14)|(1<<13)|(1<<12)|(1<<11)|(1<<10)|(1<<9)|(1<<8)|(1<<7)  |(1<<6)|(1<<5)|(1<<4)|(1<<3)|(1<<2)|(1<<1);
        self.bitboard[Color::Empty as usize][0][5]  |= (1<<15)|(1<<14)|(1<<13)|(1<<12)|(1<<11)|(1<<10)|(1<<9)|(1<<8)|(1<<7)|(1<<6)  |(1<<5)|(1<<4)|(1<<3)|(1<<2)|(1<<1);
        self.bitboard[Color::Empty as usize][0][6]  |= (1<<15)|(1<<14)|(1<<13)|(1<<12)|(1<<11)|(1<<10)|(1<<9)|(1<<8)|(1<<7)|(1<<6)|(1<<5);
        self.bitboard[Color::Empty as usize][0][7]  |= (1<<15)|(1<<14)|(1<<13)|(1<<12)|(1<<11)|(1<<10)|(1<<9)|(1<<8)|(1<<7)|(1<<6)|(1<<5)|(1<<4);
        self.bitboard[Color::Empty as usize][0][8]  |= (1<<15)|(1<<14)|(1<<13)|(1<<12)|(1<<11)|(1<<10)|(1<<9)|(1<<8)|(1<<7)|(1<<6)|(1<<5)|(1<<4)|(1<<3);
        self.bitboard[Color::Empty as usize][0][9]  |= (1<<15)|(1<<14)|(1<<13)|(1<<12)|(1<<11)|(1<<10)|(1<<9)|(1<<8)|(1<<7)|(1<<6)|(1<<5)|(1<<4)|(1<<3)|(1<<2);
        self.bitboard[Color::Empty as usize][0][10] |= (1<<15)|(1<<14)|(1<<13)|(1<<12)|(1<<11)|(1<<10)|(1<<9)|(1<<8)|(1<<7)|(1<<6)|(1<<5)|(1<<4)|(1<<3)|(1<<2)|(1<<1);
        self.bitboard[Color::Empty as usize][0][11] |= (1<<15)|(1<<14)|(1<<13)|(1<<12)|(1<<11)|(1<<10)|(1<<9)|(1<<8)|(1<<7)|(1<<6)|(1<<5)|(1<<4)|(1<<3)|(1<<2);
        self.bitboard[Color::Empty as usize][0][12] |= (1<<15)|(1<<14)|(1<<13)|(1<<12)|(1<<11)|(1<<10)|(1<<9)|(1<<8)|(1<<7)|(1<<6)|(1<<5)|(1<<4)|(1<<3);
        self.bitboard[Color::Empty as usize][0][13] |= (1<<15)|(1<<14)|(1<<13)|(1<<12)|(1<<11)|(1<<10)|(1<<9)|(1<<8)|(1<<7)|(1<<6)|(1<<5)|(1<<4);
        self.bitboard[Color::Empty as usize][0][14] |= (1<<15)|(1<<14)|(1<<13)|(1<<12)|(1<<11)|(1<<10)|(1<<9)|(1<<8)|(1<<7)|(1<<6)|(1<<5);
        self.bitboard[Color::Empty as usize][0][15] |= (1<<15)|(1<<14)|(1<<13)|(1<<12)|(1<<11)|(1<<10)|(1<<9)|(1<<8)|(1<<7)|(1<<6);

        self.bitboard[Color::Empty as usize][1][0]  |= (1<<15)|(1<<14)|(1<<13)|(1<<12)|(1<<11);
        self.bitboard[Color::Empty as usize][1][1]  |= (1<<15)|(1<<14)|(1<<13)|(1<<12)|(1<<11)|(1<<10)  |(1<<9)|(1<<8)|(1<<7)|(1<<6)|(1<<5)|(1<<4)|(1<<3)|(1<<2)|(1<<1);
        self.bitboard[Color::Empty as usize][1][2]  |= (1<<15)|(1<<14)|(1<<13)|(1<<12)|(1<<11)|(1<<10)|(1<<9)  |(1<<8)|(1<<7)|(1<<6)|(1<<5)|(1<<4)|(1<<3)|(1<<2)|(1<<1);
        self.bitboard[Color::Empty as usize][1][3]  |= (1<<15)|(1<<14)|(1<<13)|(1<<12)|(1<<11)|(1<<10)|(1<<9)|(1<<8)  |(1<<7)|(1<<6)|(1<<5)|(1<<4)|(1<<3)|(1<<2)|(1<<1);
        self.bitboard[Color::Empty as usize][1][4]  |= (1<<15)|(1<<14)|(1<<13)|(1<<12)|(1<<11)|(1<<10)|(1<<9)|(1<<8)|(1<<7)  |(1<<6)|(1<<5)|(1<<4)|(1<<3)|(1<<2)|(1<<1);
        self.bitboard[Color::Empty as usize][1][5]  |= (1<<15)|(1<<14)|(1<<13)|(1<<12)|(1<<11)|(1<<10)|(1<<9)|(1<<8)|(1<<7)|(1<<6)  |(1<<5)|(1<<4)|(1<<3)|(1<<2)|(1<<1);
        self.bitboard[Color::Empty as usize][1][6]  |= (1<<15)|(1<<14)|(1<<13)|(1<<12)|(1<<11)|(1<<10)|(1<<9)|(1<<8)|(1<<7)|(1<<6)|(1<<5);
        self.bitboard[Color::Empty as usize][1][7]  |= (1<<15)|(1<<14)|(1<<13)|(1<<12)|(1<<11)|(1<<10)|(1<<9)|(1<<8)|(1<<7)|(1<<6)|(1<<5)|(1<<4);
        self.bitboard[Color::Empty as usize][1][8]  |= (1<<15)|(1<<14)|(1<<13)|(1<<12)|(1<<11)|(1<<10)|(1<<9)|(1<<8)|(1<<7)|(1<<6)|(1<<5)|(1<<4)|(1<<3);
        self.bitboard[Color::Empty as usize][1][9]  |= (1<<15)|(1<<14)|(1<<13)|(1<<12)|(1<<11)|(1<<10)|(1<<9)|(1<<8)|(1<<7)|(1<<6)|(1<<5)|(1<<4)|(1<<3)|(1<<2);
        self.bitboard[Color::Empty as usize][1][10] |= (1<<15)|(1<<14)|(1<<13)|(1<<12)|(1<<11)|(1<<10)|(1<<9)|(1<<8)|(1<<7)|(1<<6)|(1<<5)|(1<<4)|(1<<3)|(1<<2)|(1<<1);
        self.bitboard[Color::Empty as usize][1][11] |= (1<<15)|(1<<14)|(1<<13)|(1<<12)|(1<<11)|(1<<10)|(1<<9)|(1<<8)|(1<<7)|(1<<6)|(1<<5)|(1<<4)|(1<<3)|(1<<2);
        self.bitboard[Color::Empty as usize][1][12] |= (1<<15)|(1<<14)|(1<<13)|(1<<12)|(1<<11)|(1<<10)|(1<<9)|(1<<8)|(1<<7)|(1<<6)|(1<<5)|(1<<4)|(1<<3);
        self.bitboard[Color::Empty as usize][1][13] |= (1<<15)|(1<<14)|(1<<13)|(1<<12)|(1<<11)|(1<<10)|(1<<9)|(1<<8)|(1<<7)|(1<<6)|(1<<5)|(1<<4);
        self.bitboard[Color::Empty as usize][1][14] |= (1<<15)|(1<<14)|(1<<13)|(1<<12)|(1<<11)|(1<<10)|(1<<9)|(1<<8)|(1<<7)|(1<<6)|(1<<5);
        self.bitboard[Color::Empty as usize][1][15] |= (1<<15)|(1<<14)|(1<<13)|(1<<12)|(1<<11)|(1<<10)|(1<<9)|(1<<8)|(1<<7)|(1<<6);
    } 

    pub fn do_move(&mut self, mv: Move) {

        let atk: Side = self.p_turn;
        let def: Side = side_opp(atk);

        let mv = mv as usize;

        match self.p_turn {
            Color::Black => { self.state[mv as usize] = Color::Black;

                              //for i in 0..4 {
                              //    self.bitboard[Color::Black as usize][i][ MAPMOVEIDX[i][mv as usize] as usize ] |= MAPMOVEVALUE[i][mv as usize];
                              //    self.bitboard[Color::Empty as usize][i][ MAPMOVEIDX[i][mv as usize] as usize ] ^= MAPMOVEVALUE[i][mv as usize];
                              //}


                                  self.bitboard[Color::Black as usize][0][ MAPMOVEIDX[0][mv] as usize ] |= MAPMOVEVALUE[0][mv];
                                  self.bitboard[Color::Empty as usize][0][ MAPMOVEIDX[0][mv] as usize ] ^= MAPMOVEVALUE[0][mv];
                                  self.bitboard[Color::Black as usize][1][ MAPMOVEIDX[1][mv] as usize ] |= MAPMOVEVALUE[1][mv];
                                  self.bitboard[Color::Empty as usize][1][ MAPMOVEIDX[1][mv] as usize ] ^= MAPMOVEVALUE[1][mv];
                                  self.bitboard[Color::Black as usize][0][ MAPMOVEIDX[2][mv] as usize ] |= MAPMOVEVALUE[2][mv];
                                  self.bitboard[Color::Empty as usize][0][ MAPMOVEIDX[2][mv] as usize ] ^= MAPMOVEVALUE[2][mv];
                                  self.bitboard[Color::Black as usize][1][ MAPMOVEIDX[3][mv] as usize ] |= MAPMOVEVALUE[3][mv];
                                  self.bitboard[Color::Empty as usize][1][ MAPMOVEIDX[3][mv] as usize ] ^= MAPMOVEVALUE[3][mv];
            },

            Color::White => { self.state[mv as usize] = Color::White;

                              //for i in 0..4 {
                              //    self.bitboard[Color::White as usize][i][ MAPMOVEIDX[i][mv as usize] as usize ] |= MAPMOVEVALUE[i][mv as usize];
                              //    self.bitboard[Color::Empty as usize][i][ MAPMOVEIDX[i][mv as usize] as usize ] ^= MAPMOVEVALUE[i][mv as usize];
                              //}
                                  self.bitboard[Color::White as usize][0][ MAPMOVEIDX[0][mv] as usize ] |= MAPMOVEVALUE[0][mv];
                                  self.bitboard[Color::Empty as usize][0][ MAPMOVEIDX[0][mv] as usize ] ^= MAPMOVEVALUE[0][mv];
                                  self.bitboard[Color::White as usize][1][ MAPMOVEIDX[1][mv] as usize ] |= MAPMOVEVALUE[1][mv];
                                  self.bitboard[Color::Empty as usize][1][ MAPMOVEIDX[1][mv] as usize ] ^= MAPMOVEVALUE[1][mv];
                                  self.bitboard[Color::White as usize][0][ MAPMOVEIDX[2][mv] as usize ] |= MAPMOVEVALUE[2][mv];
                                  self.bitboard[Color::Empty as usize][0][ MAPMOVEIDX[2][mv] as usize ] ^= MAPMOVEVALUE[2][mv];
                                  self.bitboard[Color::White as usize][1][ MAPMOVEIDX[3][mv] as usize ] |= MAPMOVEVALUE[3][mv];
                                  self.bitboard[Color::Empty as usize][1][ MAPMOVEIDX[3][mv] as usize ] ^= MAPMOVEVALUE[3][mv];
            },

            Color::Empty => panic!{},
            Color::Border => panic!{},
        }

        self.p_turn = def; 
    }

    fn turn(&self) -> Side {
        self.p_turn
    }

    pub fn can_play(&self, from: Square) -> bool {
        if self.state[from as usize] == Color::Empty { true } else { false }
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
}

// functions
//
fn square_make(fl: i32, rk: i32) -> Square {
    rk * (FILE_SIZE + 1) + fl
}

fn side_opp(sd: Side) -> Side {

    let mut out: Side; 

    match sd {
        Side::White => out = Side::Black,
        Side::Black => out = Side::White,
        Side::Empty => panic!(""),
        Side::Border => panic!(""),
    }

    out
}

fn pos_is_winner_scan(pos : &Pos) -> bool {

   let current_side = side_opp(pos.p_turn);

   check_pattern5(&pos, current_side)
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

    let mut out: bool = false;

    if found == true && !pos_is_winner_scan(pos) { out = true; }

    out
}

fn pos_is_draw_avx512f(pos : &Pos) -> bool {

    let mut test: bool = false;

    if pos.bitboard[Color::Empty as usize][0][0] == 0 &&
       pos.bitboard[Color::Empty as usize][0][1] == 0 &&
       pos.bitboard[Color::Empty as usize][0][2] == 0 &&
       pos.bitboard[Color::Empty as usize][0][3] == 0 &&
       pos.bitboard[Color::Empty as usize][0][4] == 0 &&
       pos.bitboard[Color::Empty as usize][0][5] == 0 &&
       pos.bitboard[Color::Empty as usize][0][6] == 0 &&
       pos.bitboard[Color::Empty as usize][0][7] == 0 &&
       pos.bitboard[Color::Empty as usize][0][8] == 0 &&
       pos.bitboard[Color::Empty as usize][0][9] == 0 &&
       pos.bitboard[Color::Empty as usize][0][10] == 0 &&
       pos.bitboard[Color::Empty as usize][0][11] == 0 &&
       pos.bitboard[Color::Empty as usize][0][12] == 0 &&
       pos.bitboard[Color::Empty as usize][0][13] == 0 &&
       pos.bitboard[Color::Empty as usize][0][14] == 0 &&
       pos.bitboard[Color::Empty as usize][0][15] == 0 { test = true; } else { test = false; }

    let mut out: bool = false;

    if test && unsafe {!pos_is_winner_avx512(pos)} { out = true; }

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

fn gen_moves(list : &mut List, pos: &Pos) {

    list.clear();

    for rk in 0..RANK_SIZE {
        for fl in 0..FILE_SIZE {
            let sq : Square = square_make(fl, rk);
            if pos.can_play(sq) { list.add(sq); }
        }
    }
}

fn search_real(pos: &Pos, alpha: i32, beta: i32, depth: i32, ply: i32) -> i32 {

    assert!(-EVAL_INF <= alpha && alpha < beta && beta <= EVAL_INF);
    // leaf?

//    unsafe { if pos_is_winner_avx512(&pos) != pos_is_winner_scan(&pos) { println!("avx512 wrong!!!!!!!!!!!!!!!!!!!!!!!!!! avx512 = {}", pos_is_winner_avx512(&pos));  pos_disp(&pos); panic!();} }
    //if pos_is_winner_scan(&pos) { return -EVAL_INF + ply }
    if unsafe { pos_is_winner_avx512(&pos) } { return -EVAL_INF + ply }

    unsafe { if pos_is_draw_avx512f(&pos) != pos_is_draw(&pos) { println!("avx512 wrong!!!!!!!!!!!!!!!!!!!!!!!!!! is_draw");  pos_disp(&pos); panic!();} }
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
            bitboard: pos.bitboard,
        };

        new_pos.do_move(mv);

        let sc: i32 = -search_real(&new_pos, -beta, -cmp::max(alpha, bs), depth - 1, ply + 1);

        if sc > bs { bm = mv; bs = sc; }
        }
    }

    assert!(bm != MOVE_NONE);
    assert!(bs >= -EVAL_INF && bs <= EVAL_INF);

    if ply == 0 { bm } else { bs } //best move at the root node, best score elsewhere
    //bs
}

fn eval(pos: &Pos, ply: i32) -> i32 {

    let atk: Side = pos.turn();
    let def: Side = side_opp(atk);

    let check_live4: Side = def; 
    let check_live4_opp: Side = atk; 


    if unsafe { check_pattern4_once_avx512(&pos, check_live4) } != check_patternlive4(&pos, check_live4) { println!("avx512 wrong!!!!!!!!!!!!!!!!!!!!!!!!!! file4_once !!!!!! self ");  pos_disp(&pos); panic!(); }
    if unsafe { check_pattern4_once_avx512(&pos, check_live4_opp) } != check_patternlive4(&pos, check_live4_opp) { println!("avx512 wrong!!!!!!!!!!!!!!!!!!!!!!!!!! file4_once !!!!!! opp ");  pos_disp(&pos); panic!(); }

    unsafe { 
             let result = check_pattern4_dead_avx512(&pos, check_live4_opp); 
              if result != check_patterndead4(&pos, check_live4_opp) { println!("avx512 wrong!!!!!!!!!!!!!!!!!!!!!!!!!! file4_dead !!!!!! opp ");  pos_disp(&pos); panic!();}
    } 

     unsafe { 
              let result = check_pattern4_dead_avx512(&pos, check_live4); 
              let c4: i32 = check_patterndead4(&pos, check_live4);
              if c4 != result { println!("avx512 wrong!!!!!!!!!!!!!!!!!!!!!!!!!! file4_dead_count !!!!!! opp org = {}, new = {}", c4, result);  pos_disp(&pos); panic!();}
    }

    unsafe { 
              let result = check_pattern3_live_avx512(&pos, check_live4); 
              let count1: i32 = check_patternlive3(&pos, check_live4);

              if result != count1 { println!("avx512 wrong!!!!!!!!!!!!!!!!!!!!!!!!!! live3_dead !!!!!! self org = {}, new = {}", count1, result);  pos_disp(&pos); panic!(); }
    } 

    //if check_patternlive4(&pos, check_live4) { return -4096 }
    unsafe { if check_pattern4_once_avx512(&pos, check_live4) { return -4096 } }
    
    //if check_patternlive4(&pos, check_live4_opp) { return 2560 }
    unsafe { if check_pattern4_once_avx512(&pos, check_live4_opp) { return 2560 } }

    //if check_patterndead4(&pos, check_live4_opp) > 0 { return 2560 }
    unsafe { if check_pattern4_dead_avx512(&pos, check_live4_opp) > 0 { return 2560 } }

    // 4,3
    //let n_c4: i32 = check_patterndead4(&pos, check_live4);
    //let n_c3: i32 = check_patternlive3(&pos, check_live4);
    let mut n_c4: i32 = 0;
    unsafe { n_c4 = check_pattern4_dead_avx512(&pos, check_live4); }
    let mut n_c3: i32 = 0;
    unsafe { n_c3 = check_pattern3_live_avx512(&pos, check_live4); }

    if n_c4 > 1 { return -2048 }
    if n_c4 == 1 && n_c3 > 0 { return -3048 }

    //---------------------------------------------------------------------------
    
    //if check_patternlive3(&pos, check_live4_opp) > 1 { return 2560 }
    if unsafe { check_pattern3_live_avx512(&pos, check_live4_opp) > 1 } { return 2560 }

    if n_c3 > 1 { return -2048 }

    0 
}

fn check_pattern5(pos: &Pos, sd: Side) -> bool {

    let mut n: i32 = 0;

    for rk in 0..RANK_SIZE {
        for fl in 0..FILE_SIZE {
            let sq : Square = square_make(fl, rk);

            for pat in 0..4 {
                let idx0 = sq;
                let idx1 = sq + PATTERN[pat][0];
                let idx2 = sq + PATTERN[pat][1];
                let idx3 = sq + PATTERN[pat][2];
                let idx4 = sq + PATTERN[pat][3];

                let val0 = pos.state[idx0 as usize];
                let val1 = pos.state[idx1 as usize];
                let val2 = pos.state[idx2 as usize];
                let val3 = pos.state[idx3 as usize];
                let val4 = pos.state[idx4 as usize];

                if val0 == sd && val1 == sd && val2 == sd && val3 == sd && val4 == sd { n += 1; }
            }
        }
    }

    if n > 0 { true } else { false }

    //false
}

fn check_patternlive4(pos: &Pos, sd: Side) -> bool {

    let mut n: i32 = 0;

    for rk in 0..RANK_SIZE {
        for fl in 0..FILE_SIZE {
            let sq : Square = square_make(fl, rk);

            for pat in 0..4 {
                let idx0 = sq;
                let idx1 = sq + PATTERN[pat][0];
                let idx2 = sq + PATTERN[pat][1];
                let idx3 = sq + PATTERN[pat][2];
                let idx4 = sq + PATTERN[pat][3];
                let idx5 = sq + PATTERN[pat][4];

                let val0 = pos.state[idx0 as usize];
                let val1 = pos.state[idx1 as usize];
                let val2 = pos.state[idx2 as usize];
                let val3 = pos.state[idx3 as usize];
                let val4 = pos.state[idx4 as usize];
                let val5 = pos.state[idx5 as usize];

                if val0 == Color::Empty && val1 == sd && val2 == sd && val3 == sd && val4 == sd && val5 == Color::Empty { n += 1; }
            }
        } 
    } 

    if n > 0 { true } else { false }
}

fn check_patterndead4(pos: &Pos, sd: Side) -> i32 {

    let mut n: i32 = 0;

    for rk in 0..RANK_SIZE {
        for fl in 0..FILE_SIZE {
            let sq : Square = square_make(fl, rk);

            for pat in 0..4 {
                let idx0 = sq;
                let idx1 = sq + PATTERN[pat][0];
                let idx2 = sq + PATTERN[pat][1];
                let idx3 = sq + PATTERN[pat][2];
                let idx4 = sq + PATTERN[pat][3];

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
    } 

    n 
}

fn check_patternlive3(pos: &Pos, sd: Side) -> i32 {

    let mut n: i32 = 0;

    for rk in 0..RANK_SIZE {
        for fl in 0..FILE_SIZE {
            let sq : Square = square_make(fl, rk);

            for pat in 0..4 {
                let idx0 = sq;
                let idx1 = sq + PATTERN[pat][0];
                let idx2 = sq + PATTERN[pat][1];
                let idx3 = sq + PATTERN[pat][2];
                let idx4 = sq + PATTERN[pat][3];
                let idx5 = sq + PATTERN[pat][4];

                let val0 = pos.state[idx0 as usize];
                let val1 = pos.state[idx1 as usize];
                let val2 = pos.state[idx2 as usize];
                let val3 = pos.state[idx3 as usize];
                let val4 = pos.state[idx4 as usize];
                let val5 = pos.state[idx5 as usize];

                if val0 == Color::Empty && val1 == sd && val2 == sd && val3 == sd && val4 == Color::Empty { n +=1 ; }
                if val0 == Color::Empty && val1 == sd && val2 == sd && val3 == Color::Empty && val4 == sd && val5 == Color::Empty { n += 1; }
                if val0 == Color::Empty && val1 == sd && val2 == Color::Empty && val3 == sd && val4 == sd && val5 == Color::Empty { n += 1; }
            }
        }  
    } 

    n
}

#[target_feature(enable = "avx512f")]
unsafe fn pos_is_winner_avx512(pos : &Pos) -> bool {

    let current_side = side_opp(pos.p_turn);
    let coloridx = current_side as usize;

    let board0org: [__m512i; 2]  = [_mm512_set_epi32(pos.bitboard[coloridx][0][15], pos.bitboard[coloridx][0][14], pos.bitboard[coloridx][0][13], pos.bitboard[coloridx][0][12], pos.bitboard[coloridx][0][11], pos.bitboard[coloridx][0][10], pos.bitboard[coloridx][0][9], pos.bitboard[coloridx][0][8], pos.bitboard[coloridx][0][7], pos.bitboard[coloridx][0][6], pos.bitboard[coloridx][0][5], pos.bitboard[coloridx][0][4], pos.bitboard[coloridx][0][3], pos.bitboard[coloridx][0][2], pos.bitboard[coloridx][0][1], pos.bitboard[coloridx][0][0]),
_mm512_set_epi32(pos.bitboard[coloridx][1][15], pos.bitboard[coloridx][1][14], pos.bitboard[coloridx][1][13], pos.bitboard[coloridx][1][12], pos.bitboard[coloridx][1][11], pos.bitboard[coloridx][1][10], pos.bitboard[coloridx][1][9], pos.bitboard[coloridx][1][8], pos.bitboard[coloridx][1][7], pos.bitboard[coloridx][1][6], pos.bitboard[coloridx][1][5], pos.bitboard[coloridx][1][4], pos.bitboard[coloridx][1][3], pos.bitboard[coloridx][1][2], pos.bitboard[coloridx][1][1], pos.bitboard[coloridx][1][0])];

    let answer = _mm512_set1_epi32( (1<<31)|(1<<30)|(1<<29)|(1<<28)|(1<<27) ); // an unbroken chain of five moves

    let answer_mask: [__mmask16; 11] = [0b11111111_11111111,
                                        0b11111111_11111111,
                                        0b11111111_11111101,
                                        0b11111111_11111001,
                                        0b11111111_11110001,
                                        0b11111111_11100001,
                                        0b11111111_11000011,
                                        0b11111111_10000111,
                                        0b11111111_10001111,
                                        0b11111111_10011111,
                                        0b11111111_10111111,];

    let mut temp_mask: [[__mmask16; (11+11)]; 2] = [[0; (11+11)]; 2]; //

    for dir in 0..2 {
        let mut board0 = board0org[dir];

        let boardf = _mm512_and_epi32(answer, board0);
        temp_mask[dir][0] = _mm512_mask_cmp_epi32_mask(0b01111111_11111111, answer, boardf, 0);

        for i in 1..11 {
            board0 = _mm512_rol_epi32(board0, 1); // rotate one space left
            let boardf = _mm512_and_epi32(answer, board0); // focus on the pattern
            temp_mask[dir][i as usize] = _mm512_mask_cmp_epi32_mask(0b01111111_11111111, answer, boardf, 0); // see if it matches the pattern 
        }

        board0 = _mm512_rol_epi32(board0, 6); //whatever 11,12,13,14 are occupid, it cannot match an unbroken chain of five moves. Therefore, shift to another pattern.

        let boardf = _mm512_and_epi32(answer, board0);
        temp_mask[dir][11] = _mm512_mask_cmp_epi32_mask(answer_mask[0], answer, boardf, 0);

        for i in 12..22 {
            let idx: i32 = i - 11;
            board0 = _mm512_rol_epi32(board0, 1);
            let boardf = _mm512_and_epi32(answer, board0);
            temp_mask[dir][i as usize] = _mm512_mask_cmp_epi32_mask(answer_mask[idx as usize], answer, boardf, 0);
        }
    }

    let mut n: i32 = 0;

    // calculate how many patterns matched
    for i in 0..2 {
        for j in 0..(11+11) {
            n += _popcnt32(temp_mask[i][j] as i32);
        }
    }

    if n > 0 { return true } else { return false }
}

#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn pos_is_winner_avx512bw(pos : &Pos) -> bool {

    let current_side = side_opp(pos.p_turn);
    let coloridx = current_side as usize;

    let board0org: [__m512i; 2] = [
        _mm512_loadu_epi32(&pos.bitboard[coloridx][0][0]),
        _mm512_loadu_epi32(&pos.bitboard[coloridx][1][0]),
    ]; // load states from bitboard

    let answer = _mm512_set1_epi8( (1<<7)|(1<<6)|(1<<5)|(1<<4)|(1<<3) ); // an unbroken chain of five moves

    let answer_mask: [__mmask64; 8] = [0b0111_0111_0111_1111_1111_1111_1111_1111_0111_0111_0111_0111_1111_1111_1111_0111,
                                       0b0111_0111_0111_0111_1111_1111_1111_0111_0111_0111_0111_1111_1111_1111_1111_0011,
                                       0b0111_0111_0111_0111_0111_1111_0111_0111_0111_0111_1111_1111_1111_1111_1011_0011,
                                       0b0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0001_0001_0001,
                                       0b0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0001_0001_0001_0001,
                                       0b0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0101_0001_0001_0001_0001_0001,
                                       0b0001_0101_0101_0101_0101_0101_0101_0101_0101_0101_0001_0001_0001_0001_0001_0001,
                                       0b0001_0001_0101_0101_0101_0101_0101_0101_0101_0001_0001_0001_0001_0001_0001_0001];

    let mut temp_mask: __mmask64 = 0;
    let mut count_match: i64 = 0;

    for dir in 0..2 {
        let mut board0 = board0org[dir];

        let boardf = _mm512_and_epi32(answer, board0);
        temp_mask = _mm512_mask_cmpeq_epi8_mask(answer_mask[0], answer, boardf);

        count_match += _mm_popcnt_u64(temp_mask);

        for i in 1..8 {
            board0 = _mm512_rol_epi32(board0, 1); // rotate one space left
            let boardf = _mm512_and_epi32(answer, board0); // focus on the pattern
            temp_mask = _mm512_mask_cmpeq_epi8_mask(answer_mask[i], answer, boardf);
            count_match += _mm_popcnt_u64(temp_mask);
        }
    }

    if count_match > 0 { return true } else { return false }
}

#[target_feature(enable = "avx512f")]
unsafe fn check_pattern4_once_avx512(pos : &Pos, sd: Side) -> bool {

    let coloridx = sd as usize;
    let emptyidx = Color::Empty as usize;

    let answer_color = _mm512_set1_epi32(         (1<<30)|(1<<29)|(1<<28)|(1<<27)         );
    let answer_empty = _mm512_set1_epi32( (1<<31)|                                (1<<26) );
    let answer       = _mm512_set1_epi32( (1<<31)|(1<<30)|(1<<29)|(1<<28)|(1<<27)|(1<<26) );

    let answer_mask: [__mmask16; 10] = [0b11111111_11111110, 
                                        0b11111111_11111100,
                                        0b11111111_11111000,
                                        0b11111111_11110000,
                                        0b11111111_11100000,
                                        0b01111111_11000000,
                                        0b00111111_10000010, //Row xxx
                                        0b00011111_00000110,
                                        0b00001110_00001110,
                                        0b00000100_00011110,];

    let board0org: [__m512i; 2]  = [_mm512_set_epi32(pos.bitboard[coloridx][0][15], pos.bitboard[coloridx][0][14], pos.bitboard[coloridx][0][13], pos.bitboard[coloridx][0][12], pos.bitboard[coloridx][0][11], pos.bitboard[coloridx][0][10], pos.bitboard[coloridx][0][9], pos.bitboard[coloridx][0][8], pos.bitboard[coloridx][0][7], pos.bitboard[coloridx][0][6], pos.bitboard[coloridx][0][5], pos.bitboard[coloridx][0][4], pos.bitboard[coloridx][0][3], pos.bitboard[coloridx][0][2], pos.bitboard[coloridx][0][1], pos.bitboard[coloridx][0][0]),
_mm512_set_epi32(pos.bitboard[coloridx][1][15], pos.bitboard[coloridx][1][14], pos.bitboard[coloridx][1][13], pos.bitboard[coloridx][1][12], pos.bitboard[coloridx][1][11], pos.bitboard[coloridx][1][10], pos.bitboard[coloridx][1][9], pos.bitboard[coloridx][1][8], pos.bitboard[coloridx][1][7], pos.bitboard[coloridx][1][6], pos.bitboard[coloridx][1][5], pos.bitboard[coloridx][1][4], pos.bitboard[coloridx][1][3], pos.bitboard[coloridx][1][2], pos.bitboard[coloridx][1][1], pos.bitboard[coloridx][1][0])];

    let board1org: [__m512i; 2] = [_mm512_set_epi32(pos.bitboard[emptyidx][0][15], pos.bitboard[emptyidx][0][14], pos.bitboard[emptyidx][0][13], pos.bitboard[emptyidx][0][12], pos.bitboard[emptyidx][0][11], pos.bitboard[emptyidx][0][10], pos.bitboard[emptyidx][0][9], pos.bitboard[emptyidx][0][8], pos.bitboard[emptyidx][0][7], pos.bitboard[emptyidx][0][6], pos.bitboard[emptyidx][0][5], pos.bitboard[emptyidx][0][4], pos.bitboard[emptyidx][0][3], pos.bitboard[emptyidx][0][2], pos.bitboard[emptyidx][0][1], pos.bitboard[emptyidx][0][0]),
    _mm512_set_epi32(pos.bitboard[emptyidx][1][15], pos.bitboard[emptyidx][1][14], pos.bitboard[emptyidx][1][13], pos.bitboard[emptyidx][1][12], pos.bitboard[emptyidx][1][11], pos.bitboard[emptyidx][1][10], pos.bitboard[emptyidx][1][9], pos.bitboard[emptyidx][1][8], pos.bitboard[emptyidx][1][7], pos.bitboard[emptyidx][1][6], pos.bitboard[emptyidx][1][5], pos.bitboard[emptyidx][1][4], pos.bitboard[emptyidx][1][3], pos.bitboard[emptyidx][1][2], pos.bitboard[emptyidx][1][1], pos.bitboard[emptyidx][1][0])];

    let mut temp_mask: [[__mmask16; 10+10]; 2] = [[0; 10+10]; 2];

    for dir in 0..2 {

        let mut board0 = board0org[dir];
        let mut board1 = board1org[dir];

        let boardf1 = _mm512_and_epi32(answer_color, board0);// check sd
        let boardf2 = _mm512_and_epi32(answer_empty, board1);// check empty
        let boardf  = _mm512_or_epi32(boardf1, boardf2);

        temp_mask[dir][0] = _mm512_mask_cmp_epi32_mask(0b01111111_11111111, answer, boardf, 0);

        for i in 1..10 {
            board0 = _mm512_rol_epi32(board0, 1);//rot sd
            board1 = _mm512_rol_epi32(board1, 1);//rot empty

            let boardf1 = _mm512_and_epi32(answer_color, board0);
            let boardf2 = _mm512_and_epi32(answer_empty, board1);
            let boardf  = _mm512_or_epi32(boardf1, boardf2);

            temp_mask[dir][i as usize] = _mm512_mask_cmp_epi32_mask(0b01111111_11111111, answer, boardf, 0);
        }

        board0 = _mm512_rol_epi32(board0, 7);//rot sd
        board1 = _mm512_rol_epi32(board1, 7);//rot empty
    
        let boardf1 = _mm512_and_epi32(answer_color, board0);// check sd
        let boardf2 = _mm512_and_epi32(answer_empty, board1);// check empty
        let boardf  = _mm512_or_epi32(boardf1, boardf2);

        temp_mask[dir][10] = _mm512_mask_cmp_epi32_mask(answer_mask[0], answer, boardf, 0);

        for i in 11..20 {

            let idx: i32 = i - 10;

            board0 = _mm512_rol_epi32(board0, 1);//rot sd
            board1 = _mm512_rol_epi32(board1, 1);//rot empty

            let boardf1 = _mm512_and_epi32(answer_color, board0);
            let boardf2 = _mm512_and_epi32(answer_empty, board1);
            let boardf  = _mm512_or_epi32(boardf1, boardf2);

            temp_mask[dir][i as usize] = _mm512_mask_cmp_epi32_mask(answer_mask[idx as usize], answer, boardf, 0);
        }
    }

    let mut n: i32 = 0;

    for i in 0..2 {
        for j in 0..20 {
            n += _popcnt32(temp_mask[i][j] as i32);
        }
    }

    if n > 0 { return true } else { return false }
}

#[target_feature(enable = "avx512f")]
unsafe fn check_pattern4_dead_avx512(pos : &Pos, sd: Side) -> i32 {

    let coloridx = sd as usize;
    let emptyidx = Color::Empty as usize;

    let answer_color: [__m512i; 5] = [_mm512_set1_epi32(         (1<<30)|(1<<29)|(1<<28)|(1<<27) ),
                                      _mm512_set1_epi32( (1<<31)|        (1<<29)|(1<<28)|(1<<27) ),
                                      _mm512_set1_epi32( (1<<31)|(1<<30)        |(1<<28)|(1<<27) ),
                                      _mm512_set1_epi32( (1<<31)|(1<<30)|(1<<29)        |(1<<27) ),
                                      _mm512_set1_epi32( (1<<31)|(1<<30)|(1<<29)|(1<<28)         )];

    let answer_empty: [__m512i; 5]= [_mm512_set1_epi32( (1<<31) ),
                                     _mm512_set1_epi32(          (1<<30) ),
                                     _mm512_set1_epi32(                  (1<<29) ),
                                     _mm512_set1_epi32(                          (1<<28) ),
                                     _mm512_set1_epi32(                                   (1<<27))];

    let answer       = _mm512_set1_epi32( (1<<31)|(1<<30)|(1<<29)|(1<<28)|(1<<27));

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

    let board0org: [__m512i; 2]  = [_mm512_set_epi32(pos.bitboard[coloridx][0][15], pos.bitboard[coloridx][0][14], pos.bitboard[coloridx][0][13], pos.bitboard[coloridx][0][12], pos.bitboard[coloridx][0][11], pos.bitboard[coloridx][0][10], pos.bitboard[coloridx][0][9], pos.bitboard[coloridx][0][8], pos.bitboard[coloridx][0][7], pos.bitboard[coloridx][0][6], pos.bitboard[coloridx][0][5], pos.bitboard[coloridx][0][4], pos.bitboard[coloridx][0][3], pos.bitboard[coloridx][0][2], pos.bitboard[coloridx][0][1], pos.bitboard[coloridx][0][0]),
_mm512_set_epi32(pos.bitboard[coloridx][1][15], pos.bitboard[coloridx][1][14], pos.bitboard[coloridx][1][13], pos.bitboard[coloridx][1][12], pos.bitboard[coloridx][1][11], pos.bitboard[coloridx][1][10], pos.bitboard[coloridx][1][9], pos.bitboard[coloridx][1][8], pos.bitboard[coloridx][1][7], pos.bitboard[coloridx][1][6], pos.bitboard[coloridx][1][5], pos.bitboard[coloridx][1][4], pos.bitboard[coloridx][1][3], pos.bitboard[coloridx][1][2], pos.bitboard[coloridx][1][1], pos.bitboard[coloridx][1][0])];
        //let mut board0 = _mm512_load_epi32(&pos.bitboard[coloridx][dir][0]);

    let board1org: [__m512i; 2] = [_mm512_set_epi32(pos.bitboard[emptyidx][0][15], pos.bitboard[emptyidx][0][14], pos.bitboard[emptyidx][0][13], pos.bitboard[emptyidx][0][12], pos.bitboard[emptyidx][0][11], pos.bitboard[emptyidx][0][10], pos.bitboard[emptyidx][0][9], pos.bitboard[emptyidx][0][8], pos.bitboard[emptyidx][0][7], pos.bitboard[emptyidx][0][6], pos.bitboard[emptyidx][0][5], pos.bitboard[emptyidx][0][4], pos.bitboard[emptyidx][0][3], pos.bitboard[emptyidx][0][2], pos.bitboard[emptyidx][0][1], pos.bitboard[emptyidx][0][0]),
    _mm512_set_epi32(pos.bitboard[emptyidx][1][15], pos.bitboard[emptyidx][1][14], pos.bitboard[emptyidx][1][13], pos.bitboard[emptyidx][1][12], pos.bitboard[emptyidx][1][11], pos.bitboard[emptyidx][1][10], pos.bitboard[emptyidx][1][9], pos.bitboard[emptyidx][1][8], pos.bitboard[emptyidx][1][7], pos.bitboard[emptyidx][1][6], pos.bitboard[emptyidx][1][5], pos.bitboard[emptyidx][1][4], pos.bitboard[emptyidx][1][3], pos.bitboard[emptyidx][1][2], pos.bitboard[emptyidx][1][1], pos.bitboard[emptyidx][1][0])];
        //let mut board1 = _mm512_load_epi32(&pos.bitboard[emptyidx][dir][0]);

    let mut temp_mask: [[[__mmask16; 11+11]; 2]; 5] = [[[0; 11+11]; 2]; 5];

    for pattern in 0..5 {

        for dir in 0..2 {

            let mut board0 = board0org[dir];
            let mut board1 = board1org[dir]; 

           let boardf1 = _mm512_and_epi32(answer_color[pattern], board0);// check sd
           let boardf2 = _mm512_and_epi32(answer_empty[pattern], board1);// check empty
           let boardf  = _mm512_or_epi32(boardf1, boardf2);

           temp_mask[pattern][dir][0] = _mm512_mask_cmp_epi32_mask(0b01111111_11111111, answer, boardf, 0);

           for i in 1..11 {
               board0 = _mm512_rol_epi32(board0, 1);//rot sd
               board1 = _mm512_rol_epi32(board1, 1);//rot empty

               let boardf1 = _mm512_and_epi32(answer_color[pattern], board0);
               let boardf2 = _mm512_and_epi32(answer_empty[pattern], board1);
               let boardf  = _mm512_or_epi32(boardf1, boardf2);

               temp_mask[pattern][dir][i as usize] = _mm512_mask_cmp_epi32_mask(0b01111111_11111111, answer, boardf, 0);
           }

           board0 = _mm512_rol_epi32(board0, 6);//rot sd
           board1 = _mm512_rol_epi32(board1, 6);//rot empty

           let boardf1 = _mm512_and_epi32(answer_color[pattern], board0);// check sd
           let boardf2 = _mm512_and_epi32(answer_empty[pattern], board1);// check empty
           let boardf  = _mm512_or_epi32(boardf1, boardf2);

           temp_mask[pattern][dir][11] = _mm512_mask_cmp_epi32_mask(answer_mask[0], answer, boardf, 0);

           for i in 12..22 {

               let idx: i32 = i - 11;

               board0 = _mm512_rol_epi32(board0, 1);//rot sd
               board1 = _mm512_rol_epi32(board1, 1);//rot empty

               let boardf1 = _mm512_and_epi32(answer_color[pattern], board0);
               let boardf2 = _mm512_and_epi32(answer_empty[pattern], board1);
               let boardf  = _mm512_or_epi32(boardf1, boardf2);

               temp_mask[pattern][dir][i as usize] = _mm512_mask_cmp_epi32_mask(answer_mask[idx as usize], answer, boardf, 0);
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
unsafe fn check_pattern3_live_avx512(pos : &Pos, sd: Side) -> i32 {

    let coloridx = sd as usize;
    let emptyidx = Color::Empty as usize;

    let board0org: [__m512i; 2]  = [_mm512_set_epi32(pos.bitboard[coloridx][0][15], pos.bitboard[coloridx][0][14], pos.bitboard[coloridx][0][13], pos.bitboard[coloridx][0][12], pos.bitboard[coloridx][0][11], pos.bitboard[coloridx][0][10], pos.bitboard[coloridx][0][9], pos.bitboard[coloridx][0][8], pos.bitboard[coloridx][0][7], pos.bitboard[coloridx][0][6], pos.bitboard[coloridx][0][5], pos.bitboard[coloridx][0][4], pos.bitboard[coloridx][0][3], pos.bitboard[coloridx][0][2], pos.bitboard[coloridx][0][1], pos.bitboard[coloridx][0][0]),
_mm512_set_epi32(pos.bitboard[coloridx][1][15], pos.bitboard[coloridx][1][14], pos.bitboard[coloridx][1][13], pos.bitboard[coloridx][1][12], pos.bitboard[coloridx][1][11], pos.bitboard[coloridx][1][10], pos.bitboard[coloridx][1][9], pos.bitboard[coloridx][1][8], pos.bitboard[coloridx][1][7], pos.bitboard[coloridx][1][6], pos.bitboard[coloridx][1][5], pos.bitboard[coloridx][1][4], pos.bitboard[coloridx][1][3], pos.bitboard[coloridx][1][2], pos.bitboard[coloridx][1][1], pos.bitboard[coloridx][1][0])];

    let board1org: [__m512i; 2] = [_mm512_set_epi32(pos.bitboard[emptyidx][0][15], pos.bitboard[emptyidx][0][14], pos.bitboard[emptyidx][0][13], pos.bitboard[emptyidx][0][12], pos.bitboard[emptyidx][0][11], pos.bitboard[emptyidx][0][10], pos.bitboard[emptyidx][0][9], pos.bitboard[emptyidx][0][8], pos.bitboard[emptyidx][0][7], pos.bitboard[emptyidx][0][6], pos.bitboard[emptyidx][0][5], pos.bitboard[emptyidx][0][4], pos.bitboard[emptyidx][0][3], pos.bitboard[emptyidx][0][2], pos.bitboard[emptyidx][0][1], pos.bitboard[emptyidx][0][0]),
    _mm512_set_epi32(pos.bitboard[emptyidx][1][15], pos.bitboard[emptyidx][1][14], pos.bitboard[emptyidx][1][13], pos.bitboard[emptyidx][1][12], pos.bitboard[emptyidx][1][11], pos.bitboard[emptyidx][1][10], pos.bitboard[emptyidx][1][9], pos.bitboard[emptyidx][1][8], pos.bitboard[emptyidx][1][7], pos.bitboard[emptyidx][1][6], pos.bitboard[emptyidx][1][5], pos.bitboard[emptyidx][1][4], pos.bitboard[emptyidx][1][3], pos.bitboard[emptyidx][1][2], pos.bitboard[emptyidx][1][1], pos.bitboard[emptyidx][1][0])];

    let answer_color: [__m512i; 1] = [_mm512_set1_epi32(         (1<<30)|(1<<29)|(1<<28)         )];
    let answer_empty: [__m512i; 1] = [_mm512_set1_epi32( (1<<31)|                         (1<<27) )];
    let answer: __m512i = _mm512_set1_epi32( (1<<31)|(1<<30)|(1<<29)|(1<<28)|(1<<27) );

    let mut temp_mask: [[[__mmask16; 11+11]; 2]; 1] = [[[0; 11+11]; 2]; 1];

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

    for pattern in 0..1 {

        for dir in 0..2 {

            let mut board0 = board0org[dir];
            let mut board1 = board1org[dir];

            let boardf1 = _mm512_and_epi32(answer_color[pattern], board0);// check sd
            let boardf2 = _mm512_and_epi32(answer_empty[pattern], board1);// check empty
            let boardf  = _mm512_or_epi32(boardf1, boardf2);

            temp_mask[pattern][dir][0] = _mm512_mask_cmp_epi32_mask(0b01111111_11111111, answer, boardf, 0);

            for i in 1..11 {

                board0 = _mm512_rol_epi32(board0, 1);//rot sd
                board1 = _mm512_rol_epi32(board1, 1);//rot empty

                let boardf1 = _mm512_and_epi32(answer_color[pattern], board0);
                let boardf2 = _mm512_and_epi32(answer_empty[pattern], board1);
                let boardf  = _mm512_or_epi32(boardf1, boardf2);

                temp_mask[pattern][dir][i as usize] = _mm512_mask_cmp_epi32_mask(0b01111111_11111111, answer, boardf, 0);
            }

            board0 = _mm512_rol_epi32(board0, 6);//rot sd
            board1 = _mm512_rol_epi32(board1, 6);//rot empty
          
            let boardf1 = _mm512_and_epi32(answer_color[pattern], board0);// check sd
            let boardf2 = _mm512_and_epi32(answer_empty[pattern], board1);// check empty
            let boardf  = _mm512_or_epi32(boardf1, boardf2);

            temp_mask[pattern][dir][11] = _mm512_mask_cmp_epi32_mask(answer_mask[0], answer, boardf, 0);

            for i in 12..22 {

                let idx: i32 = i - 11;

                board0 = _mm512_rol_epi32(board0, 1);//rot sd
                board1 = _mm512_rol_epi32(board1, 1);//rot empty

                let boardf1 = _mm512_and_epi32(answer_color[pattern], board0);
                let boardf2 = _mm512_and_epi32(answer_empty[pattern], board1);
                let boardf  = _mm512_or_epi32(boardf1, boardf2);

                temp_mask[pattern][dir][i as usize] = _mm512_mask_cmp_epi32_mask(answer_mask[idx as usize], answer, boardf, 0);
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

    let answer_color: [__m512i; 2] = [_mm512_set1_epi32(          (1<<30)|        (1<<28)|(1<<27) ),
                                      _mm512_set1_epi32(          (1<<30)|(1<<29)        |(1<<27) )];

    let answer_empty: [__m512i; 2] = [_mm512_set1_epi32( (1<<31)|         (1<<29)|                (1<<26) ),
                                      _mm512_set1_epi32( (1<<31)|                 (1<<28)|        (1<<26) )];

    let answer: __m512i = _mm512_set1_epi32( (1<<31)|(1<<30)|(1<<29)|(1<<28)|(1<<27)|(1<<26) );

    let mut temp_mask: [[[__mmask16; 10+10]; 2]; 2] = [[[0; 10+10]; 2]; 2];

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

    for pattern in 0..2 {

        for dir in 0..2 {

            let mut board0 = board0org[dir];
            let mut board1 = board1org[dir];

            let boardf1 = _mm512_and_epi32(answer_color[pattern], board0);// check sd
            let boardf2 = _mm512_and_epi32(answer_empty[pattern], board1);// check empty
            let boardf  = _mm512_or_epi32(boardf1, boardf2);

            temp_mask[pattern][dir][0] = _mm512_mask_cmp_epi32_mask(0b01111111_11111111, answer, boardf, 0);

            for i in 1..10 {

                board0 = _mm512_rol_epi32(board0, 1);//rot sd
                board1 = _mm512_rol_epi32(board1, 1);//rot empty

                let boardf1 = _mm512_and_epi32(answer_color[pattern], board0);
                let boardf2 = _mm512_and_epi32(answer_empty[pattern], board1);
                let boardf  = _mm512_or_epi32(boardf1, boardf2);

                temp_mask[pattern][dir][i as usize] = _mm512_mask_cmp_epi32_mask(0b01111111_11111111, answer, boardf, 0);
                }

           board0 = _mm512_rol_epi32(board0, 7);//rot sd
           board1 = _mm512_rol_epi32(board1, 7);//rot empty
          
           let boardf1 = _mm512_and_epi32(answer_color[pattern], board0);// check sd
           let boardf2 = _mm512_and_epi32(answer_empty[pattern], board1);// check empty
           let boardf  = _mm512_or_epi32(boardf1, boardf2);

           temp_mask[pattern][dir][10] = _mm512_mask_cmp_epi32_mask(answer_mask[0], answer, boardf, 0);

           for i in 11..20 {

               let idx: i32 = i - 10;

               board0 = _mm512_rol_epi32(board0, 1);//rot sd
               board1 = _mm512_rol_epi32(board1, 1);//rot empty

               let boardf1 = _mm512_and_epi32(answer_color[pattern], board0);
               let boardf2 = _mm512_and_epi32(answer_empty[pattern], board1);
               let boardf  = _mm512_or_epi32(boardf1, boardf2);

               temp_mask[pattern][dir][i as usize] = _mm512_mask_cmp_epi32_mask(answer_mask[idx as usize], answer, boardf, 0);
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

    loop
    {

    let start = Instant::now();

    println!("Hello, this is connect 6!");

    let test_state: [Color; SQUARE_SIZE as usize] = [Color::Empty; SQUARE_SIZE as usize];
    let test_bitboard: [[[i32; 16]; 2]; 3] = [[[0; 16]; 2]; 3];

    let mut test1 = Pos {
        state: test_state,
        p_turn: Color::Black,

        bitboard: test_bitboard,
    };

    test1.init();

    //pos_disp(&test1);

    for i in 0..(FILE_SIZE*RANK_SIZE) {

     //   println!("----------------------------------------\n\n\n\n");
      //  println!("MOVE {}!!!!\n\n\n\n", i);


    let mut next_move: Move = square_make(1,7);
    //if i > 0 {  next_move = search(&test1, 2, 4); }
    if i > 0 {  next_move = search_real(&test1, -EVAL_INF, EVAL_INF, 2, 0); }

    test1.do_move(next_move);
    pos_disp(&test1);

    if pos_is_end(&test1) { 
        println!("Game over!!!!!!");
        println!("MOVE {}!!!!\n", i);
        break; }
    }

    let duration = start.elapsed();

    println!("Time elapsed in expensive_function() is: {:?}", duration);
    }
}
