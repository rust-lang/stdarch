//! Utility macros.

macro_rules! constify_imm6 {
    ($imm8:expr, $expand:ident) => {
        #[allow(overflowing_literals)]
        match ($imm8) & 0b1_1111 {
            0 => $expand!(0),
            1 => $expand!(1),
            2 => $expand!(2),
            3 => $expand!(3),
            4 => $expand!(4),
            5 => $expand!(5),
            6 => $expand!(6),
            7 => $expand!(7),
            8 => $expand!(8),
            9 => $expand!(9),
            10 => $expand!(10),
            11 => $expand!(11),
            12 => $expand!(12),
            13 => $expand!(13),
            14 => $expand!(14),
            15 => $expand!(15),
            16 => $expand!(16),
            17 => $expand!(17),
            18 => $expand!(18),
            19 => $expand!(19),
            20 => $expand!(20),
            21 => $expand!(21),
            22 => $expand!(22),
            23 => $expand!(23),
            24 => $expand!(24),
            25 => $expand!(25),
            26 => $expand!(26),
            27 => $expand!(27),
            28 => $expand!(28),
            29 => $expand!(29),
            30 => $expand!(30),
            _ => $expand!(31),
        }
    };
}

macro_rules! constify_imm4 {
    ($imm8:expr, $expand:ident) => {
        #[allow(overflowing_literals)]
        match ($imm8) & 0b1111 {
            0 => $expand!(0),
            1 => $expand!(1),
            2 => $expand!(2),
            3 => $expand!(3),
            4 => $expand!(4),
            5 => $expand!(5),
            6 => $expand!(6),
            7 => $expand!(7),
            8 => $expand!(8),
            9 => $expand!(9),
            10 => $expand!(10),
            11 => $expand!(11),
            12 => $expand!(12),
            13 => $expand!(13),
            14 => $expand!(14),
            _ => $expand!(15),
        }
    };
}

macro_rules! constify_imm3 {
    ($imm8:expr, $expand:ident) => {
        #[allow(overflowing_literals)]
        match ($imm8) & 0b111 {
            0 => $expand!(0),
            1 => $expand!(1),
            2 => $expand!(2),
            3 => $expand!(3),
            4 => $expand!(4),
            5 => $expand!(5),
            6 => $expand!(6),
            _ => $expand!(7),
        }
    };
}

macro_rules! constify_imm2 {
    ($imm8:expr, $expand:ident) => {
        #[allow(overflowing_literals)]
        match ($imm8) & 0b11 {
            0 => $expand!(0),
            1 => $expand!(1),
            2 => $expand!(2),
            _ => $expand!(3),
        }
    };
}

// Constifies 5 bits along with an sae option without rounding control.
// See: https://github.com/llvm/llvm-project/blob/bd50cf905fa7c0c7caa134301c6ca0658c81eeb1/clang/lib/Sema/SemaChecking.cpp#L3497
#[allow(unused)]
macro_rules! constify_imm5_sae {
    ($imm5:expr, $imm4:expr, $expand:ident) => {
        #[allow(overflowing_literals)]
        match ($imm5 & 0b1111_1, $imm4 & 0b1111) {
            (0, 4) => $expand!(0, 4),
            (0, 8) => $expand!(0, 8),
            (0, 12) => $expand!(0, 12),
            (1, 4) => $expand!(1, 4),
            (1, 8) => $expand!(1, 8),
            (1, 12) => $expand!(1, 12),
            (2, 4) => $expand!(2, 4),
            (2, 8) => $expand!(2, 8),
            (2, 12) => $expand!(2, 12),
            (3, 4) => $expand!(3, 4),
            (3, 8) => $expand!(3, 8),
            (3, 12) => $expand!(3, 12),
            (4, 4) => $expand!(4, 4),
            (4, 8) => $expand!(4, 8),
            (4, 12) => $expand!(4, 12),
            (5, 4) => $expand!(5, 4),
            (5, 8) => $expand!(5, 8),
            (5, 12) => $expand!(5, 12),
            (6, 4) => $expand!(6, 4),
            (6, 8) => $expand!(6, 8),
            (6, 12) => $expand!(6, 12),
            (7, 4) => $expand!(7, 4),
            (7, 8) => $expand!(7, 8),
            (7, 12) => $expand!(7, 12),
            (8, 4) => $expand!(8, 4),
            (8, 8) => $expand!(8, 8),
            (8, 12) => $expand!(8, 12),
            (9, 4) => $expand!(9, 4),
            (9, 8) => $expand!(9, 8),
            (9, 12) => $expand!(9, 12),
            (10, 4) => $expand!(10, 4),
            (10, 8) => $expand!(10, 8),
            (10, 12) => $expand!(10, 12),
            (11, 4) => $expand!(11, 4),
            (11, 8) => $expand!(11, 8),
            (11, 12) => $expand!(11, 12),
            (12, 4) => $expand!(12, 4),
            (12, 8) => $expand!(12, 8),
            (12, 12) => $expand!(12, 12),
            (13, 4) => $expand!(13, 4),
            (13, 8) => $expand!(13, 8),
            (13, 12) => $expand!(13, 12),
            (14, 4) => $expand!(14, 4),
            (14, 8) => $expand!(14, 8),
            (14, 12) => $expand!(14, 12),
            (15, 4) => $expand!(15, 4),
            (15, 8) => $expand!(15, 8),
            (15, 12) => $expand!(15, 12),
            (16, 4) => $expand!(16, 4),
            (16, 8) => $expand!(16, 8),
            (16, 12) => $expand!(16, 12),
            (17, 4) => $expand!(17, 4),
            (17, 8) => $expand!(17, 8),
            (17, 12) => $expand!(17, 12),
            (18, 4) => $expand!(18, 4),
            (18, 8) => $expand!(18, 8),
            (18, 12) => $expand!(18, 12),
            (19, 4) => $expand!(19, 4),
            (19, 8) => $expand!(19, 8),
            (19, 12) => $expand!(19, 12),
            (20, 4) => $expand!(20, 4),
            (20, 8) => $expand!(20, 8),
            (20, 12) => $expand!(20, 12),
            (21, 4) => $expand!(21, 4),
            (21, 8) => $expand!(21, 8),
            (21, 12) => $expand!(21, 12),
            (22, 4) => $expand!(22, 4),
            (22, 8) => $expand!(22, 8),
            (22, 12) => $expand!(22, 12),
            (23, 4) => $expand!(23, 4),
            (23, 8) => $expand!(23, 8),
            (23, 12) => $expand!(23, 12),
            (24, 4) => $expand!(24, 4),
            (24, 8) => $expand!(24, 8),
            (24, 12) => $expand!(24, 12),
            (25, 4) => $expand!(25, 4),
            (25, 8) => $expand!(25, 8),
            (25, 12) => $expand!(25, 12),
            (26, 4) => $expand!(26, 4),
            (26, 8) => $expand!(26, 8),
            (26, 12) => $expand!(26, 12),
            (27, 4) => $expand!(27, 4),
            (27, 8) => $expand!(27, 8),
            (27, 12) => $expand!(27, 12),
            (28, 4) => $expand!(28, 4),
            (28, 8) => $expand!(28, 8),
            (28, 12) => $expand!(28, 12),
            (29, 4) => $expand!(29, 4),
            (29, 8) => $expand!(29, 8),
            (29, 12) => $expand!(29, 12),
            (30, 4) => $expand!(30, 4),
            (30, 8) => $expand!(30, 8),
            (30, 12) => $expand!(30, 12),
            (31, 4) => $expand!(31, 4),
            (31, 8) => $expand!(31, 8),
            (31, 12) => $expand!(31, 12),
            (_, _) => panic!("Invalid sae value"),
        }
    };
}

// For gather instructions, the only valid values for scale are 1, 2, 4 and 8.
// This macro enforces that.
#[allow(unused)]
macro_rules! constify_imm8_gather {
    ($imm8:expr, $expand:ident) => {
        #[allow(overflowing_literals)]
        match ($imm8) {
            1 => $expand!(1),
            2 => $expand!(2),
            4 => $expand!(4),
            8 => $expand!(8),
            _ => panic!("Only 1, 2, 4, and 8 are valid values"),
        }
    };
}

// For round instructions, the only valid values for rounding are 4, 8, 9, 10 and 11.
// This macro enforces that.
#[allow(unused)]
macro_rules! constify_imm4_round {
    ($imm8:expr, $expand:ident) => {
        #[allow(overflowing_literals)]
        match ($imm8) & 0b1111 {
            4 => $expand!(4),
            8 => $expand!(8),
            9 => $expand!(9),
            10 => $expand!(10),
            11 => $expand!(11),
            _ => panic!("Invalid round value"),
        }
    };
}

// For sae instructions, the only valid values for sae are 4 and 8.
// This macro enforces that.
#[allow(unused)]
macro_rules! constify_imm4_sae {
    ($imm8:expr, $expand:ident) => {
        #[allow(overflowing_literals)]
        match ($imm8) & 0b1111 {
            4 => $expand!(4),
            8 => $expand!(8),
            _ => panic!("Invalid sae value"),
        }
    };
}

// Two mantissas parameters.
// This macro enforces that.
#[allow(unused)]
macro_rules! constify_imm4_mantissas {
    ($imm4:expr, $imm2:expr, $expand:ident) => {
        #[allow(overflowing_literals)]
        match ($imm4, $imm2) {
            (0, 0) => $expand!(0, 0),
            (0, 1) => $expand!(0, 1),
            (0, 2) => $expand!(0, 2),
            (0, 3) => $expand!(0, 3),
            (1, 0) => $expand!(1, 0),
            (1, 1) => $expand!(1, 1),
            (1, 2) => $expand!(1, 2),
            (1, 3) => $expand!(1, 3),
            (2, 0) => $expand!(2, 0),
            (2, 1) => $expand!(2, 1),
            (2, 2) => $expand!(2, 2),
            (2, 3) => $expand!(2, 3),
            (3, 0) => $expand!(3, 0),
            (3, 1) => $expand!(3, 1),
            (3, 2) => $expand!(3, 2),
            (3, 3) => $expand!(3, 3),
            (4, 0) => $expand!(4, 0),
            (4, 1) => $expand!(4, 1),
            (4, 2) => $expand!(4, 2),
            (4, 3) => $expand!(4, 3),
            (5, 0) => $expand!(5, 0),
            (5, 1) => $expand!(5, 1),
            (5, 2) => $expand!(5, 2),
            (5, 3) => $expand!(5, 3),
            (6, 0) => $expand!(6, 0),
            (6, 1) => $expand!(6, 1),
            (6, 2) => $expand!(6, 2),
            (6, 3) => $expand!(6, 3),
            (7, 0) => $expand!(7, 0),
            (7, 1) => $expand!(7, 1),
            (7, 2) => $expand!(7, 2),
            (7, 3) => $expand!(7, 3),
            (8, 0) => $expand!(8, 0),
            (8, 1) => $expand!(8, 1),
            (8, 2) => $expand!(8, 2),
            (8, 3) => $expand!(8, 3),
            (9, 0) => $expand!(9, 0),
            (9, 1) => $expand!(9, 1),
            (9, 2) => $expand!(9, 2),
            (9, 3) => $expand!(9, 3),
            (10, 0) => $expand!(10, 0),
            (10, 1) => $expand!(10, 1),
            (10, 2) => $expand!(10, 2),
            (10, 3) => $expand!(10, 3),
            (11, 0) => $expand!(11, 0),
            (11, 1) => $expand!(11, 1),
            (11, 2) => $expand!(11, 2),
            (11, 3) => $expand!(11, 3),
            (12, 0) => $expand!(12, 0),
            (12, 1) => $expand!(12, 1),
            (12, 2) => $expand!(12, 2),
            (12, 3) => $expand!(12, 3),
            (13, 0) => $expand!(13, 0),
            (13, 1) => $expand!(13, 1),
            (13, 2) => $expand!(13, 2),
            (13, 3) => $expand!(13, 3),
            (14, 0) => $expand!(14, 0),
            (14, 1) => $expand!(14, 1),
            (14, 2) => $expand!(14, 2),
            (14, 3) => $expand!(14, 3),
            (15, 0) => $expand!(15, 0),
            (15, 1) => $expand!(15, 1),
            (15, 2) => $expand!(15, 2),
            (15, 3) => $expand!(15, 3),
            (_,_) => panic!("Invalid sae value"),
        }
    };
}

// Include mantissas parameters.
// For sae instructions, the only valid values for sae are 4 and 8.
// This macro enforces that.
#[allow(unused)]
macro_rules! constify_imm4_mantissas_sae {
    ($imm4_1:expr, $imm2:expr, $imm4_2:expr, $expand:ident) => {
        #[allow(overflowing_literals)]
        match ($imm4_1, $imm2, $imm4_2) {
            (0, 0, 4) => $expand!(0, 0, 4),
            (0, 0, 8) => $expand!(0, 0, 8),
            (0, 1, 4) => $expand!(0, 1, 4),
            (0, 1, 8) => $expand!(0, 1, 8),
            (0, 2, 4) => $expand!(0, 2, 4),
            (0, 2, 8) => $expand!(0, 2, 8),
            (0, 3, 4) => $expand!(0, 3, 4),
            (0, 3, 8) => $expand!(0, 3, 8),
            (1, 0, 4) => $expand!(1, 0, 4),
            (1, 0, 8) => $expand!(1, 0, 8),
            (1, 1, 4) => $expand!(1, 1, 4),
            (1, 1, 8) => $expand!(1, 1, 8),
            (1, 2, 4) => $expand!(1, 2, 4),
            (1, 2, 8) => $expand!(1, 2, 8),
            (1, 3, 4) => $expand!(1, 3, 4),
            (1, 3, 8) => $expand!(1, 3, 8),
            (2, 0, 4) => $expand!(2, 0, 4),
            (2, 0, 8) => $expand!(2, 0, 8),
            (2, 1, 4) => $expand!(2, 1, 4),
            (2, 1, 8) => $expand!(2, 1, 8),
            (2, 2, 4) => $expand!(2, 2, 4),
            (2, 2, 8) => $expand!(2, 2, 8),
            (2, 3, 4) => $expand!(2, 3, 4),
            (2, 3, 8) => $expand!(2, 3, 8),
            (3, 0, 4) => $expand!(3, 0, 4),
            (3, 0, 8) => $expand!(3, 0, 8),
            (3, 1, 4) => $expand!(3, 1, 4),
            (3, 1, 8) => $expand!(3, 1, 8),
            (3, 2, 4) => $expand!(3, 2, 4),
            (3, 2, 8) => $expand!(3, 2, 8),
            (3, 3, 4) => $expand!(3, 3, 4),
            (3, 3, 8) => $expand!(3, 3, 8),
            (4, 0, 4) => $expand!(4, 0, 4),
            (4, 0, 8) => $expand!(4, 0, 8),
            (4, 1, 4) => $expand!(4, 1, 4),
            (4, 1, 8) => $expand!(4, 1, 8),
            (4, 2, 4) => $expand!(4, 2, 4),
            (4, 2, 8) => $expand!(4, 2, 8),
            (4, 3, 4) => $expand!(4, 3, 4),
            (4, 3, 8) => $expand!(4, 3, 8),
            (5, 0, 4) => $expand!(5, 0, 4),
            (5, 0, 8) => $expand!(5, 0, 8),
            (5, 1, 4) => $expand!(5, 1, 4),
            (5, 1, 8) => $expand!(5, 1, 8),
            (5, 2, 4) => $expand!(5, 2, 4),
            (5, 2, 8) => $expand!(5, 2, 8),
            (5, 3, 4) => $expand!(5, 3, 4),
            (5, 3, 8) => $expand!(5, 3, 8),
            (6, 0, 4) => $expand!(6, 0, 4),
            (6, 0, 8) => $expand!(6, 0, 8),
            (6, 1, 4) => $expand!(6, 1, 4),
            (6, 1, 8) => $expand!(6, 1, 8),
            (6, 2, 4) => $expand!(6, 2, 4),
            (6, 2, 8) => $expand!(6, 2, 8),
            (6, 3, 4) => $expand!(6, 3, 4),
            (6, 3, 8) => $expand!(6, 3, 8),
            (7, 0, 4) => $expand!(7, 0, 4),
            (7, 0, 8) => $expand!(7, 0, 8),
            (7, 1, 4) => $expand!(7, 1, 4),
            (7, 1, 8) => $expand!(7, 1, 8),
            (7, 2, 4) => $expand!(7, 2, 4),
            (7, 2, 8) => $expand!(7, 2, 8),
            (7, 3, 4) => $expand!(7, 3, 4),
            (7, 3, 8) => $expand!(7, 3, 8),
            (8, 0, 4) => $expand!(8, 0, 4),
            (8, 0, 8) => $expand!(8, 0, 8),
            (8, 1, 4) => $expand!(8, 1, 4),
            (8, 1, 8) => $expand!(8, 1, 8),
            (8, 2, 4) => $expand!(8, 2, 4),
            (8, 2, 8) => $expand!(8, 2, 8),
            (8, 3, 4) => $expand!(8, 3, 4),
            (8, 3, 8) => $expand!(8, 3, 8),
            (9, 0, 4) => $expand!(9, 0, 4),
            (9, 0, 8) => $expand!(9, 0, 8),
            (9, 1, 4) => $expand!(9, 1, 4),
            (9, 1, 8) => $expand!(9, 1, 8),
            (9, 2, 4) => $expand!(9, 2, 4),
            (9, 2, 8) => $expand!(9, 2, 8),
            (9, 3, 4) => $expand!(9, 3, 4),
            (9, 3, 8) => $expand!(9, 3, 8),
            (10, 0, 4) => $expand!(10, 0, 4),
            (10, 0, 8) => $expand!(10, 0, 8),
            (10, 1, 4) => $expand!(10, 1, 4),
            (10, 1, 8) => $expand!(10, 1, 8),
            (10, 2, 4) => $expand!(10, 2, 4),
            (10, 2, 8) => $expand!(10, 2, 8),
            (10, 3, 4) => $expand!(10, 3, 4),
            (10, 3, 8) => $expand!(10, 3, 8),
            (11, 0, 4) => $expand!(11, 0, 4),
            (11, 0, 8) => $expand!(11, 0, 8),
            (11, 1, 4) => $expand!(11, 1, 4),
            (11, 1, 8) => $expand!(11, 1, 8),
            (11, 2, 4) => $expand!(11, 2, 4),
            (11, 2, 8) => $expand!(11, 2, 8),
            (11, 3, 4) => $expand!(11, 3, 4),
            (11, 3, 8) => $expand!(11, 3, 8),
            (12, 0, 4) => $expand!(12, 0, 4),
            (12, 0, 8) => $expand!(12, 0, 8),
            (12, 1, 4) => $expand!(12, 1, 4),
            (12, 1, 8) => $expand!(12, 1, 8),
            (12, 2, 4) => $expand!(12, 2, 4),
            (12, 2, 8) => $expand!(12, 2, 8),
            (12, 3, 4) => $expand!(12, 3, 4),
            (12, 3, 8) => $expand!(12, 3, 8),
            (13, 0, 4) => $expand!(13, 0, 4),
            (13, 0, 8) => $expand!(13, 0, 8),
            (13, 1, 4) => $expand!(13, 1, 4),
            (13, 1, 8) => $expand!(13, 1, 8),
            (13, 2, 4) => $expand!(13, 2, 4),
            (13, 2, 8) => $expand!(13, 2, 8),
            (13, 3, 4) => $expand!(13, 3, 4),
            (13, 3, 8) => $expand!(13, 3, 8),
            (14, 0, 4) => $expand!(14, 0, 4),
            (14, 0, 8) => $expand!(14, 0, 8),
            (14, 1, 4) => $expand!(14, 1, 4),
            (14, 1, 8) => $expand!(14, 1, 8),
            (14, 2, 4) => $expand!(14, 2, 4),
            (14, 2, 8) => $expand!(14, 2, 8),
            (14, 3, 4) => $expand!(14, 3, 4),
            (14, 3, 8) => $expand!(14, 3, 8),
            (15, 0, 4) => $expand!(15, 0, 4),
            (15, 0, 8) => $expand!(15, 0, 8),
            (15, 1, 4) => $expand!(15, 1, 4),
            (15, 1, 8) => $expand!(15, 1, 8),
            (15, 2, 4) => $expand!(15, 2, 4),
            (15, 2, 8) => $expand!(15, 2, 8),
            (15, 3, 4) => $expand!(15, 3, 4),
            (15, 3, 8) => $expand!(15, 3, 8),
            (_,_,_) => panic!("Invalid sae value"),
        }
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
