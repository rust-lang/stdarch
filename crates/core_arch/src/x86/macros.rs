//! Utility macros.
//!
// Helper struct used to trigger const eval errors when the const generic immediate value `imm` is
// not a round number.
pub(crate) struct ValidateConstRound<const IMM: i32>;
impl<const IMM: i32> ValidateConstRound<IMM> {
    pub(crate) const VALID: () = {
        let _ = 1 / ((IMM == 4 || IMM == 8 || IMM == 9 || IMM == 10 || IMM == 11) as usize);
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
        let _ = 1 / ((IMM == 4 || IMM == 8) as usize);
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
        let _ = 1 / ((IMM == 4 || IMM == 8 || IMM == 12) as usize);
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
        let _ = 1 / ((IMM >= MIN && IMM <= MAX) as usize);
    };
}

#[allow(unused_macros)]
macro_rules! static_assert_imm_u8 {
    ($imm:ident) => {
        let _ =
            $crate::core_arch::x86::macros::ValidateConstImmU32::<$imm, 0, { (1 << 8) - 1 }>::VALID;
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

// Constifies 8 bits along with an sae option without rounding control.
// The only valid values are 0 to 255.
// This macro enforces that.
#[allow(unused)]
macro_rules! constify_imm8_sae {
    ($imm8:expr, $expand:ident) => {
        #[allow(overflowing_literals)]
        match ($imm8) & 0b1111_1111 {
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
            31 => $expand!(31),
            32 => $expand!(32),
            33 => $expand!(33),
            34 => $expand!(34),
            35 => $expand!(35),
            36 => $expand!(36),
            37 => $expand!(37),
            38 => $expand!(38),
            39 => $expand!(39),
            40 => $expand!(40),
            41 => $expand!(41),
            42 => $expand!(42),
            43 => $expand!(43),
            44 => $expand!(44),
            45 => $expand!(45),
            46 => $expand!(46),
            47 => $expand!(47),
            48 => $expand!(48),
            49 => $expand!(49),
            50 => $expand!(50),
            51 => $expand!(51),
            52 => $expand!(52),
            53 => $expand!(53),
            54 => $expand!(54),
            55 => $expand!(55),
            56 => $expand!(56),
            57 => $expand!(57),
            58 => $expand!(58),
            59 => $expand!(59),
            60 => $expand!(60),
            61 => $expand!(61),
            62 => $expand!(62),
            63 => $expand!(63),
            64 => $expand!(64),
            65 => $expand!(65),
            66 => $expand!(66),
            67 => $expand!(67),
            68 => $expand!(68),
            69 => $expand!(69),
            70 => $expand!(70),
            71 => $expand!(71),
            72 => $expand!(72),
            73 => $expand!(73),
            74 => $expand!(74),
            75 => $expand!(75),
            76 => $expand!(76),
            77 => $expand!(77),
            78 => $expand!(78),
            79 => $expand!(79),
            80 => $expand!(80),
            81 => $expand!(81),
            82 => $expand!(82),
            83 => $expand!(83),
            84 => $expand!(84),
            85 => $expand!(85),
            86 => $expand!(86),
            87 => $expand!(87),
            88 => $expand!(88),
            89 => $expand!(89),
            90 => $expand!(90),
            91 => $expand!(91),
            92 => $expand!(92),
            93 => $expand!(93),
            94 => $expand!(94),
            95 => $expand!(95),
            96 => $expand!(96),
            97 => $expand!(97),
            98 => $expand!(98),
            99 => $expand!(99),
            100 => $expand!(100),
            101 => $expand!(101),
            102 => $expand!(102),
            103 => $expand!(103),
            104 => $expand!(104),
            105 => $expand!(105),
            106 => $expand!(106),
            107 => $expand!(107),
            108 => $expand!(108),
            109 => $expand!(109),
            110 => $expand!(110),
            111 => $expand!(111),
            112 => $expand!(112),
            113 => $expand!(113),
            114 => $expand!(114),
            115 => $expand!(115),
            116 => $expand!(116),
            117 => $expand!(117),
            118 => $expand!(118),
            119 => $expand!(119),
            120 => $expand!(120),
            121 => $expand!(121),
            122 => $expand!(122),
            123 => $expand!(123),
            124 => $expand!(124),
            125 => $expand!(125),
            126 => $expand!(126),
            127 => $expand!(127),
            128 => $expand!(128),
            129 => $expand!(129),
            130 => $expand!(130),
            131 => $expand!(131),
            132 => $expand!(132),
            133 => $expand!(133),
            134 => $expand!(134),
            135 => $expand!(135),
            136 => $expand!(136),
            137 => $expand!(137),
            138 => $expand!(138),
            139 => $expand!(139),
            140 => $expand!(140),
            141 => $expand!(141),
            142 => $expand!(142),
            143 => $expand!(143),
            144 => $expand!(144),
            145 => $expand!(145),
            146 => $expand!(146),
            147 => $expand!(147),
            148 => $expand!(148),
            149 => $expand!(149),
            150 => $expand!(150),
            151 => $expand!(151),
            152 => $expand!(152),
            153 => $expand!(153),
            154 => $expand!(154),
            155 => $expand!(155),
            156 => $expand!(156),
            157 => $expand!(157),
            158 => $expand!(158),
            159 => $expand!(159),
            160 => $expand!(160),
            161 => $expand!(161),
            162 => $expand!(162),
            163 => $expand!(163),
            164 => $expand!(164),
            165 => $expand!(165),
            166 => $expand!(166),
            167 => $expand!(167),
            168 => $expand!(168),
            169 => $expand!(169),
            170 => $expand!(170),
            171 => $expand!(171),
            172 => $expand!(172),
            173 => $expand!(173),
            174 => $expand!(174),
            175 => $expand!(175),
            176 => $expand!(176),
            177 => $expand!(177),
            178 => $expand!(178),
            179 => $expand!(179),
            180 => $expand!(180),
            181 => $expand!(181),
            182 => $expand!(182),
            183 => $expand!(183),
            184 => $expand!(184),
            185 => $expand!(185),
            186 => $expand!(186),
            187 => $expand!(187),
            188 => $expand!(188),
            189 => $expand!(189),
            190 => $expand!(190),
            191 => $expand!(191),
            192 => $expand!(192),
            193 => $expand!(193),
            194 => $expand!(194),
            195 => $expand!(195),
            196 => $expand!(196),
            197 => $expand!(197),
            198 => $expand!(198),
            199 => $expand!(199),
            200 => $expand!(200),
            201 => $expand!(201),
            202 => $expand!(202),
            203 => $expand!(203),
            204 => $expand!(204),
            205 => $expand!(205),
            206 => $expand!(206),
            207 => $expand!(207),
            208 => $expand!(208),
            209 => $expand!(209),
            210 => $expand!(210),
            211 => $expand!(211),
            212 => $expand!(212),
            213 => $expand!(213),
            214 => $expand!(214),
            215 => $expand!(215),
            216 => $expand!(216),
            217 => $expand!(217),
            218 => $expand!(218),
            219 => $expand!(219),
            220 => $expand!(220),
            221 => $expand!(221),
            222 => $expand!(222),
            223 => $expand!(223),
            224 => $expand!(224),
            225 => $expand!(225),
            226 => $expand!(226),
            227 => $expand!(227),
            228 => $expand!(228),
            229 => $expand!(229),
            230 => $expand!(230),
            231 => $expand!(231),
            232 => $expand!(232),
            233 => $expand!(233),
            234 => $expand!(234),
            235 => $expand!(235),
            236 => $expand!(236),
            237 => $expand!(237),
            238 => $expand!(238),
            239 => $expand!(239),
            240 => $expand!(240),
            241 => $expand!(241),
            242 => $expand!(242),
            243 => $expand!(243),
            244 => $expand!(244),
            245 => $expand!(245),
            246 => $expand!(246),
            247 => $expand!(247),
            248 => $expand!(248),
            249 => $expand!(249),
            250 => $expand!(250),
            251 => $expand!(251),
            252 => $expand!(252),
            253 => $expand!(253),
            254 => $expand!(254),
            255 => $expand!(255),
            _ => panic!("Invalid sae value"),
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
