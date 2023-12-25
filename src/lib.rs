pub mod solutions;

use std::hash::BuildHasherDefault;

use anyhow::Result;

use indexmap::{IndexMap, IndexSet};
use rustc_hash::FxHasher;
use solutions::*;

pub type FxIndexMap<K, V> = IndexMap<K, V, BuildHasherDefault<FxHasher>>;
pub type FxIndexSet<T> = IndexSet<T, BuildHasherDefault<FxHasher>>;

type SolutionFn = fn(&str) -> Result<(usize, usize)>;
// Aggregated here (instead of main.rs) since our benchmarks also need this list
pub static ALL_SOLUTIONS: &[SolutionFn] = &[
    day1,
    day2,
    day3,
    day4,
    day5,
    day6,
    day7,
    day8,
    day9,
    day10::<140>,
    day11::<140, 1000000>,
    day12,
    day13,
    day14::<100>,
    day15,
    day16::<110>,
    day17::<141>,
    day18,
    day19,
    day20,
    day21,
    day22,
    day23::<141>,
    day24::<200000000000000, 400000000000000>,
    day25,
];

pub fn read_day_input(n: usize) -> String {
    std::fs::read_to_string(format!("inputs/{}.txt", n)).unwrap()
}

pub trait AsciiByteSliceExt {
    fn trim_space(&self) -> &[u8];
    fn trim_space_start(&self) -> &[u8];
    fn trim_space_end(&self) -> &[u8];
    fn parse_usize(&self) -> usize;
    fn parse_isize(&self) -> isize;
}

// Trim implementations are copied from the still unstable `feature(byte_slice_trim_ascii)`.
// (github.com/rust-lang/rust/issues/94035)
impl AsciiByteSliceExt for [u8] {
    fn trim_space(&self) -> &[u8] {
        self.trim_space_start().trim_space_end()
    }

    fn trim_space_start(&self) -> &[u8] {
        let mut bytes = self;
        while let [first, rest @ ..] = bytes {
            if *first == b' ' {
                bytes = rest;
            } else {
                break;
            }
        }
        bytes
    }

    fn trim_space_end(&self) -> &[u8] {
        let mut bytes = self;
        while let [rest @ .., last] = bytes {
            if *last == b' ' {
                bytes = rest;
            } else {
                break;
            }
        }
        bytes
    }

    /// Result is only correct if the bytes represent a valid positive number without any additional
    /// characters!
    fn parse_usize(&self) -> usize {
        let mut ret = 0;
        for b in self {
            ret = ret * 10 + (b - b'0') as usize;
        }
        ret
    }

    /// Result is only correct if the bytes represent a valid number without any additional
    /// characters!
    fn parse_isize(&self) -> isize {
        let mut bytes = self;
        let mut negative = false;
        if self[0] == b'-' {
            bytes = &self[1..];
            negative = true;
        }
        let mut ret = 0;
        for b in bytes {
            ret = ret * 10 + (b - b'0') as isize;
        }
        if negative {
            -ret
        } else {
            ret
        }
    }
}
