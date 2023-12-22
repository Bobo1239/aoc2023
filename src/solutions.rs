#![allow(clippy::needless_range_loop)]

use std::{
    array,
    cmp::Ordering,
    collections::{HashMap, VecDeque},
    hash::Hasher,
    iter,
    num::Wrapping,
    ops::{Range, RangeInclusive},
    sync::atomic::{self, AtomicUsize},
    usize,
};

use aho_corasick::AhoCorasick;
use anyhow::Result;
use num::{traits::Pow, Integer};
use rayon::prelude::*;
use regex::bytes::Regex;
use rustc_hash::{FxHashMap, FxHasher};

use crate::{AsciiByteSliceExt, FxIndexMap};

pub fn day1(input: &str) -> Result<(usize, usize)> {
    // NOTE: regex doesn't work since it doesn't support overlapping matches (look-around)
    let patterns = &[
        "\n", "1", "2", "3", "4", "5", "6", "7", "8", "9", "one", "two", "three", "four", "five",
        "six", "seven", "eight", "nine",
    ];
    let ac = AhoCorasick::new(patterns)?;

    let mut sum_part1 = 0;
    let mut sum_part2 = 0;

    // First/last digit
    let mut part1 = [0, 0];
    let mut part2 = [0, 0];

    for mat in ac.find_overlapping_iter(input) {
        let (digit, real_digit) = match mat.pattern().as_usize() {
            0 => {
                sum_part1 += part1[0] * 10 + part1[1];
                sum_part2 += part2[0] * 10 + part2[1];
                part1 = [0, 0];
                part2 = [0, 0];
                continue;
            }
            d @ 1..=9 => (d, true),
            d => (d - 9, false),
        };

        if real_digit {
            if part1[0] == 0 {
                part1[0] = digit;
            }
            part1[1] = digit;
        }

        if part2[0] == 0 {
            part2[0] = digit;
        }
        part2[1] = digit;
    }

    Ok((sum_part1, sum_part2))
}

pub fn day2(input: &str) -> Result<(usize, usize)> {
    let mut valid_sum = 0;
    let mut power_sum = 0;
    for (i, l) in input.lines().enumerate() {
        // We don't care about the distinction between `;` and `,`
        let amount_and_colors = l.split(':').nth(1).unwrap().split([';', ',']);

        let mut valid = true;
        let mut max = [0; 3];
        for amount_and_color in amount_and_colors {
            let mut parts = amount_and_color.trim().split(' ');
            let amount: usize = parts.next().unwrap().parse()?;
            let (idx, limit) = match parts.next().unwrap() {
                "red" => (0, 12),
                "green" => (1, 13),
                "blue" => (2, 14),
                _ => unreachable!(),
            };
            if amount > limit {
                valid = false;
            }
            max[idx] = max[idx].max(amount);
        }

        if valid {
            valid_sum += i + 1;
        }
        power_sum += max.iter().product::<usize>();
    }
    Ok((valid_sum, power_sum))
}

pub fn day3(input: &str) -> Result<(usize, usize)> {
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Cell {
        Gear(u16), // Index into gear_adjacents
        Other,
        None,
    }
    const GRID_SIZE: usize = 140;

    let mut grid = [[Cell::None; GRID_SIZE]; GRID_SIZE];
    let mut gear_adjacents = Vec::new();
    let re_number = Regex::new(r"([0-9]+)")?;

    // NOTE: This is much faster than using a Regex even when going over the input multiple times
    let patterns = [
        [b'$', b'%'],
        [b'*', b'='],
        [b'/', b'+'],
        [b'@', b'#'],
        [b'-', b'&'],
    ];
    for pattern in patterns {
        for pos in memchr::memchr2_iter(pattern[0], pattern[1], input.as_bytes()) {
            let row = pos / (GRID_SIZE + 1);
            let col = pos % (GRID_SIZE + 1);
            let cell = if input.as_bytes()[pos] == b'*' {
                gear_adjacents.push(Vec::new());
                Cell::Gear(gear_adjacents.len() as u16 - 1)
            } else {
                Cell::Other
            };
            let row_low = if row == 0 { 0 } else { row - 1 };
            let row_high = (row + 1).min(GRID_SIZE - 1);
            let col_low = if col == 0 { 0 } else { col - 1 };
            let col_high = (col + 1).min(GRID_SIZE - 1);
            for y in row_low..=row_high {
                for x in col_low..=col_high {
                    grid[y][x] = cell;
                }
            }
        }
    }

    let mut sum = 0;
    for cap in re_number.find_iter(input.as_bytes()) {
        let row = cap.start() / (GRID_SIZE + 1);
        let col = cap.start() % (GRID_SIZE + 1);

        for i in 0..cap.len() {
            let cell = grid[row][col + i];
            if cell == Cell::None {
                continue;
            }
            let num = cap.as_bytes().parse_usize();
            sum += num;
            if let Cell::Gear(idx) = cell {
                gear_adjacents[idx as usize].push(num);
            }
            // Apparently numbers are never adjacent to two symbols so this is fine
            break;
        }
    }

    let sum_gear_ratios = gear_adjacents
        .iter()
        .filter(|nums| nums.len() == 2)
        .map(|nums| nums.iter().product::<usize>())
        .sum();
    Ok((sum, sum_gear_ratios))
}

pub fn day4(input: &str) -> Result<(usize, usize)> {
    let offset_colon = input.find(':').unwrap();
    let offset_bar = input.find('|').unwrap();
    let mut sum = 0;
    let mut card_amounts = vec![1; input.lines().count()];
    for (i, l) in input.lines().enumerate() {
        let (my_nums, win_nums) = l.split_at(offset_bar);
        let my_nums = &my_nums[offset_colon + 1..];
        let win_nums = &win_nums[1..];

        // Numbers only go up to 99 so we can use a u128 bitfield
        let my = my_nums
            .as_bytes()
            .chunks(3)
            .map(|c| c.trim_space_start().parse_usize())
            .fold(0u128, |acc, n| acc | 1 << n);
        let win = win_nums
            .as_bytes()
            .chunks(3)
            .map(|c| c.trim_space_start().parse_usize())
            .fold(0u128, |acc, n| acc | 1 << n);

        let matches = (my & win).count_ones();
        if matches > 0 {
            sum += 1 << (matches - 1);
            for j in i + 1..i + 1 + matches as usize {
                card_amounts[j] += card_amounts[i];
            }
        }
    }
    let total_cards = card_amounts.iter().sum();
    Ok((sum, total_cards))
}

pub fn day5(input: &str) -> Result<(usize, usize)> {
    fn range_intersects(a: &Range<usize>, b: &Range<usize>) -> Option<Range<usize>> {
        let intersection = a.start.max(b.start)..a.end.min(b.end);
        if intersection.is_empty() {
            None
        } else {
            Some(intersection)
        }
    }

    let mut lines = input.lines();
    let mut seeds: Vec<_> = lines.next().unwrap()[7..]
        .split(' ')
        .map(|s| s.as_bytes().parse_usize())
        .collect();
    lines.nth(1); // `advance_by()` is unstable (https://github.com/rust-lang/rust/issues/77404)

    let mut seed_ranges: Vec<_> = seeds
        .chunks(2)
        .map(|chunk| chunk[0]..chunk[0] + chunk[1])
        .collect();

    let mut maps = Vec::new();
    let mut current_map: Vec<[usize; 3]> = Vec::new();
    while let Some(l) = lines.next() {
        if l.is_empty() {
            maps.push(current_map);
            lines.next(); // Skip header
            current_map = Vec::new();
        } else {
            let mut numbers = l.split(' ').map(|s| s.as_bytes().parse_usize());
            current_map.push(array::from_fn(|_| numbers.next().unwrap()));
        }
    }
    maps.push(current_map);

    for map in &maps {
        for seed in &mut seeds {
            for [dst, src, len] in map {
                if (*src..src + len).contains(seed) {
                    *seed = dst + *seed - *src;
                    break;
                }
            }
        }

        let mut new_seed_ranges = Vec::new();
        'seed_ranges: while let Some(seed_range) = seed_ranges.pop() {
            for [dst, src, len] in map {
                let map_range = *src..src + len;
                if let Some(intersection) = range_intersects(&seed_range, &map_range) {
                    let left = seed_range.start..intersection.start;
                    let right = intersection.end..seed_range.end;
                    if !left.is_empty() {
                        seed_ranges.push(left);
                    }
                    if !right.is_empty() {
                        seed_ranges.push(right);
                    }
                    let new_start = dst + intersection.start - src;
                    let new_end = new_start + intersection.len();
                    new_seed_ranges.push(new_start..new_end);
                    continue 'seed_ranges;
                }
            }
            // No intersection => range stays
            new_seed_ranges.push(seed_range);
        }
        seed_ranges = new_seed_ranges;
    }

    let min = seeds.into_iter().min().unwrap();
    let min_range = seed_ranges
        .into_iter()
        .map(|range| range.start)
        .min()
        .unwrap();
    Ok((min, min_range))
}

pub fn day6(input: &str) -> Result<(usize, usize)> {
    fn calculate_ways(time: usize, distance: usize) -> usize {
        let time = time as f64;
        let distance = distance as f64;
        let x0 = (-time + (time * time - 4. * distance).sqrt()) / -2.;
        let x1 = (-time - (time * time - 4. * distance).sqrt()) / -2.;
        let x_min = x0.min(x1);
        let x_max = x0.max(x1);
        x_max.floor() as usize - x_min.ceil() as usize + 1
    }

    let numbers: Vec<_> = input
        .split_ascii_whitespace()
        .flat_map(|x| x.parse::<usize>().ok())
        .collect();
    let (times, distances) = numbers.split_at(numbers.len() / 2);
    let ways_product: usize = times
        .iter()
        .zip(distances)
        .map(|(time, dist)| calculate_ways(*time, *dist))
        .product();

    let numbers_combined: Vec<_> = input
        .lines()
        .map(|l| {
            l.replace(|c: char| !c.is_ascii_digit(), "")
                .parse::<usize>()
                .unwrap()
        })
        .collect();
    let ways_combined = calculate_ways(numbers_combined[0], numbers_combined[1]);

    Ok((ways_product, ways_combined))
}

pub fn day7(input: &str) -> Result<(usize, usize)> {
    #[derive(Debug, PartialEq, Eq)]
    struct Hand {
        hand_type: Type,
        hand_type_with_joker: Type,
        cards: [u8; 5],
        bid: usize,
        #[cfg(debug_assertions)]
        cards_string: String,
    }

    // Order is relevant!
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    enum Type {
        HighCard,
        OnePair,
        TwoPair,
        ThreeOfAKind,
        FullHouse,
        FourOfAKind,
        FiveOfAKind,
    }

    fn hand_type(mut card_counts: [u8; 5], joker: u8) -> Type {
        if joker != 0 {
            if card_counts[0] == joker && card_counts[1] != joker {
                card_counts[1] += joker;
                card_counts[0] = 0;
            } else {
                card_counts[0] += joker;
                for c in &mut card_counts {
                    if *c == joker {
                        *c = 0;
                        break;
                    }
                }
            }
            card_counts.sort_unstable_by(|a, b| b.cmp(a));
        }
        match card_counts {
            [5, ..] => Type::FiveOfAKind,
            [4, ..] => Type::FourOfAKind,
            [3, 2, ..] => Type::FullHouse,
            [3, ..] => Type::ThreeOfAKind,
            [2, 2, ..] => Type::TwoPair,
            [2, ..] => Type::OnePair,
            _ => Type::HighCard,
        }
    }

    let mut hands = Vec::new();
    for l in input.lines() {
        let (cards_str, bid) = l.split_once(' ').unwrap();
        let mut cards: [u8; 5] = cards_str.as_bytes().try_into()?;
        cards = cards.map(|c| match c {
            b'T' => 8,
            b'J' => 9,
            b'Q' => 10,
            b'K' => 11,
            b'A' => 12,
            n => n - b'2',
        });

        let mut cards_sorted = cards;
        cards_sorted.sort_unstable();

        let mut card_counts = [0; 5];
        let mut idx = 0;
        let mut last_card = cards_sorted[0];
        let mut joker = 0;
        for c in cards_sorted {
            // Joker
            if c == 9 {
                joker += 1;
            }
            if c == last_card {
                card_counts[idx] += 1;
            } else {
                idx += 1;
                card_counts[idx] += 1;
                last_card = c;
            }
        }
        card_counts.sort_unstable_by(|a, b| b.cmp(a));

        hands.push(Hand {
            hand_type: hand_type(card_counts, 0),
            hand_type_with_joker: hand_type(card_counts, joker),
            cards,
            bid: bid.parse()?,
            #[cfg(debug_assertions)]
            cards_string: cards_str.to_string(),
        });
    }

    hands.sort_by_cached_key(|hand| (hand.hand_type, hand.cards));
    let part1 = hands
        .iter()
        .enumerate()
        .map(|(i, hand)| (i + 1) * hand.bid)
        .sum();

    hands.sort_by_cached_key(|hand| {
        (
            hand.hand_type_with_joker,
            hand.cards.map(|c| if c == 9 { 0 } else { c + 1 }),
        )
    });
    let part2 = hands
        .iter()
        .enumerate()
        .map(|(i, hand)| (i + 1) * hand.bid)
        .sum();

    Ok((part1, part2))
}

pub fn day8(input: &str) -> Result<(usize, usize)> {
    let mut lines = input.lines();
    let directions = lines.next().unwrap().as_bytes();

    let mut start_nodes = Vec::new();
    let mut end_nodes = Vec::new();
    let mut part1_start_end = (0, 0);

    let mut node_to_idx = FxHashMap::default();
    let mut node_edges = Vec::new();
    for (node_idx, l) in lines.skip(1).enumerate() {
        let l = l.as_bytes();
        let node = &l[..3];
        let left = &l[7..10];
        let right = &l[12..15];
        node_to_idx.insert(node, node_idx);
        if node[2] == b'A' {
            start_nodes.push(node_idx);
            if node == b"AAA" {
                part1_start_end.0 = node_idx;
            }
        } else if node[2] == b'Z' {
            end_nodes.push(node_idx);
            if node == b"ZZZ" {
                part1_start_end.1 = node_idx;
            }
        }
        node_edges.push((left, right));
    }

    let node_edges: Vec<_> = node_edges
        .into_iter()
        .map(|(l, r)| (node_to_idx[l], node_to_idx[r]))
        .collect();

    fn find_steps_to_goal(
        mut start_node: usize,
        goal: &[usize],
        node_edges: &[(usize, usize)],
        directions: &[u8],
    ) -> usize {
        let mut steps = 0;
        for d in directions.iter().cycle() {
            if *d == b'L' {
                start_node = node_edges[start_node].0
            } else {
                start_node = node_edges[start_node].1
            }
            steps += 1;
            if goal.contains(&start_node) {
                return steps;
            }
        }
        unreachable!()
    }

    let part1 = AtomicUsize::new(0);
    let part2 = start_nodes
        .into_par_iter()
        .map(|start| {
            let steps = find_steps_to_goal(start, &end_nodes, &node_edges, directions);
            if start == part1_start_end.0 {
                part1.store(steps, atomic::Ordering::SeqCst);
            }
            steps
        })
        .reduce(|| 1, |a: usize, b: usize| a.lcm(&b));
    Ok((part1.load(atomic::Ordering::SeqCst), part2))
}

pub fn day9(input: &str) -> Result<(usize, usize)> {
    fn solve_problem(numbers: &mut [isize]) -> (isize, isize) {
        let len = numbers.len();
        let first = numbers[0];
        let last = numbers[len - 1];

        let mut last_diff = 0;
        let mut need_to_go_deeper = false;
        for i in 0..numbers.len() - 1 {
            let diff = numbers[i + 1] - numbers[i];
            numbers[i] = diff;
            if i != 0 && numbers[i + 1] - numbers[i] != last_diff {
                need_to_go_deeper = true;
            }
            last_diff = diff;
        }
        let delta = if need_to_go_deeper {
            solve_problem(&mut numbers[..len - 1])
        } else {
            (last_diff, last_diff)
        };
        (first - delta.0, last + delta.1)
    }

    let sums = input
        .lines()
        .map(|l| {
            let mut numbers = Vec::with_capacity(32);
            numbers.extend(l.as_bytes().split(|b| *b == b' ').map(|x| x.parse_isize()));
            solve_problem(&mut numbers)
        })
        .fold((0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
    Ok((sums.1 as usize, sums.0 as usize))
}

pub fn day10<const GRID_SIZE: usize>(input: &str) -> Result<(usize, usize)> {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Cell {
        Vertical,
        Horizontal,
        NE,
        NW,
        SW,
        SE,
        Start,
        None,
    }

    impl Cell {
        fn walk(self, in_dir: Direction) -> Option<Direction> {
            Some(match in_dir {
                Direction::North => match self {
                    Cell::Vertical => Direction::South,
                    Cell::NE => Direction::East,
                    Cell::NW => Direction::West,
                    _ => return None,
                },
                Direction::East => match self {
                    Cell::Horizontal => Direction::West,
                    Cell::NE => Direction::North,
                    Cell::SE => Direction::South,
                    _ => return None,
                },
                Direction::South => match self {
                    Cell::Vertical => Direction::North,
                    Cell::SE => Direction::East,
                    Cell::SW => Direction::West,
                    _ => return None,
                },
                Direction::West => match self {
                    Cell::Horizontal => Direction::East,
                    Cell::NW => Direction::North,
                    Cell::SW => Direction::South,
                    _ => return None,
                },
            })
        }

        fn from_out_dirs(dir0: Direction, dir1: Direction) -> Cell {
            for cell in [
                Cell::Vertical,
                Cell::Horizontal,
                Cell::NE,
                Cell::NW,
                Cell::SW,
                Cell::SE,
            ] {
                if let Some(out_dir) = cell.walk(dir0) {
                    if out_dir == dir1 {
                        return cell;
                    }
                }
            }
            panic!("dir0 and dir1 mustn't be the same")
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Direction {
        North,
        East,
        South,
        West,
    }

    impl Direction {
        const ALL: [Direction; 4] = [
            Direction::North,
            Direction::East,
            Direction::South,
            Direction::West,
        ];

        fn flip(self) -> Direction {
            match self {
                Direction::North => Direction::South,
                Direction::East => Direction::West,
                Direction::South => Direction::North,
                Direction::West => Direction::East,
            }
        }

        fn delta(self) -> (isize, isize) {
            match self {
                Direction::North => (-1, 0),
                Direction::East => (0, 1),
                Direction::South => (1, 0),
                Direction::West => (0, -1),
            }
        }
    }

    let mut grid = [[Cell::None; GRID_SIZE]; GRID_SIZE];

    let mut start = (0, 0);
    let mut input = input.as_bytes().iter().filter(|x| **x != b'\n');
    for y in 0..GRID_SIZE {
        for x in 0..GRID_SIZE {
            grid[y][x] = match input.next().unwrap() {
                b'|' => Cell::Vertical,
                b'-' => Cell::Horizontal,
                b'L' => Cell::NE,
                b'J' => Cell::NW,
                b'7' => Cell::SW,
                b'F' => Cell::SE,
                b'.' => Cell::None,
                b'S' => {
                    start = (y as isize, x as isize);
                    Cell::Start
                }
                _ => unreachable!(),
            }
        }
    }

    // Determine cell type of start cell and correct its type in the grid
    let mut current_cells = [(0, 0, Direction::North); 2];
    let mut idx = 0;
    for dir in Direction::ALL {
        let delta = dir.delta();
        let y = start.0 + delta.0;
        let x = start.1 + delta.1;
        if let Some(cell) = grid.get(y as usize).and_then(|r| r.get(x as usize)) {
            if cell.walk(dir.flip()).is_some() {
                current_cells[idx] = (start.0, start.1, dir);
                idx += 1;
            }
        }
    }
    debug_assert_eq!(idx, 2);
    grid[start.0 as usize][start.1 as usize] =
        Cell::from_out_dirs(current_cells[0].2, current_cells[1].2);

    let mut part_of_loop = [[false; GRID_SIZE]; GRID_SIZE];
    part_of_loop[start.0 as usize][start.1 as usize] = true;

    // Walk along both ends until the loop closes
    let mut len = 0;
    let mut last = (0, 0);
    'outer: loop {
        for (y, x, out_dir) in &mut current_cells {
            let delta = out_dir.delta();
            let new_y = *y + delta.0;
            let new_x = *x + delta.1;
            let cell = grid[new_y as usize][new_x as usize];
            let new_out_dir = cell.walk(out_dir.flip()).unwrap();

            *y = new_y;
            *x = new_x;
            *out_dir = new_out_dir;
            part_of_loop[*y as usize][*x as usize] = true;

            // Break if both ends reach each other
            if new_y == last.0 && new_x == last.1 {
                len += 1;
                break 'outer;
            }
            last.0 = new_y;
            last.1 = new_x;
        }
        len += 1;
    }

    // Determine cells inside loop
    let mut count = 0;
    for y in 0..GRID_SIZE {
        let mut inside = false;
        for x in 0..GRID_SIZE {
            if part_of_loop[y][x] {
                if matches!(grid[y][x], Cell::NE | Cell::NW | Cell::Vertical) {
                    inside = !inside;
                }
            } else if inside {
                count += 1;
            }
        }
    }

    Ok((len, count))
}

pub fn day11<const GRID_SIZE: usize, const PART2_FACTOR: isize>(
    input: &str,
) -> Result<(usize, usize)> {
    let mut galaxies = Vec::new();
    let mut cluster_row = 0isize;
    let mut occupany_cols = [false; GRID_SIZE];
    for (row, l) in input.lines().enumerate() {
        let mut empty_row = true;
        for (col, b) in l.as_bytes().iter().enumerate() {
            if *b == b'#' {
                galaxies.push(((row as isize, col as isize), (cluster_row, 0)));
                empty_row = false;
                occupany_cols[col] = true;
            }
        }
        if empty_row {
            cluster_row += 1;
        }
    }

    let empty_cols: Vec<_> = occupany_cols
        .into_iter()
        .enumerate()
        .filter(|(_, b)| !*b)
        .map(|(col, _)| col as isize)
        .collect();
    for ((_, col), (_, cluster_col)) in &mut galaxies {
        *cluster_col = empty_cols.iter().take_while(|c| *col > **c).count() as isize;
    }

    let mut sum = 0;
    let mut extra = 0;
    for i in 0..galaxies.len() {
        for j in i..galaxies.len() {
            let (a, cluster_a) = galaxies[i];
            let (b, cluster_b) = galaxies[j];
            sum += (b.0 - a.0).abs() + (b.1 - a.1).abs();
            extra += (cluster_b.0 - cluster_a.0).abs() + (cluster_b.1 - cluster_a.1).abs();
        }
    }

    Ok((
        (sum + extra) as usize,
        (sum + extra * (PART2_FACTOR - 1)) as usize,
    ))
}

pub fn day12(input: &str) -> Result<(usize, usize)> {
    const MAX_STATE_LEN: usize = 20 * 5 + 4; // `* 5 + 4` due to folding
    const MAX_CHUNKS_LEN: usize = 32;

    struct Problem {
        state: Vec<u8>,
        chunks: Vec<u8>,
        cache: [[usize; MAX_STATE_LEN + 1]; MAX_CHUNKS_LEN + 1],
    }

    impl Problem {
        fn new(state: Vec<u8>, chunks: Vec<u8>) -> Problem {
            Problem {
                state,
                chunks,
                cache: [[usize::MAX; MAX_STATE_LEN + 1]; MAX_CHUNKS_LEN + 1],
            }
        }

        fn solve(&mut self, offset: usize, chunk_idx: usize) -> usize {
            let cached_value = self.cache[chunk_idx][offset];
            if cached_value != usize::MAX {
                cached_value
            } else {
                let ret = self.solve_(offset, chunk_idx);
                self.cache[chunk_idx][offset] = ret;
                ret
            }
        }

        fn solve_(&mut self, mut offset: usize, chunk_idx: usize) -> usize {
            let mut state = &self.state[offset..];
            let chunks = &self.chunks[chunk_idx..];

            match (state.is_empty(), chunks.is_empty()) {
                (true, true) => return 1,
                (true, false) => return 0,
                (false, true) => {
                    return if state.iter().all(|b| *b != b'#') {
                        1
                    } else {
                        0
                    };
                }
                (false, false) => {}
            }

            while state[0] == b'.' {
                offset += 1;
                state = &state[1..];
                if state.is_empty() {
                    // `chunks` isn't empty here so there are no arrangements
                    return 0;
                }
            }

            let damaged = chunks[0] as usize;
            match state[0] {
                b'#' => {
                    if state.iter().take(damaged).any(|b| *b == b'.')
                        || state.get(damaged) == Some(&b'#')
                    {
                        0
                    } else {
                        // `state[1]` must be a `.` or a `?` which becomes a `.`
                        if damaged > state.len() {
                            0
                        } else if damaged + 1 > state.len() {
                            // We're at the end already
                            if chunks[1..].is_empty() {
                                1
                            } else {
                                0
                            }
                        } else {
                            self.solve(offset + damaged + 1, chunk_idx + 1)
                        }
                    }
                }
                b'?' => {
                    let mut ret = 0;

                    // Assume `#`
                    ret += if state.iter().take(damaged).all(|b| *b != b'.') {
                        match damaged.cmp(&state.len()) {
                            Ordering::Less => {
                                if state[damaged] == b'#' {
                                    0 // Overshoot
                                } else {
                                    // state[damaged] must be a `.` (either directly or through `?`)
                                    self.solve(offset + damaged + 1, chunk_idx + 1)
                                }
                            }
                            Ordering::Equal => {
                                if chunks[1..].is_empty() {
                                    1
                                } else {
                                    0
                                }
                            }
                            Ordering::Greater => 0,
                        }
                    } else {
                        0
                    };

                    // Assume `.`
                    ret += self.solve(offset + 1, chunk_idx);

                    ret
                }
                _ => unreachable!(),
            }
        }
    }

    let sums = input
        .par_lines()
        .map(|l| {
            let (string, numbers) = l.split_once(' ').unwrap();
            let state = string.as_bytes().to_vec();
            let chunks: Vec<_> = numbers
                .split(',')
                .map(|x| x.as_bytes().parse_usize() as u8)
                .collect();

            let state2: Vec<_> = state
                .iter()
                .copied()
                .chain(iter::once(b'?'))
                .cycle()
                .take(state.len() * 5 + 4)
                .collect();
            let chunks2: Vec<_> = chunks
                .iter()
                .copied()
                .cycle()
                .take(chunks.len() * 5)
                .collect();
            (
                Problem::new(state, chunks).solve(0, 0),
                Problem::new(state2, chunks2).solve(0, 0),
            )
        })
        .reduce(|| (0, 0), |a, b| (a.0 + b.0, a.1 + b.1));

    Ok(sums)
}

pub fn day13(input: &str) -> Result<(usize, usize)> {
    fn find_symmetry(lines: &[u32]) -> (Option<usize>, Option<usize>) {
        let mut current_symmetry = None;
        let mut potential_symmetry = None;
        'outer: for i in 0..lines.len() - 1 {
            let mut off_by_one_lines = 0;
            for j in 0..=i.min(lines.len() - i - 2) {
                match (lines[i - j] ^ lines[i + 1 + j]).count_ones() {
                    0 => {}
                    1 => off_by_one_lines += 1,
                    _ => continue 'outer,
                }
            }
            if off_by_one_lines == 1 {
                potential_symmetry = Some(i + 1);
            } else {
                current_symmetry = Some(i + 1);
            }
        }
        (current_symmetry, potential_symmetry)
    }

    let mut rows = Vec::new();
    let mut row_length = 0;
    let mut part1 = 0;
    let mut part2 = 0;
    for l in input.lines().chain(iter::once("")) {
        if !l.is_empty() {
            row_length = l.len();
            rows.push(
                l.as_bytes()
                    .iter()
                    .fold(0, |acc, x| (acc << 1) + if *x == b'#' { 1 } else { 0 }),
            )
        } else {
            let mut cols = vec![0; row_length];
            for row in &rows {
                for i in 0..row_length {
                    cols[i] = (cols[i] << 1) + ((row >> (row_length - i - 1)) & 0x1);
                }
            }
            let sym_rows = find_symmetry(&rows);
            let sym_cols = find_symmetry(&cols);
            part1 += 100 * sym_rows.0.unwrap_or(0) + sym_cols.0.unwrap_or(0);
            part2 += 100 * sym_rows.1.unwrap_or(0) + sym_cols.1.unwrap_or(0);

            rows.clear();
        }
    }
    Ok((part1, part2))
}

pub fn day14<const GRID_SIZE: usize>(input: &str) -> Result<(usize, usize)> {
    const PART2_CYCLES: usize = 1000000000;
    const MAX_PARTS_PER_LINE: usize = 20;

    let mut stones = Vec::new();
    let mut partitions_all_rows = Vec::new();
    let mut partitions_all_cols = vec![vec![]; GRID_SIZE];
    let mut last_fixed_in_col = vec![0; GRID_SIZE];
    for (row, l) in input.lines().enumerate() {
        let mut parts_row = Vec::new();
        let mut last_fixed_in_row = 0;
        for (col, b) in l.as_bytes().iter().enumerate() {
            match b {
                b'#' => {
                    let part_row = last_fixed_in_row..col;
                    let part_col = last_fixed_in_col[col]..row;
                    if !part_row.is_empty() {
                        parts_row.push(last_fixed_in_row..col);
                    }
                    if !part_col.is_empty() {
                        partitions_all_cols[col].push(last_fixed_in_col[col]..row);
                    }
                    last_fixed_in_row = col + 1;
                    last_fixed_in_col[col] = row + 1;
                }
                b'O' => stones.push((col, row)),
                _ => {}
            }
        }
        if last_fixed_in_row != GRID_SIZE {
            parts_row.push(last_fixed_in_row..GRID_SIZE);
        }
        partitions_all_rows.push(parts_row);
    }
    for col in 0..GRID_SIZE {
        if last_fixed_in_col[col] != GRID_SIZE {
            partitions_all_cols[col].push(last_fixed_in_col[col]..GRID_SIZE);
        }
    }

    // Use arrays so we do less pointer chasing in `tilt()`
    let mut stones_in_partition_all_cols = [[0; MAX_PARTS_PER_LINE]; GRID_SIZE];
    let mut stones_in_partition_all_rows = [[0; MAX_PARTS_PER_LINE]; GRID_SIZE];

    for (col, row) in &stones {
        let part_idx = partitions_all_cols[*col]
            .iter()
            .enumerate()
            .find(|(_, x)| x.contains(row))
            .unwrap()
            .0;
        stones_in_partition_all_cols[*col][part_idx] += 1;
    }

    let mut load_part1 = 0;
    for col in 0..GRID_SIZE {
        let partitions = &partitions_all_cols[col];
        let stones = &stones_in_partition_all_cols[col];
        for (range, amount) in partitions.iter().zip(stones.iter()) {
            for x in range.start..range.start + *amount as usize {
                load_part1 += GRID_SIZE - x;
            }
        }
    }

    // Part 2

    fn tilt<const GRID_SIZE: usize, const REVERSE: bool>(
        in_parts: &[Vec<Range<usize>>],
        in_stones: &[[u8; MAX_PARTS_PER_LINE]],
        out_pos_to_part_idx: &[[usize; GRID_SIZE]; GRID_SIZE],
        out_stones: &mut [[u8; MAX_PARTS_PER_LINE]],
    ) {
        out_stones.fill([0; MAX_PARTS_PER_LINE]);
        for x in 0..GRID_SIZE {
            let partitions = &in_parts[x];
            let stones = &in_stones[x];
            for (range, amount) in partitions.iter().zip(stones.iter()) {
                let range = if REVERSE {
                    range.end - (*amount as usize)..range.end
                } else {
                    range.start..range.start + *amount as usize
                };
                for out_idx in range {
                    let idx = out_pos_to_part_idx[out_idx][x];
                    out_stones[out_idx][idx] += 1;
                }
            }
        }
    }

    let mut pos_to_part_idx_rows = [[0xffff; GRID_SIZE]; GRID_SIZE];
    for i in 0..GRID_SIZE {
        for (idx, part) in partitions_all_rows[i].iter().enumerate() {
            pos_to_part_idx_rows[i][part.clone()].fill(idx);
        }
    }

    let mut pos_to_part_idx_cols = [[0xffff; GRID_SIZE]; GRID_SIZE];
    for i in 0..GRID_SIZE {
        for (idx, part) in partitions_all_cols[i].iter().enumerate() {
            pos_to_part_idx_cols[i][part.clone()].fill(idx);
        }
    }

    let mut hashmap = FxHashMap::default();
    let mut load_history = Vec::new();
    for _ in 0..PART2_CYCLES {
        // West
        tilt::<GRID_SIZE, false>(
            &partitions_all_cols,
            &stones_in_partition_all_cols,
            &pos_to_part_idx_rows,
            &mut stones_in_partition_all_rows,
        );
        // South
        tilt::<GRID_SIZE, false>(
            &partitions_all_rows,
            &stones_in_partition_all_rows,
            &pos_to_part_idx_cols,
            &mut stones_in_partition_all_cols,
        );
        // East
        tilt::<GRID_SIZE, true>(
            &partitions_all_cols,
            &stones_in_partition_all_cols,
            &pos_to_part_idx_rows,
            &mut stones_in_partition_all_rows,
        );
        // North
        tilt::<GRID_SIZE, true>(
            &partitions_all_rows,
            &stones_in_partition_all_rows,
            &pos_to_part_idx_cols,
            &mut stones_in_partition_all_cols,
        );
        let mut load = 0;
        for row in 0..GRID_SIZE {
            load += (GRID_SIZE - row)
                * stones_in_partition_all_rows[row]
                    .iter()
                    .map(|x| *x as usize)
                    .sum::<usize>();
        }
        load_history.push(load);

        // Detect when the configuration enters a cycle
        // Don't store our state in the HashMap
        let mut hasher = FxHasher::default();
        for stones_row in &stones_in_partition_all_rows {
            hasher.write(stones_row);
        }
        let key = hasher.finish();
        if let Some(last_seen) = hashmap.insert(key, hashmap.len()) {
            let period = hashmap.len() - last_seen;
            let idx = (PART2_CYCLES - last_seen) % period;
            let load_part2 = load_history[last_seen + idx - 1];
            return Ok((load_part1, load_part2));
        }
    }
    unreachable!();
}

pub fn day15(input: &str) -> Result<(usize, usize)> {
    let mut sum = 0;
    let mut current_hash = Wrapping(0u8);
    let mut box_idx = 0;
    let mut focal_length = None;
    let mut label = Vec::with_capacity(6);

    let mut boxes = vec![FxIndexMap::default(); 256];

    for b in input.as_bytes() {
        match b {
            b',' | b'\n' => {
                sum += current_hash.0 as usize;
                if let Some(fc) = focal_length {
                    boxes[box_idx].insert(label.clone(), fc);
                } else {
                    boxes[box_idx].shift_remove(&label);
                }
                current_hash = Wrapping(0);
                focal_length = None;
                label.clear();
            }
            c => {
                match c {
                    b'=' | b'-' => {
                        box_idx = current_hash.0 as usize;
                    }
                    n if n.is_ascii_digit() => {
                        focal_length = Some(n - b'0');
                    }
                    _ => {
                        label.push(c);
                    }
                }
                current_hash += c;
                current_hash *= 17;
            }
        }
    }
    let mut total_power = 0;
    for (b, boxx) in boxes.iter().enumerate() {
        for (s, fc) in boxx.values().enumerate() {
            total_power += (b + 1) * (s + 1) * *fc as usize;
        }
    }
    Ok((sum, total_power))
}

pub fn day16<const GRID_SIZE: usize>(input: &str) -> Result<(usize, usize)> {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Cell {
        MirrorRightUp,
        MirrorRightDown,
        SplitterVertical,
        SplitterHorizontal,
        Border,
    }

    impl Cell {
        fn outgoing_rays(self, in_dir: Direction) -> &'static [Direction] {
            match self {
                Cell::MirrorRightUp => match in_dir {
                    Direction::Up => &[Direction::Right],
                    Direction::Down => &[Direction::Left],
                    Direction::Left => &[Direction::Down],
                    Direction::Right => &[Direction::Up],
                },
                Cell::MirrorRightDown => match in_dir {
                    Direction::Up => &[Direction::Left],
                    Direction::Down => &[Direction::Right],
                    Direction::Left => &[Direction::Up],
                    Direction::Right => &[Direction::Down],
                },
                Cell::SplitterVertical => match in_dir {
                    Direction::Down => &[Direction::Down],
                    Direction::Up => &[Direction::Up],
                    Direction::Left | Direction::Right => &[Direction::Up, Direction::Down],
                },
                Cell::SplitterHorizontal => match in_dir {
                    Direction::Left => &[Direction::Left],
                    Direction::Right => &[Direction::Right],
                    Direction::Up | Direction::Down => &[Direction::Left, Direction::Right],
                },
                Cell::Border => &[],
            }
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    enum Direction {
        Up = 1,
        Down = 1 << 1,
        Left = 1 << 2,
        Right = 1 << 3,
    }

    impl Direction {
        const ALL: [Direction; 4] = [
            Direction::Up,
            Direction::Down,
            Direction::Left,
            Direction::Right,
        ];
        fn is_horizontal(self) -> bool {
            matches!(self, Direction::Left | Direction::Right)
        }
        fn flip(self) -> Direction {
            match self {
                Direction::Up => Direction::Down,
                Direction::Down => Direction::Up,
                Direction::Left => Direction::Right,
                Direction::Right => Direction::Left,
            }
        }
        fn to_bitmask(self) -> u8 {
            self as u8
        }
    }

    #[derive(Debug, Clone, Copy)]
    struct Info<T> {
        data: T,
        // IDEA: Changing these to u8 improves performance but requires many casts in the algorithm
        //       which isn't worth it atm.
        x: usize,
        y: usize,
        row_idx: usize,
        col_idx: usize,
    }

    // NOTE: We add a border so we don't have to do bounds checks in our hot loop
    let mut mirrors_rows: [_; GRID_SIZE] = array::from_fn(|_| Vec::new());
    let mut mirrors_cols: [_; GRID_SIZE] = array::from_fn(|x| {
        vec![Info {
            data: Cell::Border,
            x,
            y: 0,
            row_idx: 1337,
            col_idx: 0,
        }]
    });
    for (y, l) in input.as_bytes().chunks(GRID_SIZE + 1).enumerate() {
        let mut mirrors_this_row = vec![Info {
            data: Cell::Border,
            x: 0,
            y,
            row_idx: 0,
            col_idx: 1337,
        }];
        for (x, b) in l.iter().take(GRID_SIZE).enumerate() {
            let cell = match *b {
                b'\\' => Some(Cell::MirrorRightDown),
                b'/' => Some(Cell::MirrorRightUp),
                b'|' => Some(Cell::SplitterVertical),
                b'-' => Some(Cell::SplitterHorizontal),
                _ => None,
            };
            if let Some(cell) = cell {
                let info = Info {
                    data: cell,
                    x,
                    y,
                    row_idx: mirrors_this_row.len(),
                    col_idx: mirrors_cols[x].len(),
                };
                mirrors_this_row.push(info);
                mirrors_cols[x].push(info);
            }
        }
        mirrors_this_row.push(Info {
            data: Cell::Border,
            x: GRID_SIZE - 1,
            y,
            row_idx: mirrors_this_row.len(),
            col_idx: 1337,
        });
        mirrors_rows[y] = mirrors_this_row;
    }
    for x in 0..GRID_SIZE {
        mirrors_cols[x].push(Info {
            data: Cell::Border,
            x,
            y: GRID_SIZE - 1,
            row_idx: 1337,
            col_idx: mirrors_cols[x].len(),
        });
    }

    fn calculate_energized_cells<const GRID_SIZE: usize>(
        in_dir: Direction,
        in_pos: usize,
        mirrors_rows: &[Vec<Info<Cell>>; GRID_SIZE],
        mirrors_cols: &[Vec<Info<Cell>>; GRID_SIZE],
    ) -> usize {
        // IDEA: Split into energized_rows/colsuse bitmaps
        // IDEA: Bitmaps
        // IDEA: Count while exploring instead of only at the end
        let mut energized = [[false; GRID_SIZE]; GRID_SIZE];

        // Get first cell in line
        let reverse = matches!(in_dir, Direction::Up | Direction::Left);
        let line = if in_dir.is_horizontal() {
            &mirrors_rows[in_pos]
        } else {
            &mirrors_cols[in_pos]
        };
        let Some(first) = (if reverse { line.last() } else { line.first() }) else {
            return GRID_SIZE;
        };
        debug_assert_eq!(first.data, Cell::Border);

        let mut to_explore = Vec::new();
        to_explore.push(Info {
            data: in_dir,
            x: first.x,
            y: first.y,
            row_idx: first.row_idx,
            col_idx: first.col_idx,
        });

        // 0 is a bitmask of already explored directions; This is much faster than a hashmap
        let mut explored = [[0; GRID_SIZE]; GRID_SIZE];
        while let Some(Info {
            data: dir,
            x,
            y,
            row_idx,
            col_idx,
        }) = to_explore.pop()
        {
            if (explored[x][y] & dir.to_bitmask()) != 0 {
                continue;
            }
            explored[x][y] |= dir.to_bitmask();

            // No bounds check necessary since we're guaranteed to hit a border on the edge from
            // which no new rays are outgoing
            let hit = match dir {
                Direction::Up => mirrors_cols[x][col_idx - 1],
                Direction::Down => mirrors_cols[x][col_idx + 1],
                Direction::Left => mirrors_rows[y][row_idx - 1],
                Direction::Right => mirrors_rows[y][row_idx + 1],
            };
            match dir {
                Direction::Up | Direction::Down => {
                    for y in y.min(hit.y)..=y.max(hit.y) {
                        energized[y][x] = true;
                    }
                }
                Direction::Left | Direction::Right => {
                    energized[y][x.min(hit.x)..=x.max(hit.x)].fill(true);
                }
            }
            // Also set the reverse direction as explored which allows us to skip some
            // calculations in certain constellations.
            // This is correct since there's no case where the reverse direction could reach a
            // cell which we won't be exploring anyways with our current ray.
            // Just need to special case borders since they overlap other cells.
            if hit.data != Cell::Border {
                explored[hit.x][hit.y] |= dir.flip().to_bitmask();
            }
            for new_dir in hit.data.outgoing_rays(dir) {
                to_explore.push(Info {
                    data: *new_dir,
                    x: hit.x,
                    y: hit.y,
                    row_idx: hit.row_idx,
                    col_idx: hit.col_idx,
                });
            }
        }
        energized
            .into_iter()
            .flat_map(|x| x.into_iter())
            .filter(|x| *x)
            .count()
    }

    let (part1, part2) = Direction::ALL
        .into_iter()
        .flat_map(|d| (0..GRID_SIZE).map(move |i| (d, i)))
        .par_bridge()
        .map(|(dir, pos)| {
            let n = calculate_energized_cells(dir, pos, &mirrors_rows, &mirrors_cols);
            if (dir, pos) == (Direction::Right, 0) {
                (n, n)
            } else {
                (0, n)
            }
        })
        .reduce(|| (0, 0), |a, b| (a.0 + b.0, a.1.max(b.1)));
    Ok((part1, part2))
}

pub fn day17<const GRID_SIZE: usize>(input: &str) -> Result<(usize, usize)> {
    #[derive(Debug, Copy, Clone, Eq, PartialEq)]
    struct State {
        cost: u16,
        pos: (u8, u8),
        next_dir: Direction,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    enum Direction {
        Vertical,
        Horizontal,
    }
    impl Direction {
        fn idx(self) -> usize {
            match self {
                Direction::Vertical => 0,
                Direction::Horizontal => 1,
            }
        }
        fn change(self) -> Direction {
            match self {
                Direction::Vertical => Direction::Horizontal,
                Direction::Horizontal => Direction::Vertical,
            }
        }
    }

    fn solve<const GRID_SIZE: usize, const MIN_STEPS: isize, const MAX_STEPS: isize>(
        grid: &[[u8; GRID_SIZE]; GRID_SIZE],
    ) -> usize {
        // Need to keep track of shortest distance for each direction separately!
        let mut dist = [[[u16::MAX; GRID_SIZE]; GRID_SIZE]; 2];
        for dir in [Direction::Vertical, Direction::Horizontal] {
            dist[dir.idx()][0][0] = 0;
        }

        let mut bucket_queue = vec![Vec::with_capacity(32); 2048];
        bucket_queue[0].push(State {
            cost: 0,
            pos: (0, 0),
            next_dir: Direction::Horizontal,
        });
        bucket_queue[0].push(State {
            cost: 0,
            pos: (0, 0),
            next_dir: Direction::Vertical,
        });
        let mut bucket_queue_lowest = 0;

        let mut target_dir = None;
        while let Some(State {
            cost,
            pos,
            next_dir,
        }) = bucket_queue[bucket_queue_lowest].pop()
        {
            let cost = cost as usize;
            while bucket_queue[bucket_queue_lowest].is_empty() {
                bucket_queue_lowest += 1;
            }

            // This is quite likely since we just push instead of update our priority queue
            if cost > dist[next_dir.idx()][pos.0 as usize][pos.1 as usize] as usize {
                continue;
            }
            if pos == (GRID_SIZE as u8 - 1, GRID_SIZE as u8 - 1) {
                target_dir = Some(next_dir);
                break;
            }

            let new_next_dir = next_dir.change();
            for neg in [-1, 1] {
                let mut cost_sum = 0;
                for i in 1..=MAX_STEPS {
                    let delta = match next_dir {
                        Direction::Horizontal => (1, 0),
                        Direction::Vertical => (0, 1),
                    };
                    let new_pos = (
                        pos.0 as isize + delta.0 * i * neg,
                        pos.1 as isize + delta.1 * i * neg,
                    );
                    let range = 0..GRID_SIZE as isize;
                    if !range.contains(&new_pos.0) || !range.contains(&new_pos.1) {
                        break;
                    }

                    let new_pos = (new_pos.0 as usize, new_pos.1 as usize);
                    cost_sum += grid[new_pos.1][new_pos.0] as usize;
                    let new_cost = cost + cost_sum;

                    if i < MIN_STEPS {
                        continue;
                    }

                    if new_cost < dist[new_next_dir.idx()][new_pos.0][new_pos.1] as usize {
                        dist[new_next_dir.idx()][new_pos.0][new_pos.1] = new_cost as u16;
                        let new_state = State {
                            cost: new_cost as u16,
                            pos: (new_pos.0 as u8, new_pos.1 as u8),
                            next_dir: new_next_dir,
                        };
                        // A* improves our search a just a little bit
                        let heuristic = new_cost + GRID_SIZE - new_pos.0 + GRID_SIZE - new_pos.1;
                        bucket_queue[heuristic].push(new_state);
                        if heuristic < bucket_queue_lowest {
                            bucket_queue_lowest = heuristic;
                        }
                    }
                }
            }
        }
        dist[target_dir.unwrap().idx()][GRID_SIZE - 1][GRID_SIZE - 1] as usize
    }

    let mut grid = [[0; GRID_SIZE]; GRID_SIZE];
    for (y, l) in input.as_bytes().chunks(GRID_SIZE + 1).enumerate() {
        for (x, b) in l.iter().take(GRID_SIZE).enumerate() {
            grid[y][x] = b - b'0';
        }
    }

    let solutions: Vec<_> = [solve::<GRID_SIZE, 1, 3>, solve::<GRID_SIZE, 4, 10>]
        .into_par_iter()
        .map(|f| f(&grid))
        .collect();
    Ok((solutions[0], solutions[1]))
}

pub fn day18(input: &str) -> Result<(usize, usize)> {
    let mut vertices1 = Vec::new();
    let mut vertices2 = Vec::new();
    let mut len_sum1 = 0;
    let mut len_sum2 = 0;
    let mut pos1 = (0, 0);
    let mut pos2 = (0, 0);
    for l in input.lines() {
        let mut parts = l.split_ascii_whitespace();
        let dir1 = parts.next().unwrap();
        let len1 = parts.next().unwrap().as_bytes().parse_isize();
        let hex = &parts.next().unwrap()[2..8];

        len_sum1 += len1 as usize;
        let d1 = match dir1.as_bytes()[0] {
            b'L' => (-len1, 0),
            b'R' => (len1, 0),
            b'D' => (0, len1),
            b'U' => (0, -len1),
            _ => unreachable!(),
        };
        pos1 = (pos1.0 + d1.0, pos1.1 + d1.1);
        vertices1.push(pos1);

        let len2 = isize::from_str_radix(&hex[..5], 16)?;
        len_sum2 += len2 as usize;
        let d2 = match hex.as_bytes()[5] {
            b'2' => (-len2, 0),
            b'0' => (len2, 0),
            b'1' => (0, len2),
            b'3' => (0, -len2),
            _ => unreachable!(),
        };
        pos2 = (pos2.0 + d2.0, pos2.1 + d2.1);
        vertices2.push(pos2);
    }

    fn calculate_area(vertices: &[(isize, isize)], outer_len: usize) -> usize {
        // Shoelace formula; https://stackoverflow.com/a/451482
        let mut area = 0;
        for i in 0..vertices.len() {
            let j = (i + 1) % vertices.len();
            area += vertices[i].0 * vertices[j].1;
            area -= vertices[i].1 * vertices[j].0;
        }
        area /= 2;
        // `outer_len / 2` since half of it is already accounted for by the area calculation
        // `+ 1` since we're "loosing" one square by going around the perimeter once
        area as usize + outer_len / 2 + 1
    }

    Ok((
        calculate_area(&vertices1, len_sum1),
        calculate_area(&vertices2, len_sum2),
    ))
}

pub fn day19(input: &str) -> Result<(usize, usize)> {
    fn range_intersect(a: &RangeInclusive<u16>, b: &RangeInclusive<u16>) -> RangeInclusive<u16> {
        (*a.start()).max(*b.start())..=(*a.end()).min(*b.end())
    }

    #[derive(Clone, Copy)]
    enum Variable {
        X = 0,
        M,
        A,
        S,
    }
    #[derive(Debug, Clone, Copy)]
    enum Outcome<'a> {
        Workflow(&'a str),
        Accept,
        Reject,
    }
    enum Comparator {
        GreaterThan(u16),
        LessThan(u16),
    }

    struct Rule<'a> {
        comparison: Option<(Variable, Comparator)>,
        outcome: Outcome<'a>,
    }
    impl Rule<'_> {
        fn outcome(&self, values: [u16; 4]) -> Option<Outcome> {
            if let Some((var, cmp)) = &self.comparison {
                let val = values[*var as usize];
                match cmp {
                    Comparator::GreaterThan(n) => val > *n,
                    Comparator::LessThan(n) => val < *n,
                }
                .then_some(self.outcome)
            } else {
                Some(self.outcome)
            }
        }
        fn calculate_total_accepts(
            &self,
            mut ranges: [RangeInclusive<u16>; 4],
            workflows: &FxHashMap<&str, Workflow>,
        ) -> usize {
            if let Some((var, hit_range)) = self.hit_range() {
                ranges[var as usize] = range_intersect(&ranges[var as usize], &hit_range);
            }
            match self.outcome {
                Outcome::Workflow(wf) => workflows[wf].calculate_total_accepts(ranges, workflows),
                Outcome::Accept => ranges.iter().map(|x| x.len()).product::<usize>(),
                Outcome::Reject => 0,
            }
        }
        fn hit_range(&self) -> Option<(Variable, RangeInclusive<u16>)> {
            self.comparison.as_ref().map(|(var, cmp)| match cmp {
                Comparator::GreaterThan(n) => (*var, n + 1..=4000),
                Comparator::LessThan(n) => (*var, 1..=n - 1),
            })
        }
        fn non_hit_range(&self) -> Option<(Variable, RangeInclusive<u16>)> {
            self.comparison.as_ref().map(|(var, cmp)| match cmp {
                Comparator::LessThan(n) => (*var, *n..=4000),
                Comparator::GreaterThan(n) => (*var, 1..=*n),
            })
        }
    }

    struct Workflow<'a> {
        rules: Vec<Rule<'a>>,
    }
    impl Workflow<'_> {
        fn accepts(&self, values: [u16; 4], workflows: &FxHashMap<&str, Workflow>) -> bool {
            let outcome = self.rules.iter().find_map(|r| r.outcome(values)).unwrap();
            match outcome {
                Outcome::Workflow(wf) => workflows[wf].accepts(values, workflows),
                Outcome::Accept => true,
                Outcome::Reject => false,
            }
        }

        fn calculate_total_accepts(
            &self,
            mut ranges: [RangeInclusive<u16>; 4],
            workflows: &FxHashMap<&str, Workflow>,
        ) -> usize {
            let mut sum = 0;
            for r in &self.rules {
                sum += r.calculate_total_accepts(ranges.clone(), workflows);
                if let Some((var, non_hit_range)) = r.non_hit_range() {
                    ranges[var as usize] = range_intersect(&ranges[var as usize], &non_hit_range);
                }
            }
            sum
        }
    }

    let mut workflows = FxHashMap::default();
    let mut parsing_workflows = true;
    let mut part1 = 0;
    for l in input.lines() {
        if l.is_empty() {
            parsing_workflows = false;
            continue;
        }

        if parsing_workflows {
            let (name, rules_str) = l.trim_end_matches('}').split_once('{').unwrap();
            let mut rules = Vec::new();
            for rule_str in rules_str.split(',') {
                if let Some((cond, result)) = rule_str.split_once(':') {
                    let cond_str = cond.as_bytes();
                    let variable = match cond_str[0] {
                        b'x' => Variable::X,
                        b'm' => Variable::M,
                        b'a' => Variable::A,
                        b's' => Variable::S,
                        _ => unreachable!(),
                    };
                    let n = cond_str[2..].parse_usize() as u16;
                    let comparator = match cond_str[1] {
                        b'>' => Comparator::GreaterThan(n),
                        b'<' => Comparator::LessThan(n),
                        _ => unreachable!(),
                    };
                    let outcome = match result {
                        "A" => Outcome::Accept,
                        "R" => Outcome::Reject,
                        rule => Outcome::Workflow(rule),
                    };
                    rules.push(Rule {
                        comparison: Some((variable, comparator)),
                        outcome,
                    })
                } else {
                    let outcome = match rule_str {
                        "A" => Outcome::Accept,
                        "R" => Outcome::Reject,
                        rule => Outcome::Workflow(rule),
                    };
                    rules.push(Rule {
                        comparison: None,
                        outcome,
                    });
                }
            }
            workflows.insert(name, Workflow { rules });
        } else {
            let mut values = l
                .trim_matches(['{', '}'])
                .split(',')
                .map(|val| val[2..].as_bytes().parse_usize() as u16);
            let values: [_; 4] = array::from_fn(|_| values.next().unwrap());
            if workflows["in"].accepts(values, &workflows) {
                part1 += values.iter().sum::<u16>() as usize;
            }
        }
    }
    let part2 = workflows["in"].calculate_total_accepts(array::from_fn(|_| 1..=4000), &workflows);
    Ok((part1, part2))
}

pub fn day20(input: &str) -> Result<(usize, usize)> {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Pulse {
        High,
        Low,
    }
    #[derive(Debug)]
    enum NodeType {
        Broadcaster,
        Conjunction(HashMap<usize, Pulse>),
        FlipFlop(bool),
        Output,
    }
    #[derive(Debug)]
    struct Node {
        node_type: NodeType,
        outputs: Vec<usize>,
    }
    impl Node {
        fn handle_pulse(
            &mut self,
            src_idx: usize,
            node_idx: usize,
            pulse: Pulse,
            signal_queue: &mut VecDeque<(usize, Pulse, usize)>,
        ) {
            let output = match &mut self.node_type {
                NodeType::Broadcaster => Some(pulse),
                NodeType::Conjunction(saved_states) => {
                    saved_states.insert(src_idx, pulse);
                    Some(if saved_states.values().all(|s| *s == Pulse::High) {
                        Pulse::Low
                    } else {
                        Pulse::High
                    })
                }
                NodeType::FlipFlop(state) if pulse == Pulse::Low => {
                    *state = !*state;
                    Some(if *state { Pulse::High } else { Pulse::Low })
                }
                _ => None,
            };
            if let Some(pulse) = output {
                for out_idx in &self.outputs {
                    signal_queue.push_back((node_idx, pulse, *out_idx));
                }
            }
        }
    }

    let mut nodes = Vec::new();
    let mut name_to_idx = FxHashMap::default();
    let mut node_outputs = Vec::new();

    nodes.push(Node {
        node_type: NodeType::Output,
        outputs: Vec::new(),
    });
    name_to_idx.insert("rx", 0);
    node_outputs.push("");

    let mut conjunctions = Vec::new();
    for l in input.lines() {
        let node_type = match l.as_bytes()[0] {
            b'&' => NodeType::Conjunction(HashMap::new()),
            b'%' => NodeType::FlipFlop(false),
            b'b' => NodeType::Broadcaster,
            _ => unreachable!(),
        };
        let (name, outputs) = l[1..].split_once(" -> ").unwrap();
        node_outputs.push(outputs);
        name_to_idx.insert(name, nodes.len());
        if matches!(node_type, NodeType::Conjunction(_)) {
            conjunctions.push(nodes.len());
        }
        nodes.push(Node {
            node_type,
            outputs: Vec::new(),
        });
    }
    for (idx, outputs) in (0..nodes.len()).zip(node_outputs.iter()) {
        let mut vec = Vec::new();
        for output in outputs
            .split(", ")
            .filter(|o| !o.is_empty()) // Only needed for `rx` node
            .map(|n| name_to_idx[n])
        {
            vec.push(output);
            if let NodeType::Conjunction(saved_states) = &mut nodes[output].node_type {
                saved_states.insert(idx, Pulse::Low);
            }
        }
        nodes[idx].outputs = vec;
    }

    // Determine the sub-graph period lengths by inspecting the sub-graph
    let mut periods = Vec::new();
    for mut current in &nodes[name_to_idx["roadcaster"]].outputs {
        let mut period = 0;
        'sub_graph: for i in 0.. {
            for c in &nodes[*current].outputs {
                if matches!(nodes[*c].node_type, NodeType::FlipFlop(_)) {
                    let flip = conjunctions
                        .iter()
                        .any(|c| nodes[*c].outputs.contains(current));
                    if !flip {
                        period |= 1 << i;
                    }
                    current = c;
                    continue 'sub_graph;
                }
            }
            break;
        }
        periods.push((1 << 11) + period + 1);
    }

    let mut counts = (0, 0);
    let mut signal_queue = VecDeque::new();
    for _ in 0..1000 {
        // `b` is skipped by parser...
        signal_queue.push_back((usize::MAX, Pulse::Low, name_to_idx["roadcaster"]));
        while let Some((src_idx, pulse, dst_idx)) = signal_queue.pop_front() {
            match pulse {
                Pulse::Low => counts.0 += 1,
                Pulse::High => counts.1 += 1,
            }
            nodes[dst_idx].handle_pulse(src_idx, dst_idx, pulse, &mut signal_queue);
        }
    }

    let part1 = counts.0 * counts.1;
    let part2 = periods.iter().fold(1, |a, b| a.lcm(b));
    Ok((part1, part2))
}

// TODO: Other strategy: Do a 3x3 grid of grids; Simulate for that (adjusting parity) so the outer
//       grids have the same parity as in the large case (parity is the same across all); then
//       count the marked fields in each of border grids and apply some math for solution
pub fn day21<const GRID_SIZE: usize, const MAX_STEPS1: u16, const MAX_STEPS2: usize>(
    input: &str,
) -> Result<(usize, usize)> {
    fn access<const GRID_SIZE: usize>(slice: &[u8], pos: (isize, isize)) -> u8 {
        slice[(pos.0 * (GRID_SIZE as isize + 1) + pos.1) as usize]
    }

    fn count_cells_with_parity<const GRID_SIZE: usize>(
        distance_from_center: &[[u16; GRID_SIZE]; GRID_SIZE],
        parity: bool,
        min_distance: u16,
        max_distance: u16,
        hack: bool,
    ) -> usize {
        // TODO: Opt?
        // println!(
        //     "------ {} {} {} {}",
        //     min_distance, max_distance, parity, hack
        // );
        let mut count = 0;
        let center = GRID_SIZE / 2;
        for i in 0..GRID_SIZE {
            for j in 0..GRID_SIZE {
                if true || (i + j) % 2 == parity as usize {
                    if !hack {
                        if (distance_from_center[i][j] <= max_distance) {
                            print!("+");
                            count += 1;
                        } else {
                            print!(" ");
                        }
                    } else {
                        if i.abs_diff(center) + j.abs_diff(center) > center as usize
                            && distance_from_center[i][j] != u16::MAX
                        {
                            print!("+");
                            count += 1;
                        } else {
                            print!(" ");
                        }
                    }
                } else {
                    print!(" ");
                }
            }
            println!();
        }
        println!("count: {}", count);
        count
    }

    let input = input.as_bytes();
    let start = memchr::memchr(b'S', input).unwrap();
    let start = (start / (GRID_SIZE + 1), start % (GRID_SIZE + 1));

    // TODO: u8
    let mut distance = [[u16::MAX; GRID_SIZE]; GRID_SIZE];
    distance[start.0][start.1] = 0;
    let mut frontier = VecDeque::new();
    frontier.push_back((0, (start.0 as isize, start.1 as isize)));

    while let Some((steps, pos)) = frontier.pop_front() {
        for delta in [(0, 1), (0, -1), (1, 0), (-1, 0)] {
            let new_pos = (pos.0 + delta.0, pos.1 + delta.1);
            if (0..GRID_SIZE as isize).contains(&new_pos.0)
                && (0..GRID_SIZE as isize).contains(&new_pos.1)
                && distance[new_pos.0 as usize][new_pos.1 as usize] == u16::MAX
                && access::<GRID_SIZE>(input, new_pos) != b'#'
            {
                distance[new_pos.0 as usize][new_pos.1 as usize] = steps + 1;
                frontier.push_back((steps + 1, new_pos));
            }
        }
    }

    let part1 = count_cells_with_parity(&distance, MAX_STEPS1.is_odd(), 0, MAX_STEPS1, false);
    println!("PART2");
    let part2 = if MAX_STEPS2 == 0 {
        for row in distance {
            for c in row {
                if c == u16::MAX {
                    print!("#");
                } else if c <= MAX_STEPS1 as u16 {
                    print!("O");
                } else {
                    print!(".");
                }
            }
            println!();
        }
        for row in distance {
            for c in row {
                if c == u16::MAX {
                    print!("#");
                } else if c > MAX_STEPS1 as u16 && c != u16::MAX {
                    print!("O");
                } else {
                    print!(".");
                }
            }
            println!();
        }
        0
    } else {
        // Observations:
        // - There's a straight path from the start to the edges in 65 steps
        // - There's a diamond-shaped empty pit in the grid
        // - Looking at `reachable` from part 1 confirms that we fill out a perfect diamond in 64 steps
        // => Final filled out form is a diamond (we can always go to the center and branch out)
        // (26501365 - 65) / 131 = 202300

        // Necessary assumptions for our algorithm
        assert!(GRID_SIZE.is_odd());
        let center_to_wall = (GRID_SIZE - 1) / 2;
        assert_eq!(MAX_STEPS2 % GRID_SIZE, center_to_wall);

        // n = full grids to the right of center
        let n = (MAX_STEPS2 - center_to_wall) / GRID_SIZE;
        let (f_all_even, f_all_odd) = if n.is_even() {
            ((n + 1).pow(2), n.pow(2))
        } else {
            (n.pow(2), (n + 1).pow(2))
        };
        let parity_even = MAX_STEPS2.is_odd();
        // NOTE: even/odd referes to the position of the grid (in the grid of grids)
        let all_even = count_cells_with_parity(&distance, parity_even, 0, u16::MAX - 1, false);
        let all_odd = count_cells_with_parity(&distance, !parity_even, 0, u16::MAX - 1, false);
        let center_even =
            count_cells_with_parity(&distance, parity_even, 0, center_to_wall as u16, false);
        let center_odd =
            count_cells_with_parity(&distance, !parity_even, 0, center_to_wall as u16, false);
        let mut corners_even = all_even - center_even;
        let mut corners_odd = all_odd - center_odd;
        dbg!(corners_even, corners_odd);
        if n.is_even() {
            dbg!("even => adjusting odd");
            corners_odd = count_cells_with_parity(
                &distance,
                !parity_even,
                center_to_wall as u16 + 1,
                u16::MAX,
                true,
            );
        } else {
            dbg!("odd => adjusting even");
            corners_even = count_cells_with_parity(
                &distance,
                parity_even,
                center_to_wall as u16 + 1,
                u16::MAX,
                true,
            );
        }
        dbg!(corners_even, corners_odd);

        // let unreachable_even =
        //     count_cells_with_parity(&distance, parity_even, 0, center_to_wall as u16, true);
        // let unreachable_odd = count_cells_with_parity(
        //     &distance,
        //     !parity_even,
        //     center_to_wall as u16 + 1,
        //     u16::MAX - 1,
        //     true,
        // );
        // // assert_eq!(corners_even, corners_even_hack);
        // // assert_eq!(corners_odd, corners_odd_hack);
        // // let unreachable_even = center_even + corners_even- all_even;
        // // let unreachable_odd = center_odd + corners_odd- all_odd;
        // dbg!(unreachable_even, unreachable_odd);
        // corners_even -= unreachable_even;
        // corners_odd -= unreachable_odd;

        let (add, sub) = if n.is_even() {
            (corners_odd, corners_even)
        } else {
            (corners_even, corners_odd)
        };
        dbg!((add, sub));
        for row in distance {
            for c in row {
                if c == u16::MAX {
                    print!("#");
                } else if c <= center_to_wall as u16 {
                    print!("O");
                } else {
                    print!(".");
                }
            }
            println!();
        }
        for row in distance {
            for c in row {
                if c == u16::MAX {
                    print!("#");
                } else if c > center_to_wall as u16 && c != u16::MAX {
                    print!("O");
                } else {
                    print!(".");
                }
            }
            println!();
        }
        dbg!(f_all_even);
        dbg!(all_even);
        dbg!(f_all_odd);
        dbg!(all_odd);
        dbg!(n);
        dbg!(add);
        dbg!(n + 1);
        dbg!(sub);
        let ret = dbg!(f_all_even * all_even + f_all_odd * all_odd + n * add - (n + 1) * sub);
        // FIXME: This is a hack...
        // if GRID_SIZE == 131 {
        //     ret - n
        // } else {
        ret
        // }
    };

    Ok((part1, part2))
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_day21() -> Result<()> {
        let example = indoc! {"
            ...........
            .....###.#.
            .###.##..#.
            ..#.#...#..
            ....#.#....
            .##..S####.
            .##..#...#.
            .......##..
            .##.#.####.
            .##..##.##.
            ...........
        "};
        let example2 = indoc! {"
            ...
            .S.
            ...
        "};
        let example3 = indoc! {"
            .........
            .........
            .....#...
            .....##..
            ....S....
            .........
            .........
            .........
            .........
        "};
        let example4 = indoc! {"
            .........
            .........
            .........
            .........
            ....S....
            .........
            .#.......
            ..#......
            .........
        "};
        let example5 = indoc! {"
            .........
            .#....##.
            ...#.#...
            ..#..###.
            ....S....
            ...#.#...
            .#....##.
            .##...##.
            .........
        "};
        // Part 2 doesn't work without a special input structure...
        // assert_eq!(day21::<11, 6, 0>(example)?.0, 16);
        // assert_eq!(day21::<3, 0, 4>(example2)?.1, 5 * 5);
        // assert_eq!(day21::<3, 0, 7>(example2)?.1, 8 * 8);
        // assert_eq!(day21::<3, 0, 10>(example2)?.1, 11 * 11);
        // assert_eq!(day21::<3, 0, 7>(example2)?.1, 8 * 8);
        // assert_eq!(day21::<3, 0, 10>(example2)?.1, 11 * 11);
        // assert_eq!(execute_day(21, day21::<131, 64, 65>)?, (3689, 3802));

        fn example_grid<const N: usize>(real: &str) -> Result<()>
        where
            [(); 9 * N]:, // TODO: comment
            [(); 4 + 9 * (N / 2)]:,
            Assert<{ ((4 + 9 * (N / 2)) as u16) != 0 }>: IsTrue,
        {
            assert!(N.is_odd());
            let mut real_n = real
                .lines()
                .map(|l| l.repeat(N) + "\n")
                .collect::<String>()
                .repeat(N)
                .replace('S', " ");
            let new_center = ((9 * N + 1) * (9 * (N / 2) + 4)) + (9 * N / 2);
            real_n.replace_range(new_center..new_center + 1, "S");
            assert_eq!(
                day21::<9, 0, { 4 + 9 * (N / 2) }>(real)?.1,
                day21::<{ 9 * N }, { (4 + 9 * (N / 2)) as u16 }, 0>(&real_n)?.0
            );
            Ok(())
        }
        for ex in [example3, example4, example5] {
            println!("{}", "=".repeat(50));
            println!("{}", ex.trim());
            println!("{}", "=".repeat(50));
            println!("\n\n\n");
            example_grid::<3>(ex)?;
            println!("\n\n\n");
            example_grid::<5>(ex)?;
            println!("\n\n\n");
            example_grid::<7>(ex)?;
            println!("\n\n\n");
        }

        let real = crate::read_day_input(21);
        // TODO: unify with above
        fn real_grid<const N: usize>(real: &str) -> Result<()>
        where
            [(); 131 * N]:, // TODO: comment
            [(); 65 + 131 * (N / 2)]:,
            Assert<{ ((65 + 131 * (N / 2)) as u16) != 0 }>: IsTrue,
        {
            assert!(N.is_odd());
            let mut real_n = real
                .lines()
                .map(|l| l.repeat(N) + "\n")
                .collect::<String>()
                .repeat(N)
                .replace('S', " ");
            let new_center = ((131 * N + 1) * (131 * (N / 2) + 65)) + (131 * N / 2);
            real_n.replace_range(new_center..new_center + 1, "S");
            assert_eq!(
                day21::<131, 0, { 65 + 131 * (N / 2) }>(real)?.1,
                day21::<{ 131 * N }, { (65 + 131 * (N / 2)) as u16 }, 0>(&real_n)?.0
            );
            Ok(())
        }
        real_grid::<3>(&real)?; // Off by 1 without hack
        real_grid::<5>(&real)?; // Off by 2 without hack
        real_grid::<7>(&real)?; // Off by 3 without hack
        real_grid::<9>(&real)?; // Off by 4 without hack

        assert_eq!(
            execute_day(21, day21::<131, 64, 26501365>)?,
            (3689, 610158187362102)
        );
        Ok(())
    }
    pub enum Assert<const CHECK: bool> {}
    pub trait IsTrue {}
    impl IsTrue for Assert<true> {}

    use std::fmt::Display;

    use indoc::indoc;

    use super::*;
    use crate::*;

    fn execute_day<S: Display, T: Display>(
        n: usize,
        f: fn(&str) -> Result<(S, T)>,
    ) -> Result<(S, T)> {
        f(&read_day_input(n))
    }

    #[test]
    fn test_day1() -> Result<()> {
        let example_part2 = indoc! {"
            two1nine
            eightwothree
            abcone2threexyz
            xtwone3four
            4nineeightseven2
            zoneight234
            7pqrstsixteen
        "};
        assert_eq!(day1(example_part2)?.1, 281);
        assert_eq!(day1("twone\n")?.1, 21);
        assert_eq!(execute_day(1, day1)?, (54916, 54728));
        Ok(())
    }

    #[test]
    fn test_day2() -> Result<()> {
        assert_eq!(execute_day(2, day2)?, (2207, 62241));
        Ok(())
    }

    #[test]
    fn test_day3() -> Result<()> {
        assert_eq!(execute_day(3, day3)?, (535351, 87287096));
        Ok(())
    }

    #[test]
    fn test_day4() -> Result<()> {
        assert_eq!(execute_day(4, day4)?, (23941, 5571760));
        Ok(())
    }

    #[test]
    fn test_day5() -> Result<()> {
        let example = indoc! {"
            seeds: 79 14 55 13

            seed-to-soil map:
            50 98 2
            52 50 48

            soil-to-fertilizer map:
            0 15 37
            37 52 2
            39 0 15

            fertilizer-to-water map:
            49 53 8
            0 11 42
            42 0 7
            57 7 4

            water-to-light map:
            88 18 7
            18 25 70

            light-to-temperature map:
            45 77 23
            81 45 19
            68 64 13

            temperature-to-humidity map:
            0 69 1
            1 0 69

            humidity-to-location map:
            60 56 37
            56 93 4
        "};
        assert_eq!(day5(example)?, (35, 46));
        assert_eq!(execute_day(5, day5)?, (510109797, 9622622));
        Ok(())
    }

    #[test]
    fn test_day6() -> Result<()> {
        assert_eq!(execute_day(6, day6)?, (1083852, 23501589));
        Ok(())
    }

    #[test]
    fn test_day7() -> Result<()> {
        let example = indoc! {"
            32T3K 765
            T55J5 684
            KK677 28
            KTJJT 220
            QQQJA 483
        "};
        let jokers = indoc! {"
            2222J 1
            222JJ 10
            JJ223 100
            JJ234 1000
        "};
        assert_eq!(day7(example)?, (6440, 5905));
        assert_eq!(day7(jokers)?.1, 1234);
        assert_eq!(execute_day(7, day7)?, (241344943, 243101568));
        Ok(())
    }

    #[test]
    fn test_day8() -> Result<()> {
        assert_eq!(execute_day(8, day8)?, (18023, 14449445933179));
        Ok(())
    }

    #[test]
    fn test_day9() -> Result<()> {
        let example = indoc! {"
            0 3 6 9 12 15
            1 3 6 10 15 21
            10 13 16 21 30 45
        "};
        assert_eq!(day9(example)?, (114, 2));
        assert_eq!(execute_day(9, day9)?, (1938800261, 1112));
        Ok(())
    }

    #[test]
    fn test_day10() -> Result<()> {
        let example1 = indoc! {"
            .....
            .S-7.
            .|.|.
            .L-J.
            .....
        "};
        let example2 = indoc! {"
            ..........
            .S------7.
            .|F----7|.
            .||....||.
            .||....||.
            .|L-7F-J|.
            .|..||..|.
            .L--JL--J.
            ..........
            ..........
        "};
        let example3 = indoc! {"
            .F----7F7F7F7F-7....
            .|F--7||||||||FJ....
            .||.FJ||||||||L7....
            FJL7L7LJLJ||LJ.L-7..
            L--J.L7...LJS7F-7L7.
            ....F-J..F7FJ|L7L7L7
            ....L7.F7||L7|.L7L7|
            .....|FJLJ|FJ|F7|.LJ
            ....FJL-7.||.||||...
            ....L---J.LJ.LJLJ...
        "}
        .to_string()
            + &(".".repeat(20) + "\n").repeat(10);
        let example4 = indoc! {"
            FF7FSF7F7F7F7F7F---7
            L|LJ||||||||||||F--J
            FL-7LJLJ||||||LJL-77
            F--JF--7||LJLJ7F7FJ-
            L---JF-JLJ.||-FJLJJ7
            |F|F-JF---7F7-L7L|7|
            |FFJF7L7F-JF7|JL---7
            7-L-JL7||F7|L7F-7F7|
            L.L7LFJ|||||FJL7||LJ
            L7JLJL-JLJLJL--JLJ.L
        "}
        .to_string()
            + &(".".repeat(20) + "\n").repeat(10);
        assert_eq!(day10::<5>(example1)?, (4, 1));
        assert_eq!(day10::<10>(example2)?.1, 4);
        assert_eq!(day10::<20>(&example3)?.1, 8);
        assert_eq!(day10::<20>(&example4)?.1, 10);
        assert_eq!(execute_day(10, day10::<140>)?, (6733, 435));
        Ok(())
    }

    #[test]
    fn test_day11() -> Result<()> {
        let example = indoc! {"
            ...#......
            .......#..
            #.........
            ..........
            ......#...
            .#........
            .........#
            ..........
            .......#..
            #...#.....
        "};
        assert_eq!(day11::<10, 100>(example)?, (374, 8410));
        assert_eq!(
            execute_day(11, day11::<140, 1000000>)?,
            (9233514, 363293506944)
        );
        Ok(())
    }

    #[test]
    fn test_day12() -> Result<()> {
        let examples = [
            ("? 1", (1, 1)),
            ("# 1", (1, 1)),
            (". 1", (0, 0)),
            (".??..??...?##. 1,1,3", (4, 16384)),
            ("?#?#?#?#?#?#?#? 1,3,1,6", (1, 1)),
            ("????.#...#... 4,1,1", (1, 16)),
            ("????.######..#####. 1,6,5", (4, 2500)),
            ("?###???????? 3,2,1", (10, 506250)),
            ("?????.??##????????? 2,6,2", (48, 3011657374)),
        ];
        for (example, expected) in examples {
            println!("-----------");
            println!("Example: {}", example);
            assert_eq!(day12(example)?, expected);
        }
        assert_eq!(execute_day(12, day12)?, (7674, 4443895258186));
        Ok(())
    }

    #[test]
    fn test_day13() -> Result<()> {
        let example = indoc! {"
            #.##..##.
            ..#.##.#.
            ##......#
            ##......#
            ..#.##.#.
            ..##..##.
            #.#.##.#.

            #...##..#
            #....#..#
            ..##..###
            #####.##.
            #####.##.
            ..##..###
            #....#..#
        "};
        assert_eq!(day13(example)?, (405, 400));
        assert_eq!(execute_day(13, day13)?, (41859, 30842));
        Ok(())
    }

    #[test]
    fn test_day14() -> Result<()> {
        let example = indoc! {"
            O....#....
            O.OO#....#
            .....##...
            OO.#O....O
            .O.....O#.
            O.#..O.#.#
            ..O..#O..O
            .......O..
            #....###..
            #OO..#....
        "};
        assert_eq!(day14::<10>(example)?, (136, 64));
        assert_eq!(execute_day(14, day14::<100>)?, (109596, 96105));
        Ok(())
    }

    #[test]
    fn test_day15() -> Result<()> {
        let example = "rn=1,cm-,qp=3,cm=2,qp-,pc=4,ot=9,ab=5,pc-,pc=6,ot=7\n";
        assert_eq!(day15(example)?, (1320, 145));
        assert_eq!(execute_day(15, day15)?, (517315, 247763));
        Ok(())
    }

    #[test]
    fn test_day16() -> Result<()> {
        let example = indoc! {r"
            .|...\....
            |.-.\.....
            .....|-...
            ........|.
            ..........
            .........\
            ..../.\\..
            .-.-/..|..
            .|....-|.\
            ..//.|....
        "};
        assert_eq!(day16::<10>(example)?, (46, 51));
        assert_eq!(execute_day(16, day16::<110>)?, (8098, 8335));
        Ok(())
    }

    #[test]
    fn test_day17() -> Result<()> {
        let example = indoc! {"
            2413432311323
            3215453535623
            3255245654254
            3446585845452
            4546657867536
            1438598798454
            4457876987766
            3637877979653
            4654967986887
            4564679986453
            1224686865563
            2546548887735
            4322674655533
        "};

        assert_eq!(day17::<13>(example)?, (102, 94));
        assert_eq!(execute_day(17, day17::<141>)?, (902, 1073));
        Ok(())
    }

    #[test]
    fn test_day18() -> Result<()> {
        let example = indoc! {"
            R 6 (#70c710)
            D 5 (#0dc571)
            L 2 (#5713f0)
            D 2 (#d2c081)
            R 2 (#59c680)
            D 2 (#411b91)
            L 5 (#8ceee2)
            U 2 (#caa173)
            L 1 (#1b58a2)
            U 2 (#caa171)
            R 2 (#7807d2)
            U 3 (#a77fa3)
            L 2 (#015232)
            U 2 (#7a21e3)
        "};
        assert_eq!(day18(example)?, (62, 952408144115));
        assert_eq!(execute_day(18, day18)?, (66993, 177243763226648));
        Ok(())
    }

    #[test]
    fn test_day19() -> Result<()> {
        assert_eq!(execute_day(19, day19)?, (332145, 136661579897555));
        Ok(())
    }

    #[test]
    fn test_day20() -> Result<()> {
        let example = indoc! {"
            broadcaster -> a, b, c
            %a -> b
            %b -> c
            %c -> inv
            &inv -> a
        "};
        assert_eq!(day20(example)?.0, 32000000);
        assert_eq!(execute_day(20, day20)?, (879834312, 243037165713371));
        Ok(())
    }
}
