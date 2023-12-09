#![allow(clippy::needless_range_loop)]

use std::{
    array,
    collections::HashMap,
    ops::Range,
    sync::atomic::{AtomicUsize, Ordering},
};

use aho_corasick::AhoCorasick;
use anyhow::Result;
use aoc2023::AsciiByteSliceExt;
use num::Integer;
use rayon::prelude::*;
use regex::bytes::Regex;

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
            .map(|c| c.trim_start().parse_usize())
            .fold(0u128, |acc, n| acc | 1 << n);
        let win = win_nums
            .as_bytes()
            .chunks(3)
            .map(|c| c.trim_start().parse_usize())
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

    let mut node_to_idx = HashMap::new();
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
                part1.store(steps, Ordering::SeqCst);
            }
            steps
        })
        .reduce(|| 1, |a: usize, b: usize| a.lcm(&b));
    Ok((part1.load(Ordering::SeqCst), part2))
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
            let mut numbers = l.as_bytes().split(|b| *b == b' ').map(|x| x.parse_isize());
            let mut numbers: [isize; 21] = std::array::from_fn(|_| numbers.next().unwrap());
            solve_problem(&mut numbers)
        })
        .fold((0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
    Ok((sums.1 as usize, sums.0 as usize))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;

    use indoc::indoc;

    fn execute_day<I: ?Sized, J: AsRef<I>, S: Display, T: Display>(
        n: usize,
        f: fn(&I) -> Result<(S, T)>,
        input_loader: fn(usize) -> J,
    ) -> Result<(S, T)> {
        f(input_loader(n).as_ref())
    }

    fn execute_day_input<I: ?Sized, S: Display, T: Display>(
        f: fn(&I) -> Result<(S, T)>,
        input: &I,
    ) -> Result<(S, T)> {
        f(input)
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
        assert_eq!(execute_day_input(day1, example_part2)?.1, 281);
        assert_eq!(execute_day_input(day1, "twone\n")?.1, 21);
        assert_eq!(execute_day(1, day1, default_input)?, (54916, 54728));
        Ok(())
    }

    #[test]
    fn test_day2() -> Result<()> {
        assert_eq!(execute_day(2, day2, default_input)?, (2207, 62241));
        Ok(())
    }

    #[test]
    fn test_day3() -> Result<()> {
        assert_eq!(execute_day(3, day3, default_input)?, (535351, 87287096));
        Ok(())
    }

    #[test]
    fn test_day4() -> Result<()> {
        assert_eq!(execute_day(4, day4, default_input)?, (23941, 5571760));
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
        assert_eq!(execute_day_input(day5, example)?, (35, 46));
        assert_eq!(execute_day(5, day5, default_input)?, (510109797, 9622622));
        Ok(())
    }

    #[test]
    fn test_day6() -> Result<()> {
        assert_eq!(execute_day(6, day6, default_input)?, (1083852, 23501589));
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
        assert_eq!(execute_day_input(day7, example)?, (6440, 5905));
        assert_eq!(execute_day_input(day7, jokers)?.1, 1234);
        assert_eq!(execute_day(7, day7, default_input)?, (241344943, 243101568));
        Ok(())
    }

    #[test]
    fn test_day8() -> Result<()> {
        assert_eq!(
            execute_day(8, day8, default_input)?,
            (18023, 14449445933179)
        );
        Ok(())
    }

    #[test]
    fn test_day9() -> Result<()> {
        let example = indoc! {"
            0 3 6 9 12 15
            1 3 6 10 15 21
            10 13 16 21 30 45
        "};
        assert_eq!(execute_day_input(day9, example)?, (114, 2));
        assert_eq!(execute_day(9, day9, default_input)?, (1938800261, 1112));
        Ok(())
    }
}
