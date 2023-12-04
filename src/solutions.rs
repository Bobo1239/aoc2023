#![allow(clippy::needless_range_loop)]

use aho_corasick::AhoCorasick;
use anyhow::Result;
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
            let num = aoc2023::parse_u32_from_bytes(cap.as_bytes());
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
    let mut card_amounts = [1; 216];
    for (i, l) in input.lines().enumerate() {
        let (my_nums, win_nums) = l.split_at(offset_bar);
        let my_nums = &my_nums[offset_colon + 2..];
        let win_nums = &win_nums[2..];

        // Numbers only go up to 99 so can use a u128 bitfield
        let mut my = 0u128;
        let mut win = 0u128;
        for chunk in my_nums.as_bytes().chunks(3) {
            let n = aoc2023::parse_u32_from_bytes(chunk.trim_ascii());
            my |= 1u128 << n;
        }
        for chunk in win_nums.as_bytes().chunks(3) {
            let n = aoc2023::parse_u32_from_bytes(chunk.trim_ascii());
            win |= 1u128 << n;
        }

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
}
