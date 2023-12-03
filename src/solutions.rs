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

    let mut first_part1 = 0;
    let mut last_part1 = 0;
    let mut first_part2 = 0;
    let mut last_part2 = 0;

    for mat in ac.find_overlapping_iter(input) {
        let (digit, real_digit) = match mat.pattern().as_usize() {
            0 => {
                sum_part1 += first_part1 * 10 + last_part1;
                sum_part2 += first_part2 * 10 + last_part2;
                first_part1 = 0;
                last_part1 = 0;
                first_part2 = 0;
                last_part2 = 0;
                continue;
            }
            d @ 1..=9 => (d, true),
            d => (d - 9, false),
        };

        if real_digit {
            if first_part1 == 0 {
                first_part1 = digit;
            }
            last_part1 = digit;
        }

        if first_part2 == 0 {
            first_part2 = digit;
        }
        last_part2 = digit;
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
    let re_symbol = Regex::new(r"[^\.[0-9]\n]")?;
    let re_number = Regex::new(r"([0-9]+)")?;

    for cap in re_symbol.find_iter(input.as_bytes()) {
        let row = cap.start() / (GRID_SIZE + 1);
        let col = cap.start() % (GRID_SIZE + 1);
        let cell = if cap.as_bytes() == b"*" {
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

    let mut sum = 0;
    for cap in re_number.find_iter(input.as_bytes()) {
        let row = cap.start() / (GRID_SIZE + 1);
        let col = cap.start() % (GRID_SIZE + 1);

        for i in 0..cap.len() {
            let cell = grid[row][col + i];
            if cell == Cell::None {
                continue;
            }
            let num = std::str::from_utf8(cap.as_bytes())?.parse::<usize>()?;
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
}
