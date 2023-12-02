use std::{
    fmt::Display,
    time::{Duration, Instant},
};

use aho_corasick::AhoCorasick;

use anyhow::Result;

use aoc2023::load_input;

fn main() -> Result<()> {
    let mut total = Duration::default();
    total += execute_day(1, day1, default_input)?;
    println!("Total processing time: {}", format_duration(total));
    Ok(())
}

fn format_duration(dur: Duration) -> String {
    if dur.as_millis() != 0 {
        format!("{} ms", dur.as_millis())
    } else {
        format!("{} us", dur.as_micros())
    }
}

fn default_input(n: usize) -> String {
    load_input(&format!("{}.txt", n))
}

fn execute_day<I: ?Sized, J: AsRef<I>, S: Display, T: Display>(
    n: usize,
    f: fn(&I) -> Result<(S, T)>,
    input_loader: fn(usize) -> J,
) -> Result<Duration> {
    println!("Day {}:", n);
    let input = input_loader(n);

    let start = Instant::now();
    let (part1, part2) = f(input.as_ref())?;
    let elapsed = start.elapsed();

    println!("  Part 1: {}", part1);
    println!("  Part 2: {}", part2);
    println!("  Finished in {}", format_duration(elapsed));
    println!("---------------------");
    Ok(elapsed)
}

fn day1(input: &str) -> Result<(usize, usize)> {
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
