use aho_corasick::AhoCorasick;
use anyhow::Result;

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
    for l in input.lines() {
        let mut parts = l.split(':');
        let game_id: usize = parts.next().unwrap()[5..].parse()?;
        let sets = parts.next().unwrap().split(';');
        let mut valid = true;
        let mut max = [0; 3];
        for set in sets {
            let color_cubes = set.split(',');
            for color in color_cubes {
                let mut parts = color.trim().split(' ');
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
        }
        if valid {
            valid_sum += game_id
        }
        power_sum += max.iter().product::<usize>();
    }
    Ok((valid_sum, power_sum))
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
}
