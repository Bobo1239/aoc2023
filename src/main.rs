mod solutions;

use std::{
    fmt::Display,
    time::{Duration, Instant},
};

use anyhow::Result;

use solutions::*;

fn main() -> Result<()> {
    let mut total = Duration::default();
    let days = [day1, day2, day3, day4, day5, day6, day7];
    for (i, day) in days.into_iter().enumerate() {
        total += execute_day(i + 1, day, default_input)?;
    }
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

fn load_input(name: &str) -> String {
    std::fs::read_to_string("inputs/".to_string() + name).unwrap()
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
