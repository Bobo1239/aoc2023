use std::{
    fmt::Display,
    time::{Duration, Instant},
};

use anyhow::Result;

use aoc2023::ALL_SOLUTIONS;

fn main() -> Result<()> {
    // Initialize Rayon's global thread pool in advance so that doesn't influence our timings.
    rayon::ThreadPoolBuilder::new().build_global()?;

    let mut total = Duration::default();
    for (i, day) in ALL_SOLUTIONS.iter().enumerate() {
        total += execute_day(i + 1, *day, aoc2023::default_input)?;
    }
    println!("Total processing time: {}", format_duration(total));
    Ok(())
}

fn format_duration(dur: Duration) -> String {
    if dur.as_millis() != 0 {
        format!("{}.{:03} ms", dur.as_millis(), dur.as_micros() % 1000)
    } else {
        format!("{} us", dur.as_micros())
    }
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
