use std::{
    fmt::Display,
    time::{Duration, Instant},
};

use anyhow::Result;

use aoc2023::ALL_SOLUTIONS;

fn main() -> Result<()> {
    // Initialize Rayon's global thread pool in advance so that doesn't influence our timings.
    rayon::ThreadPoolBuilder::new().build_global()?;

    let mut timings = Vec::new();
    for (day, day_fn) in ALL_SOLUTIONS.iter().enumerate() {
        let time = execute_day(day + 1, *day_fn)?;
        timings.push((day, time));
    }

    let total = timings.iter().map(|x| x.1).sum::<Duration>();
    println!("Total processing time: {}", format_duration(&total));
    timings.sort_by_key(|x| x.1);
    timings.reverse();
    for (day, time) in &timings {
        println!("- Day {:2}: {}", day + 1, format_duration(time));
    }
    Ok(())
}

fn format_duration(dur: &Duration) -> String {
    if dur.as_millis() != 0 {
        format!("{}.{:03} ms", dur.as_millis(), dur.as_micros() % 1000)
    } else {
        format!("{} us", dur.as_micros())
    }
}

fn execute_day<S: Display, T: Display>(
    n: usize,
    f: fn(&str) -> Result<(S, T)>,
) -> Result<Duration> {
    println!("Day {}:", n);
    let input = aoc2023::read_day_input(n);

    let start = Instant::now();
    let (part1, part2) = f(input.as_ref())?;
    let elapsed = start.elapsed();

    println!("  Part 1: {}", part1);
    println!("  Part 2: {}", part2);
    println!("  Finished in {}", format_duration(&elapsed));
    println!("---------------------");
    Ok(elapsed)
}
