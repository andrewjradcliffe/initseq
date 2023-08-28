// Very simple CLI example.
// This is not the intended use case, but, if desired, one can use it in this manner.

use initseq::{init_seq, Estimator};
use std::io::{BufRead, BufReader};
use std::{env, fs, io};

#[derive(Debug)]
pub enum ParseSeqError {
    FileError(io::Error),
    FloatError(std::num::ParseFloatError, usize),
}
impl From<io::Error> for ParseSeqError {
    fn from(e: io::Error) -> Self {
        ParseSeqError::FileError(e)
    }
}

fn parse_file(file: &str) -> Result<Vec<f64>, ParseSeqError> {
    let file = match fs::File::open(file) {
        Ok(file) => file,
        Err(_) => {
            let pathbuf = std::path::Path::new(file).canonicalize()?;
            fs::File::open(pathbuf)?
        }
    };
    let file = BufReader::new(file);
    let mut x: Vec<f64> = Vec::new();
    let mut lines_iter = file.lines();
    let mut line_num: usize = 0;
    while let Some(line) = lines_iter.next() {
        line_num += 1;
        let line = line?;
        match line.parse::<f64>() {
            Ok(val) => x.push(val),
            Err(e) => return Err(ParseSeqError::FloatError(e, line_num)),
        }
    }
    Ok(x)
}

fn main() {
    let mut args = env::args();
    match args.nth(1) {
        Some(s) => match parse_file(s.as_ref()) {
            Ok(x) => {
                let initial = init_seq(&x);
                println!("{:-^80}", "Asymptotic variance estimates");
                println!(
                    "Initial positive sequence: {}",
                    initial.var(Estimator::Positive)
                );
                println!(
                    "Initial monotone sequence: {}",
                    initial.var(Estimator::Monotone)
                );
                println!(
                    "Initial convex sequence: {}",
                    initial.var(Estimator::Convex)
                );
                println!("{:-^80}", "Autocovariance sequences");
                println!(
                    "Initial positive sequence: {:?}",
                    initial.gamma(Estimator::Positive)
                );
                println!(
                    "Initial monotone sequence: {:?}",
                    initial.gamma(Estimator::Monotone)
                );
                println!(
                    "Initial convex sequence: {:?}",
                    initial.gamma(Estimator::Convex)
                );
            }
            Err(e) => match e {
                ParseSeqError::FileError(io) => {
                    println!("Some io error: {:?}", io);
                }
                ParseSeqError::FloatError(fe, line_num) => {
                    println!("Encountered something which cannot be parsed as a floating point number at line number {}", line_num);
                    println!("The specific error: {:?}", fe)
                }
            },
        },
        _ => {
            println!("No file provided.")
        }
    }
}
