#![warn(clippy::all)]

use std::ffi::OsString;
use std::path::Path;
use std::sync::mpsc::channel;
use std::time::Duration;

// CLI bits
use clap::Parser;

// Directory watch bits
use notify::{DebouncedEvent, RecommendedWatcher, RecursiveMode, Watcher};

// Hdrfix library
use hdrfix::{io::*, time_func, Hdrfix, LocalError};

fn extension(input_filename: &Path) -> &str {
    input_filename.extension().unwrap().to_str().unwrap()
}

/// hdrfix converter for HDR screenshots
#[derive(Clone, Debug, PartialEq, clap::Parser)]
pub struct Args {
    /// Input filename, must be .jxr or .png as saved by NVIDIA capture overlay.
    #[clap()]
    pub input_file: Option<String>,

    /// Output filename, must be .png
    #[clap()]
    pub output_file: Option<String>,

    #[clap(flatten)]
    pub opts: Hdrfix,

    /// Watch a folder and convert any *.jxr files that appear into *-sdr.jpg versions. Provide a folder name.
    #[clap(long)]
    pub watch: Option<String>,
}

fn main() -> Result<(), LocalError> {
    // Parse command line arguments
    let args = Args::parse();

    // Handle watch mode
    match &args.watch {
        Some(folder) => {
            let (tx, rx) = channel::<DebouncedEvent>();
            let mut watcher = RecommendedWatcher::new(tx, Duration::from_secs(2))?;
            watcher.watch(folder, RecursiveMode::Recursive)?;

            loop {
                let event = rx.recv()?;
                if let DebouncedEvent::Create(input_path) = event {
                    let ext = extension(&input_path);
                    if ext == "jxr" {
                        let mut output_filename: OsString =
                            input_path.file_stem().unwrap().to_os_string();
                        output_filename.push("-sdr.jpg");
                        let output_path = input_path.with_file_name(output_filename);
                        if !output_path.exists() {
                            hdrfix(&input_path, &output_path, &args.opts)?;
                        }
                    }
                }
            }
        }
        None => {
            let input_file = args
                .input_file
                .as_ref()
                .map(Path::new)
                .expect("input filename missing");
            let output_file = args
                .output_file
                .as_ref()
                .map(Path::new)
                .expect("output filename missing");
            hdrfix(input_file, output_file, &args.opts)?;
        }
    }

    Ok(())
}

fn hdrfix(
    input_filename: &Path,
    output_filename: &Path,
    hdrfix: &Hdrfix,
) -> Result<(), LocalError> {
    println!(
        "{} -> {}",
        input_filename.to_str().unwrap(),
        output_filename.to_str().unwrap()
    );

    // Load input file
    let source = time_func("read_input", || {
        let ext = extension(input_filename);
        match ext {
            "png" => read_png(input_filename),
            "jxr" => read_jxr(input_filename),
            _ => Err(LocalError::InvalidInputFile),
        }
    })?;

    // Apply hdrfix
    let dest = hdrfix.apply(source)?;

    // Write output file
    time_func("write output", || {
        let ext = extension(output_filename);
        match ext {
            "png" => write_png(output_filename, &dest),
            "jpg" | "jpeg" => write_jpeg(output_filename, &dest),
            _ => Err(LocalError::InvalidOutputFile),
        }
    })?;

    Ok(())
}
