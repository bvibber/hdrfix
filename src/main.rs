mod jpegxr_sys;
mod jpegxr;

use std::fs::File;
use std::io;
use std::num;

// Math bits
use glam::f32::{Mat3, Vec3};

// CLI bits
use clap::{Arg, App, ArgMatches};
use time::OffsetDateTime;

// Parallelism bits
use rayon::prelude::*;

// Error bits
use thiserror::Error;

type Result<T> = std::result::Result<T, LocalError>;

#[derive(Error, Debug)]
enum LocalError {
    #[error("I/O error")]
    IoError(#[from] io::Error),
    #[error("numeric format error")]
    ParseFloatError(#[from] num::ParseFloatError),
    #[error("PNG decoding error")]
    PNGDecodingError(#[from] png::DecodingError),
    #[error("PNG input must be in 8bpp true color")]
    PNGFormatError()
}

fn time_func<F, G>(msg: &str, func: F) -> Result<G>
    where F: FnOnce() -> Result<G>
{
    let start = OffsetDateTime::now_utc();
    let result = func()?;
    let delta = OffsetDateTime::now_utc() - start;
    println!("{} in {} ms", msg, delta.as_seconds_f64() * 1000.0);
    Ok(result)
}

// Read an input PNG and return its size and contents
// It must be a certain format (8bpp true color no alpha)
fn read_png(filename: &str)
    -> Result<(u32, u32, Vec<u8>)>
{
    use png::Decoder;
    use png::Transformations;

    let mut decoder = Decoder::new(File::open(filename)?);
    decoder.set_transformations(Transformations::IDENTITY);

    let (info, mut reader) = decoder.read_info()?;

    if info.bit_depth != png::BitDepth::Eight {
        return Err(LocalError::PNGFormatError());
    }
    if info.color_type != png::ColorType::RGB {
        return Err(LocalError::PNGFormatError());
    }

    let mut data = vec![0u8; info.buffer_size()];
    reader.next_frame(&mut data)?;

    Ok((info.width, info.height, data))
}

/*
fn read_jxr(filename: &str)
  -> Result<(u32, u32, Vec<u16>)>
{
    use jpegxr::Decoder;
    use jpegxr::PixelFormat;

    let mut decoder = Decoder::new(File::open(filename)?);

    let (info, mut reader) = decoder.read_info()?;

    let mut data = vec![0u16; info.buffer_size()];
    reader.next_frame(&mut data)?;

    Ok((info.width, info.height, data))
}
*/

struct Options {
    sdr_white: f32,
    hdr_max: f32,
    gamma: f32,
    color_map: fn(Vec3, f32) -> Vec3,
}

fn pq_to_linear(val: Vec3) -> Vec3 {
    // fixme make sure all the splats are efficient constants
    let inv_m1: f32 = 1.0 / 0.1593017578125;
    let inv_m2: f32 = 1.0 / 78.84375;
    let c1 = Vec3::splat(0.8359375);
    let c2 = Vec3::splat(18.8515625);
    let c3 = Vec3::splat(18.6875);
    let val_powered = val.powf(inv_m2);
    (Vec3::max(val_powered - c1, Vec3::ZERO)
        / (c2 - c3 * val_powered)
    ).powf(inv_m1)
}

fn bt2020_to_srgb(val: Vec3) -> Vec3 {
    let matrix = Mat3::from_cols_array(&[
        1.6605, -0.1246, -0.0182,
        -0.5876, 1.1329, -0.1006,
        -0.0728, -0.0083, 1.1187
    ]);
    matrix.mul_vec3(val)
}

const KR: f32 = 0.2126;
const KG: f32 = 0.7152;
const KB: f32 = 0.0722;

fn luma_srgb(val: Vec3) -> f32 {
    val.x * KR + val.y * KG + val.z * KB
}

fn apply_gamma(input: Vec3, gamma: f32) -> Vec3 {
    input.powf(gamma)
}

fn reinhold_tonemap(val: Vec3, white: f32) -> f32 {
    // TMO_reinhardext​(C) = C(1 + C/C_white^2​) / (1 + C)
    //
    // Do the Reinhold tone mapping on luminance.
    let luma = luma_srgb(val);
    let white2 = white * white;
    luma * (1.0 + luma / white2) / (1.0 + luma)
}

fn color_scale(input: Vec3, luma_out: f32) -> Vec3
{
    let luma_in = luma_srgb(input);
    input * (luma_out / luma_in)
}

fn clip(input: Vec3) -> Vec3 {
    input.max(Vec3::ZERO).min(Vec3::ONE)
}

fn color_clip(input: Vec3, luma_out: f32) -> Vec3
{
    clip(color_scale(input, luma_out))
}

fn color_darken(input: Vec3, luma_out: f32) -> Vec3
{
    let scaled = color_scale(input, luma_out);
    let max = scaled.max_element();
    if max > 1.0 {
        scaled / Vec3::splat(max)
    } else {
        scaled
    }
}

fn color_desaturate(c_in: Vec3, luma_out: f32) -> Vec3
{
    // algorithm of my own devise
    // only for colors out of gamut, desaturate until it matches luminance,
    // then clip anything that ends up out of bounds still (shouldn't happen)
    let scaled = color_scale(c_in, luma_out);
    let max = scaled.max_element();
    if max > 1.0 {
        let white = Vec3::splat(luma_out);
        let diff = scaled - white;
        let ratio = (max - 1.0) / max;
        let desaturated = scaled - diff * ratio;
        clip(desaturated)
    } else {
        scaled
    }
}

fn linear_to_srgb(val: Vec3) -> Vec3 {
    // fixme make sure all the splats are efficient constants
    let min = Vec3::splat(0.0031308);
    let linear = val * Vec3::splat(12.92);
    let gamma = (val * Vec3::splat(1.055)).powf(1.0 / 2.4) - Vec3::splat(0.055);
    Vec3::select(val.cmple(min), linear, gamma)
}

const BT2100_MAX: f32 = 10000.0; // the 1.0 value for BT.2100 linear

fn hdr_to_sdr_pixel(rgb_bt2100: Vec3, options: &Options) -> Vec3
{
    let scrgb_max = BT2100_MAX / options.sdr_white;
    let luminance_max = options.hdr_max / options.sdr_white;

    let mut val = rgb_bt2100;
    val = pq_to_linear(val);
    val = val * scrgb_max;
    val = bt2020_to_srgb(val);
    let luma_out = reinhold_tonemap(val, luminance_max);
    val = (options.color_map)(val, luma_out);
    val = apply_gamma(val, options.gamma);
    val = linear_to_srgb(val);
    val
}

const SCALE_OUT_8: f32 = 255.0;
const SCALE_IN_8: f32 = 1.0 / SCALE_OUT_8;

fn hdr_to_sdr(data: &mut [u8], options: &Options)
{
    let scale_in = Vec3::splat(SCALE_IN_8);
    let scale_out = Vec3::splat(SCALE_OUT_8);
    data.par_chunks_mut(3).for_each(|rgb| {
        let rgb_bt2100 = Vec3::new(rgb[0] as f32, rgb[1] as f32, rgb[2] as f32) * scale_in;
        let rgb_srgb = hdr_to_sdr_pixel(rgb_bt2100, options);
        let rgb_8 = rgb_srgb * scale_out;
        rgb[0] = rgb_8.x as u8;
        rgb[1] = rgb_8.y as u8;
        rgb[2] = rgb_8.z as u8;
    });
}

fn write_png(filename: &str,
             width: u32,
             height: u32,
             data: &[u8])
   -> Result<()>
{
    use mtpng::{CompressionLevel, Header};
    use mtpng::encoder::{Encoder, Options};
    use mtpng::ColorType;

    let writer = File::create(filename)?;

    let mut options = Options::new();
    options.set_compression_level(CompressionLevel::High)?;

    let mut header = Header::new();
    header.set_size(width, height)?;
    header.set_color(ColorType::Truecolor, 8)?;

    let mut encoder = Encoder::new(writer, &options);

    encoder.write_header(&header)?;
    encoder.write_image_rows(&data)?;
    encoder.finish()?;

    Ok(())
}

fn hdrfix(args: ArgMatches) -> Result<String> {
    let input_filename = args.value_of("input").unwrap();
    let (width, height, mut data) = time_func("read_png", || {
        read_png(input_filename)
    })?;

    let options = Options {
        sdr_white: args.value_of("sdr-white").unwrap().parse::<f32>()?,
        hdr_max: args.value_of("hdr-max").unwrap().parse::<f32>()?,
        gamma: args.value_of("gamma").unwrap().parse::<f32>()?,
        color_map: match args.value_of("color-map").unwrap() {
            "clip" => color_clip,
            "darken" => color_darken,
            "desaturate" => color_desaturate,
            _ => unreachable!("bad color option")
        },
    };
    time_func("hdr_to_sdr", || {
        Ok(hdr_to_sdr(&mut data, &options))
    })?;

    let output_filename = args.value_of("output").unwrap();
    time_func("write_png", || {
        write_png(output_filename, width, height, &data)
    })?;

    return Ok(output_filename.to_string());
}

fn main() {
    let args = App::new("hdrfix converter for HDR screenshots")
        .version("0.1.0")
        .author("Brion Vibber <brion@pobox.com>")
        .arg(Arg::with_name("input")
            .help("Input filename, must be .png as saved by Nvidia capture overlay.")
            .required(true)
            .index(1))
        .arg(Arg::with_name("output")
            .help("Output filename, must be .png.")
            .required(true)
            .index(2))
        .arg(Arg::with_name("sdr-white")
            .help("SDR white point, in nits.")
            .long("sdr-white")
            // 80 nits is the nominal SDR white point in a dark room.
            // Bright rooms often set SDR balance point brighter!
            .default_value("80"))
        .arg(Arg::with_name("hdr-max")
            .help("Max HDR luminance level to preserve, in nits.")
            .long("hdr-max")
            .default_value("10000"))
        .arg(Arg::with_name("gamma")
            .help("Gamma curve to apply on tone-mapped luminance values.")
            .long("gamma")
            .default_value("1.0"))
        .arg(Arg::with_name("color-map")
            .help("Method for mapping colors and fixing out of gamut.")
            .long("color-map")
            .possible_values(&["clip", "darken", "desaturate"])
            .default_value("desaturate"))
        .get_matches();

    match hdrfix(args) {
        Ok(outfile) => println!("Saved: {}", outfile),
        Err(e) => eprintln!("Error: {}", e),
    }
}
