mod jpegxr_sys;
mod jpegxr;

use std::fs::File;
use std::path::Path;
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

struct PixelReader {
    bytes_per_pixel: usize,
    read_rgb: fn(data: &[u8]) -> Vec3,
}

fn read_rgb24(data: &[u8]) -> Vec3 {
    let scale = Vec3::splat(1.0 / 255.0);
    let rgb_bt2100 = Vec3::new(data[0] as f32, data[1] as f32, data[2] as f32) * scale;
    let rgb_linear = pq_to_linear(rgb_bt2100);
    bt2100_to_scrgb(rgb_linear)
}

fn read_rgb128float(data: &[u8]) -> Vec3 {
    let data_ref_f32: &f32 = unsafe {
        std::mem::transmute(&data[0])
    };
    let data_f32 = unsafe {
        std::slice::from_raw_parts(data_ref_f32, data.len())
    };
    Vec3::new(data_f32[0], data_f32[1], data_f32[2])
}

struct Options {
    sdr_white: f32,
    hdr_max: f32,
    gamma: f32,
    tone_map: fn(Vec3, &Options) -> Vec3,
    color_map: fn(Vec3) -> Vec3,
}

#[derive(Error, Debug)]
enum LocalError {
    #[error("I/O error: {0}")]
    IoError(#[from] io::Error),
    #[error("numeric format error: {0}")]
    ParseFloatError(#[from] num::ParseFloatError),
    #[error("PNG decoding error: {0}")]
    PNGDecodingError(#[from] png::DecodingError),
    #[error("PNG input must be in 8bpp true color")]
    PNGFormatError,
    #[error("JPEG XR decoding error: {0}")]
    JXRError(#[from] jpegxr::JXRError),
    #[error("Invalid input file type")]
    InvalidInputFile,
    #[error("Unsupported pixel format")]
    UnsupportedPixelFormat,
}
use LocalError::*;

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
    -> Result<(u32, u32, PixelReader, Vec<u8>)>
{
    use png::Decoder;
    use png::Transformations;

    let pixel = PixelReader {
        bytes_per_pixel: 3,
        read_rgb: read_rgb24,
    };

    let mut decoder = Decoder::new(File::open(filename)?);
    decoder.set_transformations(Transformations::IDENTITY);

    let (info, mut reader) = decoder.read_info()?;

    if info.bit_depth != png::BitDepth::Eight {
        return Err(PNGFormatError);
    }
    if info.color_type != png::ColorType::RGB {
        return Err(PNGFormatError);
    }

    let mut data = vec![0u8; info.buffer_size()];
    reader.next_frame(&mut data)?;

    Ok((info.width, info.height, pixel, data))
}

fn read_jxr(filename: &str)
  -> Result<(u32, u32, PixelReader, Vec<u8>)>
{
    use jpegxr::ImageDecode;
    use jpegxr::PixelFormat;
    use jpegxr::Rect;

    let pixel = PixelReader {
        bytes_per_pixel: 16,
        read_rgb: read_rgb128float,
    };

    let input = File::open(filename)?;
    println!("creating");
    let mut decoder = ImageDecode::create(input)?;
    println!("created");

    let format = decoder.get_pixel_format()?;
    if format != PixelFormat::HDR128bppRGBAFloat {
        return Err(UnsupportedPixelFormat);
    }

    let (width, height) = decoder.get_size()?;
    let stride = width as usize * pixel.bytes_per_pixel;
    let size = stride * height as usize;
    let mut data = Vec::<u8>::with_capacity(size);
    data.resize(size, 0);

    println!("{0} {1} {2}", width, height, stride);

    let rect = Rect::new(0, 0, width, height);
    decoder.copy(&rect, &mut data, stride as u32)?;

    Ok((width as u32, height as u32, pixel, data))
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

fn bt2100_to_scrgb(val: Vec3) -> Vec3 {
    let matrix = Mat3::from_cols_array(&[
        1.6605, -0.1246, -0.0182,
        -0.5876, 1.1329, -0.1006,
        -0.0728, -0.0083, 1.1187
    ]);
    let scale = BT2100_MAX / 80.0;
    matrix.mul_vec3(val * scale)
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

fn tonemap_linear(c_in: Vec3, _options: &Options) -> Vec3 {
    c_in
}

fn tonemap_reinhard_luma(c_in: Vec3, options: &Options) -> Vec3 {
    // Map luminance from HDR to SDR domain, and scale the input color.
    //
    // Original:
    // http://www.cmap.polytechnique.fr/%7Epeyre/cours/x2005signal/hdr_photographic.pdf
    //
    // Extended:
    // https://64.github.io/tonemapping/#reinhard
    // TMO_reinhardext​(C) = C(1 + C/C_white^2​) / (1 + C)
    //
    let luma_in = luma_srgb(c_in);
    let white = options.hdr_max / options.sdr_white;
    let white2 = white * white;
    let luma_out = luma_in * (1.0 + luma_in / white2) / (1.0 + luma_in);
    let c_out = c_in * (luma_out / luma_in);
    c_out
}

fn tonemap_reinhard_rgb(c_in: Vec3, options: &Options) -> Vec3 {
    // Variant that maps R, G, and B channels separately.
    // This should desaturate very bright colors gradually, but will
    // possible cause some color shift.
    let white = options.hdr_max / options.sdr_white;
    let white2 = white * white;
    let c_out = c_in * (Vec3::ONE + c_in / white2) / (Vec3::ONE + c_in);
    c_out
}

fn clip(input: Vec3) -> Vec3 {
    input.max(Vec3::ZERO).min(Vec3::ONE)
}

fn color_clip(input: Vec3) -> Vec3
{
    clip(input)
}

fn color_darken(input: Vec3) -> Vec3
{
    let max = input.max_element();
    if max > 1.0 {
        input / Vec3::splat(max)
    } else {
        input
    }
}

fn color_desaturate(c_in: Vec3) -> Vec3
{
    // algorithm of my own devise
    // only for colors out of gamut, desaturate until it matches luminance,
    // then clip anything that ends up out of bounds still (shouldn't happen)
    let luma_out = luma_srgb(c_in);
    let luma_in = luma_srgb(c_in);
    let scaled = c_in * (luma_out / luma_in);
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
    clip(Vec3::select(val.cmple(min), linear, gamma))
}

const BT2100_MAX: f32 = 10000.0; // the 1.0 value for BT.2100 linear

fn hdr_to_sdr_pixel(rgb_scrgb: Vec3, options: &Options) -> Vec3
{
    // 1.0 in scRGB should == the SDR white level
    let scale = 80.0 / options.sdr_white;

    let mut val = rgb_scrgb;
    val = val * scale;
    val = (options.tone_map)(val, &options);
    val = (options.color_map)(val);
    val = apply_gamma(val, options.gamma);
    val = linear_to_srgb(val);
    val
}

const SCALE_OUT_8: f32 = 255.0;
const SCALE_IN_8: f32 = 1.0 / SCALE_OUT_8;

fn hdr_to_sdr(pixel: &PixelReader, in_data: &[u8], out_data: &mut [u8], options: &Options)
{
    let scale_out = Vec3::splat(SCALE_OUT_8);
    let in_iter = in_data.par_chunks(pixel.bytes_per_pixel);
    let out_iter = out_data.par_chunks_mut(3);
    let iter = in_iter.zip(out_iter);
    iter.for_each(|(rgb_in, rgb_out)| {
        let rgb_bt2100 = (pixel.read_rgb)(rgb_in);
        let rgb_srgb = hdr_to_sdr_pixel(rgb_bt2100, options);
        let rgb_8 = rgb_srgb * scale_out;
        rgb_out[0] = rgb_8.x as u8;
        rgb_out[1] = rgb_8.y as u8;
        rgb_out[2] = rgb_8.z as u8;
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
    let (width, height, pixel, mut in_data) = time_func("read_input", || {
        let ext = Path::new(&input_filename).extension().unwrap().to_str().unwrap();
        match ext {
            "png" => read_png(input_filename),
            "jxr" => read_jxr(input_filename),
            _ => Err(InvalidInputFile)
        }
    })?;

    let options = Options {
        sdr_white: args.value_of("sdr-white").unwrap().parse::<f32>()?,
        hdr_max: args.value_of("hdr-max").unwrap().parse::<f32>()?,
        gamma: args.value_of("gamma").unwrap().parse::<f32>()?,
        tone_map: match args.value_of("tone-map").unwrap() {
            "linear" => tonemap_linear,
            "reinhard-luma" => tonemap_reinhard_luma,
            "reinhard-rgb" => tonemap_reinhard_rgb,
            _ => unreachable!("bad tone-map option")
        },
        color_map: match args.value_of("color-map").unwrap() {
            "clip" => color_clip,
            "darken" => color_darken,
            "desaturate" => color_desaturate,
            _ => unreachable!("bad color-map option")
        },
    };
    let out_size = width as usize * height as usize * 3;
    let mut out_data = Vec::<u8>::with_capacity(out_size);
    out_data.resize(out_size, 0);
    time_func("hdr_to_sdr", || {
        Ok(hdr_to_sdr(&pixel, &in_data, &mut out_data, &options))
    })?;

    let output_filename = args.value_of("output").unwrap();
    time_func("write_png", || {
        write_png(output_filename, width, height, &out_data)
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
        .arg(Arg::with_name("tone-map")
            .help("Method for mapping HDR into SDR domain.")
            .long("tone-map")
            .possible_values(&["linear", "reinhard-luma", "reinhard-rgb"])
            .default_value("reinhard-luma"))
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
