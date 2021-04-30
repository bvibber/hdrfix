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

struct PixelBuffer {
    width: usize,
    height: usize,
    stride: usize,
    bytes_per_pixel: usize,
    data: Vec::<u8>,
    read_rgb_func: fn(&[u8]) -> Vec3,
    write_rgb_func: fn(&mut [u8], Vec3),
}

impl PixelBuffer {
    fn new_srgb_rgb8(width: usize, height: usize) -> Self {
        let bytes_per_pixel = 3;
        let stride = width * bytes_per_pixel;
        let size = stride * height;
        let mut data = Vec::<u8>::with_capacity(size);
        data.resize(size, 0);
        PixelBuffer {
            width: width,
            height: height,
            stride: stride,
            bytes_per_pixel: bytes_per_pixel,
            data: data,
            read_rgb_func: read_srgb_rgb24,
            write_rgb_func: write_srgb_rgb24,
        }
    }

    fn get_bytes(&self) -> &[u8] {
        &self.data
    }

    fn get_bytes_mut(&mut self) -> &mut[u8] {
        &mut self.data
    }

    // warning: these assume that stride == width
    // if that assumption needs to be broken, the fix is:
    // do a two-level iteration, start over lines
    // then over pixels in the lines
    // ideally the iterators can chain into one somehow
    // otherwise flip it around to a for_each processor
    // function that takes input and output buffers and
    // a per-pixel closure to hide the complexity
    fn par_iter(&self) -> rayon::slice::Chunks<u8> {
        self.data.par_chunks(self.bytes_per_pixel)
    }

    fn par_iter_mut(&mut self) -> rayon::slice::ChunksMut<u8> {
        self.data.par_chunks_mut(self.bytes_per_pixel)
    }

    fn read_rgb(&self, in_data: &[u8]) -> Vec3 {
        (self.read_rgb_func)(in_data)
    }

    fn write_rgb(&self, out_data: &mut [u8], rgb: Vec3) {
        (self.write_rgb_func)(out_data, rgb)
    }
}

fn read_bt2100_rgb24(data: &[u8]) -> Vec3 {
    let scale = Vec3::splat(1.0 / 255.0);
    let rgb_bt2100 = Vec3::new(data[0] as f32, data[1] as f32, data[2] as f32) * scale;
    let rgb_linear = pq_to_linear(rgb_bt2100);
    bt2100_to_scrgb(rgb_linear)
}

fn read_srgb_rgb24(data: &[u8]) -> Vec3 {
    let scale = Vec3::splat(1.0 / 255.0);
    let rgb_srgb = Vec3::new(data[0] as f32, data[1] as f32, data[2] as f32) * scale;
    srgb_to_linear(rgb_srgb)
}

fn srgb_to_linear(_val: Vec3) -> Vec3 {
    // may change these interfaces so can't happen
    panic!("not implemented, currently not used");
}

fn write_srgb_rgb24(data: &mut [u8], val: Vec3)
{
    let gamma_out = linear_to_srgb(val);
    let clipped = clip(gamma_out);
    let scaled = clipped * 255.0;
    data[0] = scaled.x as u8;
    data[1] = scaled.y as u8;
    data[2] = scaled.z as u8;
}

fn read_scrgb_rgb128float(data: &[u8]) -> Vec3 {
    let data_ref_f32: &f32 = unsafe {
        std::mem::transmute(&data[0])
    };
    let data_f32 = unsafe {
        std::slice::from_raw_parts(data_ref_f32, data.len())
    };
    Vec3::new(data_f32[0], data_f32[1], data_f32[2])
}

fn write_scrgb_rgb128float(data: &mut [u8], val: Vec3) {
    let data_ref_f32: &mut f32 = unsafe {
        std::mem::transmute(&mut data[0])
    };
    let data_f32 = unsafe {
        std::slice::from_raw_parts_mut(data_ref_f32, data.len())
    };
    data_f32[0] = val.x;
    data_f32[1] = val.y;
    data_f32[2] = val.z;
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
    -> Result<PixelBuffer>
{
    use png::Decoder;
    use png::Transformations;

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

    Ok(PixelBuffer {
        bytes_per_pixel: 3,
        width: info.width as usize,
        height: info.height as usize,
        stride: info.width as usize * 3,
        data: data,
        read_rgb_func: read_bt2100_rgb24,
        write_rgb_func: write_srgb_rgb24,
    })
}

fn read_jxr(filename: &str)
  -> Result<PixelBuffer>
{
    use jpegxr::ImageDecode;
    use jpegxr::PixelFormat;
    use jpegxr::Rect;

    let input = File::open(filename)?;
    println!("creating");
    let mut decoder = ImageDecode::create(input)?;
    println!("created");

    let format = decoder.get_pixel_format()?;
    if format != PixelFormat::HDR128bppRGBAFloat {
        return Err(UnsupportedPixelFormat);
    }

    let bytes_per_pixel = 16;
    let (width, height) = decoder.get_size()?;
    let stride = width as usize * bytes_per_pixel;
    let size = stride * height as usize;
    let mut data = Vec::<u8>::with_capacity(size);
    data.resize(size, 0);

    let rect = Rect::new(0, 0, width, height);
    decoder.copy(&rect, &mut data, stride as u32)?;

    Ok(PixelBuffer {
        bytes_per_pixel: bytes_per_pixel,
        width: width as usize,
        height: height as usize,
        stride: stride,
        data: data,
        read_rgb_func: read_scrgb_rgb128float,
        write_rgb_func: write_scrgb_rgb128float,
    })
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
    val
}

fn hdr_to_sdr(in_data: &PixelBuffer, out_data: &mut PixelBuffer, options: &Options)
{
    // hack: get this first
    let writer = out_data.write_rgb_func;

    let in_iter = in_data.par_iter();
    let out_iter = out_data.par_iter_mut();
    let iter = in_iter.zip(out_iter);

    iter.for_each(|(bytes_in, bytes_out)| {
        let rgb_bt2100 = in_data.read_rgb(bytes_in);
        let rgb_srgb = hdr_to_sdr_pixel(rgb_bt2100, options);
        writer(bytes_out, rgb_srgb);
    });
}

fn write_png(filename: &str, data: &PixelBuffer)
   -> Result<()>
{
    use mtpng::{CompressionLevel, Header};
    use mtpng::encoder::{Encoder, Options};
    use mtpng::ColorType;

    let writer = File::create(filename)?;

    let mut options = Options::new();
    options.set_compression_level(CompressionLevel::High)?;

    let mut header = Header::new();
    header.set_size(data.width as u32, data.height as u32)?;
    header.set_color(ColorType::Truecolor, 8)?;

    let mut encoder = Encoder::new(writer, &options);

    encoder.write_header(&header)?;
    encoder.write_image_rows(data.get_bytes())?;
    encoder.finish()?;

    Ok(())
}

fn hdrfix(args: ArgMatches) -> Result<String> {
    let input_filename = args.value_of("input").unwrap();
    let in_data = time_func("read_input", || {
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
    let mut out_data = PixelBuffer::new_srgb_rgb8(in_data.width, in_data.height);
    time_func("hdr_to_sdr", || {
        Ok(hdr_to_sdr(&in_data, &mut out_data, &options))
    })?;

    let output_filename = args.value_of("output").unwrap();
    time_func("write_png", || {
        write_png(output_filename, &out_data)
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
