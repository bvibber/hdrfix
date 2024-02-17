#![warn(clippy::all)]

use std::cmp::Ordering;
use std::ffi::OsString;
use std::fs::File;
use std::io::{self, Write};
use std::num;
use std::path::Path;
use std::sync::mpsc::{channel, RecvError};
use std::time::Duration;

// Math bits
use glam::f32::{Mat3, Vec3};

// CLI bits
use clap::{Arg, App, ArgMatches};
use time::OffsetDateTime;

// Parallelism bits
use rayon::prelude::*;

// Directory watch bits
use notify::{DebouncedEvent, RecursiveMode, RecommendedWatcher, Watcher};

// Error bits
use thiserror::Error;

type Result<T> = std::result::Result<T, LocalError>;

// Color fun
use oklab::{Oklab, linear_srgb_to_oklab, oklab_to_linear_srgb};

// 16-bit floats
use half::prelude::*;

#[derive(Copy, Clone, Debug)]
enum Level {
    Scalar(f32),
    Percentile(f32),
}

impl Level {
    fn with_str(source: &str) -> Result<Self> {
        match source.strip_suffix('%') {
            Some(val) => Ok(Self::Percentile(val.parse()?)),
            None => Ok(Self::Scalar(source.parse::<f32>()?)),
        }
    }
}

struct Options {
    scale: f32,
    hdr_max: f32,
    saturation: f32,
    tone_map: fn(Vec3, &Options) -> Vec3,
    color_map: fn(Vec3) -> Vec3,
}

enum PixelFormat {
    SDR8bit,
    HDR8bit,
    HDR10bit,
    HDR16bit,
    HDRFloat16,
    HDRFloat32,
}
use PixelFormat::*;

// Note: currently assumes stride == width
struct PixelBuffer {
    width: usize,
    height: usize,
    bytes_per_pixel: usize,
    data: Vec::<u8>,

    // If we wanted these could be traits
    // but we don't need that level of complexity
    read_rgb_func: fn(&[u8]) -> Vec3,
    write_rgb_func: fn(&mut [u8], Vec3),
}

impl PixelBuffer {
    fn new(width: usize, height: usize, format: PixelFormat) -> Self {
        let bytes_per_pixel = match format {
            SDR8bit | HDR8bit => 3,
            HDR10bit => 4,
            HDR16bit => 6,
            HDRFloat16 => 8,
            HDRFloat32 => 16,
        };
        let read_rgb_func = match format {
            SDR8bit => read_srgb_rgb24,
            HDR8bit => read_rec2100_rgb24,
            HDR10bit => read_rec2100_rgb32101010,
            HDR16bit => read_rec2100_rgb48,
            HDRFloat16 => read_scrgb_rgb64half,
            HDRFloat32 => read_scrgb_rgb128float
        };
        let write_rgb_func = match format {
            SDR8bit => write_srgb_rgb24,
            HDR8bit => write_rec2100_rgb24,
            HDR10bit => write_rec2100_rgb32101010,
            HDR16bit => write_rec2100_rgb48,
            HDRFloat16 => write_scrgb_rgb64half,
            HDRFloat32 => write_scrgb_rgb128float
        };
        let stride = width * bytes_per_pixel;
        let size = stride * height;
        let data = vec![0u8; size];

        PixelBuffer {
            width,
            height,
            bytes_per_pixel,
            data,
            read_rgb_func,
            write_rgb_func
        }
    }

    fn bytes(&self) -> &[u8] {
        &self.data
    }

    fn bytes_mut(&mut self) -> &mut[u8] {
        &mut self.data
    }

    fn par_iter(&self) -> impl IndexedParallelIterator<Item = &[u8]> {
        self.data.par_chunks(self.bytes_per_pixel)
    }

    fn par_iter_mut(&mut self) -> impl IndexedParallelIterator<Item = &mut [u8]> {
        self.data.par_chunks_mut(self.bytes_per_pixel)
    }

    fn pixels(&self) -> impl '_ + IndexedParallelIterator<Item = Vec3>
    {
        self.par_iter().map(self.read_rgb_func)
    }

    fn fill<T>(&mut self, source: T)
    where T: IndexedParallelIterator<Item = Vec3>
    {
        let write_rgb_func = self.write_rgb_func;
        self.par_iter_mut()
            .zip(source)
            .for_each(|(dest, rgb)| write_rgb_func(dest, rgb))
    }
}

fn read_srgb_rgb24(_data: &[u8]) -> Vec3 {
    panic!("not yet implemented");
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

fn read_rec2100_rgb24(data: &[u8]) -> Vec3 {
    let scale = Vec3::splat(1.0 / 255.0);
    let rgb_rec2100 = Vec3::new(data[0] as f32, data[1] as f32, data[2] as f32) * scale;
    let rgb_linear = pq_to_linear(rgb_rec2100);
    rec2100_to_scrgb(rgb_linear)
}

fn write_rec2100_rgb24(_data: &mut [u8], _rgb: Vec3) {
    panic!("not yet implemented");
}

fn read_rec2100_rgb48(data: &[u8]) -> Vec3 {
    let r = u16::from_be_bytes([data[0], data[1]]);
    let g = u16::from_be_bytes([data[2], data[3]]);
    let b = u16::from_be_bytes([data[4], data[5]]);
    let scale = Vec3::splat(1.0 / 65535.0);
    let rgb_rec2100 = Vec3::new(r as f32, g as f32, b as f32) * scale;
    let rgb_linear = pq_to_linear(rgb_rec2100);
    rec2100_to_scrgb(rgb_linear)
}

fn write_rec2100_rgb48(_data: &mut [u8], _rgb: Vec3) {
    panic!("not yet implemented");
}

fn read_f16_ne(data: &[u8]) -> f32 {
    f16::from_ne_bytes(*data.first_chunk::<2>().unwrap()).to_f32()
}

fn read_scrgb_rgb64half(data: &[u8]) -> Vec3 {
    let r = read_f16_ne(&data[0..]);
    let g = read_f16_ne(&data[2..]);
    let b = read_f16_ne(&data[4..]);
    Vec3::new(r, g, b)
}

fn write_f16_ne(data: &mut [u8], n: f32) -> () {
    let bytes = f16::from_f32(n).to_ne_bytes();
    data[0..2].copy_from_slice(&bytes);
}

fn write_scrgb_rgb64half(data: &mut [u8], rgb: Vec3) {
    write_f16_ne(&mut data[0..], rgb.x);
    write_f16_ne(&mut data[2..], rgb.y);
    write_f16_ne(&mut data[4..], rgb.z);
}

fn read_f32_ne(data: &[u8]) -> f32 {
    f32::from_ne_bytes(*data.first_chunk::<4>().unwrap())
}

fn read_scrgb_rgb128float(data: &[u8]) -> Vec3 {
    let r = read_f32_ne(&data[0..]);
    let g = read_f32_ne(&data[4..]);
    let b = read_f32_ne(&data[8..]);
    Vec3::new(r, g, b)
}

fn write_f32_ne(data: &mut [u8], n: f32) -> () {
    let bytes = n.to_ne_bytes();
    data[0..4].copy_from_slice(&bytes);
}

fn write_scrgb_rgb128float(data: &mut [u8], rgb: Vec3) {
    write_f32_ne(&mut data[0..], rgb.x);
    write_f32_ne(&mut data[4..], rgb.y);
    write_f32_ne(&mut data[8..], rgb.z);
}

fn read_rec2100_rgb32101010(data: &[u8]) -> Vec3 {
    let data = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    let b = (data >>  0) & 0x03ff;
    let g = (data >> 10) & 0x03ff;
    let r = (data >> 20) & 0x03ff;
    let max = 1023.0f32;
    let pq = Vec3::new(r as f32 / max, g as f32 / max, b as f32 / max);
    let linear = pq_to_linear(pq);
    rec2100_to_scrgb(linear)
}

fn write_rec2100_rgb32101010(_data: &mut [u8], _rgb: Vec3) {
    panic!("not yet implemented");
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
    #[error("Invalid output file type")]
    InvalidOutputFile,
    #[error("Unsupported pixel format")]
    UnsupportedPixelFormat,
    #[error("Folder watch error")]
    NotifyError(#[from] notify::Error),
    #[error("Recv error")]
    RecvError(#[from] RecvError),
    #[error("Image format error")]
    ImageError(#[from] image::ImageError),
    #[error("JPEG write failure")]
    JpegWriteFailure,
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
fn read_png(filename: &Path)
    -> Result<PixelBuffer>
{
    use png::Decoder;
    use png::Transformations;

    let mut decoder = Decoder::new(File::open(filename)?);
    decoder.set_transformations(Transformations::IDENTITY);

    let mut reader = decoder.read_info()?;
    let info = reader.info();

    let format = match (info.bit_depth, info.color_type) {
        (png::BitDepth::Eight, png::ColorType::Rgb) => HDR8bit,
        (png::BitDepth::Sixteen, png::ColorType::Rgb) => HDR16bit,
        (_, _) => {
            return Err(PNGFormatError);
        }
    };

    let mut buffer = PixelBuffer::new(
        info.width as usize,
        info.height as usize,
        format
    );
    reader.next_frame(buffer.bytes_mut())?;

    Ok(buffer)
}

fn read_jxr(filename: &Path)
  -> Result<PixelBuffer>
{
    use jpegxr::ImageDecode;
    use jpegxr::PixelFormat::*;
    use jpegxr::Rect;

    let input = File::open(filename)?;
    let mut decoder = ImageDecode::with_reader(input)?;

    let (width, height) = decoder.get_size()?;
    let format = decoder.get_pixel_format()?;
    let (bytes_per_pixel, buf_fmt) = match format {
        PixelFormat128bppRGBAFloat => {
            (16, HDRFloat32)
        },
        PixelFormat64bppRGBAHalf => {
            (8, HDRFloat16)
        },
        PixelFormat32bppRGB101010 => {
            (4, HDR10bit)
        }
        _ => {
            println!("Pixel format: {:?}", format);
            return Err(UnsupportedPixelFormat);
        }
    };

    let stride = width as usize * bytes_per_pixel;
    let mut buffer = PixelBuffer::new(
        width as usize,
        height as usize,
        buf_fmt
    );

    let rect = Rect::new(0, 0, width, height);
    decoder.copy(&rect, buffer.bytes_mut(), stride)?;

    Ok(buffer)
}

fn pq_to_linear(val: Vec3) -> Vec3 {
    // fixme make sure all the splats are efficient constants
    let inv_m1: f32 = 1.0 / 0.15930176;
    let inv_m2: f32 = 1.0 / 78.84375;
    let c1 = Vec3::splat(0.8359375);
    let c2 = Vec3::splat(18.851563);
    let c3 = Vec3::splat(18.6875);
    let val_powered = val.powf(inv_m2);
    (Vec3::max(val_powered - c1, Vec3::ZERO)
        / (c2 - c3 * val_powered)
    ).powf(inv_m1)
}

fn rec2100_to_scrgb(val: Vec3) -> Vec3 {
    let matrix = Mat3::from_cols_array(&[
        1.6605, -0.1246, -0.0182,
        -0.5876, 1.1329, -0.1006,
        -0.0728, -0.0083, 1.1187
    ]);
    let scale = REC2100_MAX / SDR_WHITE;
    matrix.mul_vec3(val * scale)
}

fn luma_scrgb(val: Vec3) -> f32 {
    luma_oklab(scrgb_to_oklab(val))
}

fn luma_oklab(val: Oklab) -> f32 {
    // oklab's l is not linear
    // so translate it back to linear srgb desaturated
    // and take one of its rgb values
    let oklab_gray = Oklab {
        l: val.l,
        a: 0.0,
        b: 0.0,
    };
    let rgb_gray = oklab_to_scrgb(oklab_gray);
    rgb_gray.x
}

fn tonemap_linear(c_in: Vec3, _options: &Options) -> Vec3 {
    c_in
}

fn tonemap_reinhard_rgb(c_in: Vec3, options: &Options) -> Vec3 {
    // Variant that maps R, G, and B channels separately.
    // This should desaturate very bright colors gradually, but will
    // possible cause some color shift.
    let white = options.hdr_max;
    let white2 = white * white;
    c_in * (Vec3::ONE + c_in / white2) / (Vec3::ONE + c_in)
}

fn tonemap_reinhard_oklab(c_in: Vec3, options: &Options) -> Vec3 {
    // Map luminance from HDR to SDR domain, and scale the input color
    // in oklab perceptual color space.
    //
    // oklab color space: https://bottosson.github.io/posts/oklab/
    //
    let white = options.hdr_max;
    let white2 = white * white;

    // use Oklab's L coordinate as luminance
    let oklab_in = scrgb_to_oklab(c_in);
    let luma_in = luma_oklab(oklab_in);

    // Reinhard tone-mapping algo.
    //
    // Original:
    // http://www.cmap.polytechnique.fr/%7Epeyre/cours/x2005signal/hdr_photographic.pdf
    //
    // Extended:
    // https://64.github.io/tonemapping/#reinhard
    // TMO_reinhardext​(C) = C(1 + C/C_white^2​) / (1 + C)
    //
    let luma_out = luma_in * (1.0 + luma_in / white2) / (1.0 + luma_in);
    let oklab_out = scale_oklab_desat(oklab_in, luma_out, options.saturation);
    oklab_to_scrgb(oklab_out)
}

fn oklab_l_for_luma(luma: f32) -> f32 {
    let gray_rgb = oklab::RGB::new(luma, luma, luma);
    let gray_oklab = linear_srgb_to_oklab(gray_rgb);
    gray_oklab.l
}

fn scale_oklab_desat(oklab_in: Oklab, luma_out: f32, saturation: f32) -> Oklab
{
    let l_in = oklab_in.l;
    if l_in == 0.0 {
        oklab_in
    } else {
        let l_out = oklab_l_for_luma(luma_out);
        // oklab coords scale cubically
        // 1.0 -> desaturate linearly according to luma compression ratio
        // 0.5 -> desaturate more aggressively
        // 2.0 -> saturate more aggressively
        let ratio = (l_out / l_in).powf(3.0 / saturation);
        Oklab {
            l: l_out,
            a: oklab_in.a * ratio,
            b: oklab_in.b * ratio,
        }
    }
}

fn scale_oklab(oklab_in: Oklab, luma_out: f32) -> Oklab
{
    if oklab_in.l == 0.0 {
        oklab_in
    } else {
        let gray_l = oklab_l_for_luma(luma_out);
        let ratio = gray_l / oklab_in.l;
        Oklab {
            l: gray_l,
            a: oklab_in.a * ratio,
            b: oklab_in.b * ratio,
        }
    }
}

fn clip(input: Vec3) -> Vec3 {
    input.max(Vec3::ZERO).min(Vec3::ONE)
}

fn color_clip(input: Vec3) -> Vec3
{
    clip(input)
}

fn darken_oklab(c_in: Oklab, brightness: f32) -> Vec3
{
    let c_out = Oklab {
        l: c_in.l * brightness,
        a: c_in.a * brightness,
        b: c_in.b * brightness,
    };
    oklab_to_scrgb(c_out)
}

fn desat_oklab(c_in: Oklab, saturation: f32) -> Vec3
{
    let c_out = Oklab {
        l: c_in.l,
        a: c_in.a * saturation,
        b: c_in.b * saturation,
    };
    oklab_to_scrgb(c_out)
}

const EPSILON: f32 = 0.001; // good enough for us for now

fn close_enough(a: f32, b: f32) -> Ordering {
    let delta = a - b;
    if delta.abs() < EPSILON {
        Ordering::Equal
    } else if delta < 0.0 {
        Ordering::Less
    } else {
        Ordering::Greater
    }
}

fn binary_search<I, O, F, G>(input: I, min: f32, max: f32, func: F, comparator: G) -> O
where I: Copy + Clone,
    O: Copy + Clone,
    F: Fn(I, f32) -> O,
    G: Fn(O) -> Ordering
{
    let mid = (min + max) / 2.0;
    let result = func(input, mid);
    match close_enough(min, max) {
        Ordering::Equal => result,
        _ => match comparator(result) {
            Ordering::Less => binary_search(input, mid, max, func, comparator),
            Ordering::Greater => binary_search(input, min, mid, func, comparator),
            Ordering::Equal => result,
        }
    }
}

fn color_darken_oklab(c_in: Vec3) -> Vec3
{
    let max = c_in.max_element();
    if max > 1.0 {
        let c_in_oklab = scrgb_to_oklab(c_in);
        let c_out = binary_search(c_in_oklab, 0.0, 1.0, darken_oklab, |rgb| {
            close_enough(rgb.max_element(), 1.0)
        });
        clip(c_out)
    } else {
        c_in
    }
}

fn color_desat_oklab(c_in: Vec3) -> Vec3
{
    let max = c_in.max_element();
    if max > 1.0 {
        let c_in_oklab = scrgb_to_oklab(c_in);
        let c_out = binary_search(c_in_oklab, 0.0, 1.0, desat_oklab, |rgb| {
            close_enough(rgb.max_element(), 1.0)
        });
        clip(c_out)
    } else {
        c_in
    }
}

fn luma_rgb(val: Vec3) -> f32 {
    val.x * 0.2126 + val.y * 0.7152 + val.z * 0.0722
}

fn scale_rgb(val: Vec3, luma_out: f32) -> Vec3 {
    let luma_in = luma_rgb(val);
    let scale = luma_out / luma_in;
    val * scale
}

// https://64.github.io/tonemapping/#uncharted-2
// Uncharted 2 / Hable Filmic
fn uncharted2_tonemap_partial(x: f32) -> f32
{
    const A: f32 = 0.15;
    const B: f32 = 0.50;
    const C: f32 = 0.10;
    const D: f32 = 0.20;
    const E: f32 = 0.02;
    const F: f32 = 0.30;
    ((x*(A*x+(C*B))+(D*E))/(x*(A*x+(B))+(D*F)))-(E/F)
}

fn tonemap_uncharted2(v: Vec3, _options: &Options) -> Vec3
{
    let exposure_bias: f32 = 2.0;
    let luma = luma_rgb(v);
    let curr = uncharted2_tonemap_partial(luma * exposure_bias);

    let w = 11.2f32;
    let white_scale = 1.0f32 / uncharted2_tonemap_partial(w);
    let luma_out = curr * white_scale;

    scale_rgb(v, luma_out)
}

fn tonemap_hable(val: Vec3, _options: &Options) -> Vec3
{
    // stolen from ffmpeg's vf_tonemap

    // desat
    let luma = luma_rgb(val);
    let desaturation: f32 = 2.0;
    let epsilon: f32 = 1e-6;
    let overbright = f32::max(luma - desaturation, epsilon ) / f32::max(luma, epsilon);
    let rgb_out = val * (1.0 - overbright) + luma * overbright;
    let sig_orig = f32::max(rgb_out.max_element(), epsilon);

    // hable/uncharted2
    let exposure_bias: f32 = 2.0;
    let luma = sig_orig;
    let curr = uncharted2_tonemap_partial(luma * exposure_bias);
    let w = 11.2f32;
    let white_scale = 1.0f32 / uncharted2_tonemap_partial(w);
    let sig = curr * white_scale;

    rgb_out * (sig / sig_orig)
}

// can't use glam's Mat3 as a constant literal?
type Matrix3x3 = [[f32; 3]; 3];

// https://64.github.io/tonemapping/#aces
// ACES (Academy Color Encoding System)
const ACES_INPUT_MATRIX: Matrix3x3 =
[
    [0.59719, 0.35458, 0.04823],
    [0.07600, 0.90834, 0.01566],
    [0.02840, 0.13383, 0.83777]
];

const ACES_OUTPUT_MATRIX: Matrix3x3 =
[
    [ 1.60475, -0.53108, -0.07367],
    [-0.10208,  1.10813, -0.00605],
    [-0.00327, -0.07276,  1.07602]
];

#[allow(clippy::many_single_char_names)]
fn aces_mul(m: &Matrix3x3, v: Vec3) -> Vec3
{
    let x = m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2];
    let y = m[1][0] * v[1] + m[1][1] * v[1] + m[1][2] * v[2];
    let z = m[2][0] * v[1] + m[2][1] * v[1] + m[2][2] * v[2];
    Vec3::new(x, y, z)
}

fn aces_rtt_and_odt_fit(v: Vec3) -> Vec3
{
    let a = v * (v + Vec3::splat(0.0245786)) - Vec3::splat(0.000090537);
    let b = v * (Vec3::splat(0.983729) * v + Vec3::splat(0.432951)) + Vec3::splat(0.238081);
    a / b
}

fn tonemap_aces(c_in: Vec3, _options: &Options) -> Vec3 {
    let v = c_in;
    let v = aces_mul(&ACES_INPUT_MATRIX, v);
    let v = aces_rtt_and_odt_fit(v);
    aces_mul(&ACES_OUTPUT_MATRIX, v)
}

/*
fn srgb_to_linear(val: Vec3) -> Vec3 {
    Vec3::select(
        val.cmple(Vec3::splat(0.04045)),
        val / Vec3::splat(12.92),
        ((val + Vec3::splat(0.055)) / Vec3::splat(1.055)).powf(2.4)
    )
}
*/

fn linear_to_srgb(val: Vec3) -> Vec3 {
    // fixme make sure all the splats are efficient constants
    let min = Vec3::splat(0.0031308);
    let linear = val * Vec3::splat(12.92);
    let gamma = (val * Vec3::splat(1.055)).powf(1.0 / 2.4) - Vec3::splat(0.055);
    Vec3::select(val.cmple(min), linear, gamma)
}

const REC2100_MAX: f32 = 10000.0; // the 1.0 value for BT.2100 linear
const SDR_WHITE: f32 = 80.0;

fn exposure_scale(stops: f32) -> f32
{
    2.0_f32.powf(stops)
}

fn hdr_to_sdr_pixel(rgb_scrgb: Vec3, options: &Options) -> Vec3
{
    let val = rgb_scrgb * options.scale;
    let val = (options.tone_map)(val, options);
    (options.color_map)(val)
}

fn write_png(filename: &Path, data: &PixelBuffer)
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
    encoder.write_image_rows(data.bytes())?;
    encoder.finish()?;

    Ok(())
}


fn write_jpeg(filename: &Path, data: &PixelBuffer)
   -> Result<()>
{
    // @todo allow setting jpeg quality
    // mozjpeg is much faster than image crate's encoder
    std::panic::catch_unwind(|| {
        use mozjpeg::{Compress, ColorSpace};
        let mut c = Compress::new(ColorSpace::JCS_EXT_RGB);
        c.set_size(data.width, data.height);
        c.set_quality(95.0);
        c.set_mem_dest(); // can't write direct to file?
        c.start_compress();
        if !c.write_scanlines(data.bytes()) {
            panic!("error writing scanlines");
        }
        c.finish_compress();
        let mut writer = File::create(filename).expect("error creating output file");
        let data = c.data_as_mut_slice().expect("error accessing JPEG output buffer");
        writer.write_all(data).expect("error writing output file");
    }).map_err(|_| JpegWriteFailure)
}

struct Histogram {
    luma_vals: Vec<f32>,
}

impl Histogram {
    fn new(source: &PixelBuffer) -> Self {
        // @todo maybe do a proper histogram with buckets
        // instead of sorting every pixel value
        let mut luma_vals = Vec::<f32>::new();
        source.pixels().map(luma_scrgb).collect_into_vec(&mut luma_vals);
        luma_vals.par_sort_unstable_by(|a, b| {
            match a.partial_cmp(b) {
                Some(ordering) => ordering,
                None => Ordering::Equal,
            }
        });
        Self {
            luma_vals
        }
    }

    fn percentile(&self, target: f32) -> f32 {
        let max_index = self.luma_vals.len() - 1;
        let target_index = (max_index as f64 * target as f64 / 100.0) as usize;
        self.luma_vals[target_index]
    }

    fn average_below_percentile(&self, percent: f32) -> f32 {
        let max = self.percentile(percent);
        let (sum, count) = self.luma_vals.iter().fold((0.0f32, 0usize), |(sum, count), luma| {
            if *luma > max {
                (sum, count)
            } else {
                (sum + luma, count + 1)
            }
        });
        sum / count as f32
    }
}

fn scrgb_to_linear_srgb(c: Vec3) -> oklab::RGB<f32> {
    oklab::RGB::new(c.x, c.y, c.z)
}

fn linear_srgb_to_scrgb(c: oklab::RGB<f32>) -> Vec3 {
    Vec3::new(c.r, c.g, c.b)
}

fn scrgb_to_oklab(c: Vec3) -> Oklab {
    linear_srgb_to_oklab(scrgb_to_linear_srgb(c))
}

fn oklab_to_scrgb(c: Oklab) -> Vec3 {
    linear_srgb_to_scrgb(oklab_to_linear_srgb(c))
}

fn apply_levels(c_in: Vec3, level_min: f32, level_max: f32, gamma: f32) -> Vec3 {
    let offset = level_min;
    let scale = level_max - level_min;
    let oklab_in = scrgb_to_oklab(c_in);
    let luma_in = luma_oklab(oklab_in);
    let luma_out = ((luma_in - offset) / scale).powf(gamma);
    let oklab_out = scale_oklab(oklab_in, luma_out);
    oklab_to_scrgb(oklab_out)
}

struct Lazy<T, F> where F: (FnOnce() -> T) {
    value: Option<T>,
    func: Option<F>,
}

impl<T,F> Lazy<T,F> where F: (FnOnce() -> T) {
    fn new(func: F) -> Self {
        Lazy {
            value: None,
            func: Some(func)
        }
    }

    fn force(&mut self) -> &T {
        if self.value.is_none() {
            let func = self.func.take().unwrap();
            self.value = Some(func());
        }
        self.value.as_ref().unwrap()
    }
}

impl<F> Lazy<Histogram,F> where F: (FnOnce() -> Histogram) {
    fn level(&mut self, level: Level) -> f32 {
        match level {
            Level::Scalar(val) => val,
            Level::Percentile(val) => self.force().percentile(val),
        }
    }
}

fn extension(input_filename: &Path) -> &str {
    input_filename.extension().unwrap().to_str().unwrap()
}

fn hdrfix(input_filename: &Path, output_filename: &Path, args: &ArgMatches) -> Result<()>
{
    println!("{} -> {}", input_filename.to_str().unwrap(), output_filename.to_str().unwrap());

    let source = time_func("read_input", || {
        let ext = extension(input_filename);
        match ext {
            "png" => read_png(input_filename),
            "jxr" => read_jxr(input_filename),
            _ => Err(InvalidInputFile)
        }
    })?;
    let width = source.width as usize;
    let height = source.height as usize;

    let pre_gamma: f32 = args.value_of("pre-gamma").expect("pre-gamma arg").parse()?;
    let mut pre_histogram = Lazy::new(|| Histogram::new(&source));
    let pre_levels_min = pre_histogram.level(Level::with_str(args.value_of("pre-levels-min").expect("pre-levels-min arg"))?);
    let pre_levels_max = pre_histogram.level(Level::with_str(args.value_of("pre-levels-max").expect("pre-levels-max arg"))?);
    let source = {
        let mut dest = PixelBuffer::new(width, height, PixelFormat::HDRFloat32);
        dest.fill(source.pixels().map(|rgb| apply_levels(rgb, pre_levels_min, pre_levels_max, pre_gamma)));
        dest
    };


    let mut input_histogram = Lazy::new(|| time_func("input histogram", || {
        Ok(Histogram::new(&source))
    }).unwrap());

    let exposure = args.value_of("exposure").unwrap().parse::<f32>()?;
    let auto_exposure = Level::with_str(args.value_of("auto-exposure").unwrap())?;
    let scale = exposure_scale(exposure) * 0.5 / match auto_exposure {
        Level::Scalar(level) => level,
        Level::Percentile(percent) => input_histogram.force().average_below_percentile(percent),
    };

    let hdr_max = match Level::with_str(args.value_of("hdr-max").unwrap())? {
        // hdr_max input is in nits if scalar, so scale it to scrgb
        Level::Scalar(nits) => nits / SDR_WHITE,

        // If given a percentile for hdr_max, detect from input histogram.
        Level::Percentile(val) => input_histogram.force().percentile(val),
    } * scale;

    let options = Options {
        scale,
        hdr_max,
        saturation: args.value_of("saturation").expect("saturation arg").parse()?,
        tone_map: match args.value_of("tone-map").expect("tone-map arg") {
            "linear" => tonemap_linear,
            "reinhard" => tonemap_reinhard_oklab,
            "reinhard-rgb" => tonemap_reinhard_rgb,
            "aces" => tonemap_aces,
            "uncharted2" => tonemap_uncharted2,
            "hable" => tonemap_hable,
            _ => unreachable!("bad tone-map option")
        },
        color_map: match args.value_of("color-map").expect("color-map arg") {
            "clip" => color_clip,
            "darken" => color_darken_oklab,
            "desaturate" => color_desat_oklab,
            _ => unreachable!("bad color-map option")
        },
    };

    let mut tone_mapped = PixelBuffer::new(width, height, HDRFloat32);
    time_func("hdr_to_sdr", || {
        tone_mapped.fill(source.pixels().map(|rgb| hdr_to_sdr_pixel(rgb, &options)));
        Ok(())
    })?;

    // apply histogram expansion and color gamut correction to output
    let mut lazy_histogram = Lazy::new(|| {
        time_func("levels histogram", || Ok(Histogram::new(&tone_mapped))).unwrap()
    });
    let post_levels_min = lazy_histogram.level(Level::with_str(args.value_of("post-levels-min").expect("post-levels-min arg"))?);
    let post_levels_max = lazy_histogram.level(Level::with_str(args.value_of("post-levels-max").expect("post-levels-max arg"))?);
    let post_gamma: f32 = args.value_of("post-gamma").expect("post-gamma arg").parse()?;

    let mut dest = PixelBuffer::new(width, height, SDR8bit);
    time_func("output mapping", || {
        dest.fill(tone_mapped.pixels().map(|rgb| {
            // We have to color map again
            // in case the histogram pushed things back out of gamut.
            clip((options.color_map)(apply_levels(rgb, post_levels_min, post_levels_max, post_gamma)))
        }));
        Ok(())
    })?;

    time_func("write output", || {
        let ext = extension(output_filename);
        match ext {
            "png" => write_png(output_filename, &dest),
            "jpg" | "jpeg" => write_jpeg(output_filename, &dest),
            _ => Err(InvalidOutputFile)
        }
    })?;

    Ok(())
}

fn run(args: &ArgMatches) -> Result<()> {
    match args.value_of("watch") {
        Some(folder) => {
            let (tx, rx) = channel::<DebouncedEvent>();
            let mut watcher = RecommendedWatcher::new(tx, Duration::from_secs(2))?;
            watcher.watch(folder, RecursiveMode::Recursive)?;

            loop {
                let event = rx.recv()?;
                if let DebouncedEvent::Create(input_path) = event {
                    let ext = extension(&input_path);
                    if ext == "jxr" {
                        let mut output_filename: OsString = input_path.file_stem().unwrap().to_os_string();
                        output_filename.push("-sdr.jpg");
                        let output_path = input_path.with_file_name(output_filename);
                        if !output_path.exists() {
                            hdrfix(&input_path, &output_path, args)?;
                        }
                    }
                }
            }
        },
        None => {
            let input_filename = Path::new(args.value_of("input").expect("input filename missing"));
            let output_filename = Path::new(args.value_of("output").expect("output filename missing"));
            hdrfix(input_filename, output_filename, args)
        }
    }
}

fn main() {
    let args = App::new("hdrfix converter for HDR screenshots")
        .version("1.0.7")
        .author("Brooke Vibber <bvibber@pobox.com>")
        .arg(Arg::with_name("input")
            .help("Input filename, must be .jxr or .png as saved by NVIDIA capture overlay.")
            .index(1))
        .arg(Arg::with_name("output")
            .help("Output filename, must be .png.")
            .index(2))
        .arg(Arg::with_name("auto-exposure")
            .help("Input level or percentile of input data to average to re-expose to neutral 50% mid-tone on input. Default is 0.5, which passes input through unchanged.")
            .long("auto-exposure")
            .default_value("0.5"))
        .arg(Arg::with_name("exposure")
            .help("Exposure adjustment in stops, applied after any auto exposure adjustment. May be positive or negative in stops; defaults to 0, which does not change the exposure.")
            .long("exposure")
            .default_value("0"))
        .arg(Arg::with_name("tone-map")
            .help("Method for mapping HDR into SDR domain.")
            .long("tone-map")
            .possible_values(&["linear", "reinhard", "reinhard-rgb", "aces", "uncharted2", "hable"])
            .default_value("hable"))
        .arg(Arg::with_name("hdr-max")
            .help("Max HDR luminance level for Reinhard algorithm, in nits or a percentile to be calculated from input data. The default is 100%, which represents the highest input value.")
            .long("hdr-max")
            .default_value("100%"))
        .arg(Arg::with_name("saturation")
            .help("Coefficient for how to scale saturation in tone mapping. 1.0 will desaturate linearly to the compression ratio; smaller values will desaturate more aggressively.")
            .long("saturation")
            .default_value("1"))
        .arg(Arg::with_name("color-map")
            .help("Method for mapping and fixing out of gamut colors.")
            .long("color-map")
            .possible_values(&["clip", "darken", "desaturate", "desaturate-oklab"])
            .default_value("clip"))
        .arg(Arg::with_name("pre-gamma")
            .help("Gamma power applied on input.")
            .long("pre-gamma")
            .default_value("1.0"))
        .arg(Arg::with_name("pre-levels-min")
            .help("Minimum input level to normalize to 0 when expanding input for processing. May be an absolute value in -infinity..infinity range or a percentile from 0% to 100%.")
            .long("pre-levels-min")
            .default_value("0.0"))
        .arg(Arg::with_name("pre-levels-max")
            .help("Maximum input level to normalize to 1 when expanding input for processing. May be an absolute value in -infinity..infinity range or a percentile from 0% to 100%.")
            .long("pre-levels-max")
            .default_value("1.0"))
        .arg(Arg::with_name("post-gamma")
            .help("Gamma power applied on output.")
            .long("post-gamma")
            .default_value("1.0"))
        .arg(Arg::with_name("post-levels-min")
            .help("Minimum output level to save when expanding final SDR output for saving. May be an absolute value in 0..1 range or a percentile from 0% to 100%.")
            .long("post-levels-min")
            .default_value("0.0"))
        .arg(Arg::with_name("post-levels-max")
            .help("Maximum output level to save when expanding final SDR output for saving. May be an absolute value in 0..1 range or a percentile from 0% to 100%.")
            .long("post-levels-max")
            .default_value("1.0"))
        .arg(Arg::with_name("watch")
            .help("Watch a folder and convert any *.jxr files that appear into *-sdr.jpg versions. Provide a folder name.")
            .long("watch")
            .takes_value(true))
        .get_matches();

    match run(&args) {
        Ok(_) => println!("Done."),
        Err(e) => eprintln!("Error: {}", e),
    }
}
