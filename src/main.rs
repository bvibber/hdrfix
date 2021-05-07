use std::cmp::Ordering;
use std::fs::File;
use std::io;
use std::num;
use std::path::Path;

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

// Color fun
use oklab::{Oklab, linear_srgb_to_oklab, oklab_to_linear_srgb};

#[derive(Copy, Clone, Debug)]
enum Level {
    Scalar(f32),
    Percentile(f32),
}

impl Level {
    fn with_str(source: &str) -> Result<Self> {
        match source.strip_suffix("%") {
            Some(val) => Ok(Self::Percentile(val.parse()?)),
            None => Ok(Self::Scalar(source.parse::<f32>()?)),
        }
    }
}

struct Options {
    sdr_white: f32,
    hdr_max: f32,
    tone_map: fn(Vec3, &Options) -> Vec3,
    levels_min: Level,
    levels_max: Level,
    color_map: fn(Vec3) -> Vec3,
}

enum PixelFormat {
    SDR8bit,
    HDR8bit,
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
            HDRFloat32 => 16,
        };
        let read_rgb_func = match format {
            SDR8bit => read_srgb_rgb24,
            HDR8bit => read_rec2100_rgb24,
            HDRFloat32 => read_scrgb_rgb128float
        };
        let write_rgb_func = match format {
            SDR8bit => write_srgb_rgb24,
            HDR8bit => write_rec2100_rgb24,
            HDRFloat32 => write_scrgb_rgb128float
        };
        let stride = width * bytes_per_pixel;
        let size = stride * height;
        let mut data = Vec::<u8>::with_capacity(size);
        data.resize(size, 0);
        PixelBuffer {
            width: width,
            height: height,
            bytes_per_pixel: bytes_per_pixel,
            data: data,
            read_rgb_func: read_rgb_func,
            write_rgb_func: write_rgb_func
        }
    }

    fn bytes(&self) -> &[u8] {
        &self.data
    }

    fn bytes_mut(&mut self) -> &mut[u8] {
        &mut self.data
    }

    fn par_iter_bytes<'a>(&'a self) -> impl IndexedParallelIterator<Item = &'a [u8]> {
        self.data.par_chunks(self.bytes_per_pixel)
    }

    fn par_iter_bytes_mut<'a>(&'a mut self) -> impl IndexedParallelIterator<Item = &'a mut [u8]> {
        self.data.par_chunks_mut(self.bytes_per_pixel)
    }

    fn par_iter_rgb<'a>(&'a self) -> impl 'a + IndexedParallelIterator<Item = Vec3>
    {
        self.par_iter_bytes().map(self.read_rgb_func)
    }

    fn map<'a, F>(&'a self, func: F) -> impl 'a + IndexedParallelIterator<Item = Vec3>
    where F: (Fn(Vec3) -> Vec3) + Sync + Send + 'a
    {
        self.par_iter_rgb().map(func)
    }

    fn fill<T>(&mut self, source: T)
    where T: IndexedParallelIterator<Item = Vec3>
    {
        let write_rgb_func = self.write_rgb_func;
        self.par_iter_bytes_mut()
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

fn read_scrgb_rgb128float(data: &[u8]) -> Vec3 {
    let data_ref_f32: &f32 = unsafe {
        std::mem::transmute(&data[0])
    };
    let data_f32 = unsafe {
        std::slice::from_raw_parts(data_ref_f32, data.len())
    };
    Vec3::new(data_f32[0], data_f32[1], data_f32[2])
}

fn write_scrgb_rgb128float(data: &mut [u8], rgb: Vec3) {
    let data_ref_f32: &mut f32 = unsafe {
        std::mem::transmute(&mut data[0])
    };
    let data_f32 = &mut unsafe {
        std::slice::from_raw_parts_mut(data_ref_f32, data.len())
    };
    data_f32[0] = rgb.x;
    data_f32[1] = rgb.y;
    data_f32[2] = rgb.z;
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

    let mut buffer = PixelBuffer::new(
        info.width as usize,
        info.height as usize,
        HDR8bit
    );
    reader.next_frame(buffer.bytes_mut())?;

    Ok(buffer)
}

fn read_jxr(filename: &str)
  -> Result<PixelBuffer>
{
    use jpegxr::ImageDecode;
    use jpegxr::PixelFormat::*;
    use jpegxr::Rect;

    let input = File::open(filename)?;
    let mut decoder = ImageDecode::with_reader(input)?;

    let format = decoder.get_pixel_format()?;
    if format != PixelFormat128bppRGBAFloat {
        return Err(UnsupportedPixelFormat);
    }

    let (width, height) = decoder.get_size()?;
    let bytes_per_pixel = 16;
    let stride = width as usize * bytes_per_pixel;
    let mut buffer = PixelBuffer::new(
        width as usize,
        height as usize,
        HDRFloat32
    );

    let rect = Rect::new(0, 0, width, height);
    decoder.copy(&rect, buffer.bytes_mut(), stride)?;

    Ok(buffer)
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

fn rec2100_to_scrgb(val: Vec3) -> Vec3 {
    let matrix = Mat3::from_cols_array(&[
        1.6605, -0.1246, -0.0182,
        -0.5876, 1.1329, -0.1006,
        -0.0728, -0.0083, 1.1187
    ]);
    let scale = REC2100_MAX / SDR_WHITE;
    matrix.mul_vec3(val * scale)
}

fn luma_oklab(val: Vec3) -> f32 {
    let oklab_in = scrgb_to_oklab(val);
    // oklab's l is not linear
    // so translate it back to linear srgb desaturated
    // and take one of its rgb values
    let oklab_gray = Oklab {
        l: oklab_in.l,
        a: 0.0,
        b: 0.0,
    };
    let rgb_gray = oklab_to_scrgb(oklab_gray);
    rgb_gray.x
}

fn tonemap_linear(c_in: Vec3, _options: &Options) -> Vec3 {
    c_in
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
    let luma_in = luma_oklab(c_in);

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
    scale_oklab(c_in, luma_out)
}

fn oklab_l_for_luma(luma: f32) -> f32 {
    let gray_rgb = oklab::RGB::new(luma, luma, luma);
    let gray_oklab = linear_srgb_to_oklab(gray_rgb);
    gray_oklab.l
}

fn scale_oklab(c_in: Vec3, luma_out: f32) -> Vec3
{
    let gray_l = oklab_l_for_luma(luma_out);

    let oklab_in = scrgb_to_oklab(c_in);
    if oklab_in.l == 0.0 {
        c_in
    } else {
        let ratio = gray_l / oklab_in.l;
        let oklab_out = Oklab {
            l: gray_l,
            a: oklab_in.a * ratio,
            b: oklab_in.b * ratio,
        };
        oklab_to_scrgb(oklab_out)
    }
}

fn clip(input: Vec3) -> Vec3 {
    input.max(Vec3::ZERO).min(Vec3::ONE)
}

fn color_clip(input: Vec3) -> Vec3
{
    clip(input)
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

fn linear_to_srgb(val: Vec3) -> Vec3 {
    // fixme make sure all the splats are efficient constants
    let min = Vec3::splat(0.0031308);
    let linear = val * Vec3::splat(12.92);
    let gamma = (val * Vec3::splat(1.055)).powf(1.0 / 2.4) - Vec3::splat(0.055);
    clip(Vec3::select(val.cmple(min), linear, gamma))
}

const REC2100_MAX: f32 = 10000.0; // the 1.0 value for BT.2100 linear
const SDR_WHITE: f32 = 80.0;

fn hdr_to_sdr_pixel(rgb_scrgb: Vec3, options: &Options) -> Vec3
{
    let mut val = rgb_scrgb;
    val = val * SDR_WHITE / options.sdr_white;
    val = (options.tone_map)(val, &options);
    val = (options.color_map)(val);
    val
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
    encoder.write_image_rows(data.bytes())?;
    encoder.finish()?;

    Ok(())
}

struct Histogram {
    luma_vals: Vec<f32>,
}

impl Histogram {
    fn new(source: &PixelBuffer) -> Self {
        let mut luma_vals = Vec::<f32>::new();
        source.par_iter_rgb().map(luma_oklab).collect_into_vec(&mut luma_vals);
        luma_vals.par_sort_unstable_by(|a, b| {
            match a.partial_cmp(b) {
                Some(ordering) => ordering,
                None => Ordering::Equal,
            }
        });
        Self {
            luma_vals: luma_vals,
        }
    }

    fn percentile(&self, target: f32) -> f32 {
        let max_index = self.luma_vals.len() - 1;
        let target_index = (max_index as f32 * target / 100.0) as usize;
        self.luma_vals[target_index]
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

fn apply_levels(c_in: Vec3, level_min: f32, level_max: f32) -> Vec3 {
    let offset = level_min;
    let scale = level_max - level_min;
    let luma_in = luma_oklab(c_in);
    let luma_out = (luma_in - offset) / scale;
    scale_oklab(c_in, luma_out)
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

fn hdrfix(args: ArgMatches) -> Result<String> {
    let input_filename = args.value_of("input").unwrap();
    let source = time_func("read_input", || {
        let ext = Path::new(&input_filename).extension().unwrap().to_str().unwrap();
        match ext {
            "png" => read_png(input_filename),
            "jxr" => read_jxr(input_filename),
            _ => Err(InvalidInputFile)
        }
    })?;

    let sdr_white = args.value_of("sdr-white").unwrap().parse::<f32>()?;

    let hdr_max = match Level::with_str(args.value_of("hdr-max").unwrap())? {
        // hdr_max is in nits if scalar, so scale it back to scrgb
        Level::Scalar(val) => val / sdr_white,

        // If given a percentile for hdr_max, detect from input histogram.
        Level::Percentile(val) => {
            let histogram = time_func("input histogram", || {
                Ok(Histogram::new(&source))
            }).unwrap();
            histogram.percentile(val)
        }
    };

    let options = Options {
        sdr_white: sdr_white,
        hdr_max: hdr_max,
        tone_map: match args.value_of("tone-map").expect("tone-map arg") {
            "linear" => tonemap_linear,
            "reinhard" => tonemap_reinhard_oklab,
            _ => unreachable!("bad tone-map option")
        },
        color_map: match args.value_of("color-map").expect("color-map arg") {
            "clip" => color_clip,
            "desaturate" => color_desat_oklab,
            _ => unreachable!("bad color-map option")
        },
        levels_min: Level::with_str(args.value_of("levels-min").expect("levels-min arg"))?,
        levels_max: Level::with_str(args.value_of("levels-max").expect("levels-max arg"))?,
    };

    let width = source.width as usize;
    let height = source.height as usize;
    let mut tone_mapped = PixelBuffer::new(width, height, HDRFloat32);
    time_func("hdr_to_sdr", || {
        Ok(tone_mapped.fill(source.map(|rgb| hdr_to_sdr_pixel(rgb, &options))))
    })?;

    // apply histogram expansion and color gamut correction to output
    let mut lazy_histogram = Lazy::new(|| {
        time_func("levels histogram", || Ok(Histogram::new(&tone_mapped))).unwrap()
    });
    let levels_min = lazy_histogram.level(options.levels_min);
    let levels_max = lazy_histogram.level(options.levels_max);

    let mut dest = PixelBuffer::new(width, height, SDR8bit);
    time_func("output mapping", || {
        Ok(dest.fill(tone_mapped.map(|rgb| {
            // We have to color map again
            // in case the histogram pushed things back out of gamut.
            clip((options.color_map)(apply_levels(rgb, levels_min, levels_max)))
        })))
    })?;

    let output_filename = args.value_of("output").unwrap();
    time_func("write_png", || {
        write_png(output_filename, &dest)
    })?;

    return Ok(output_filename.to_string());
}

fn main() {
    let args = App::new("hdrfix converter for HDR screenshots")
        .version("0.1.0")
        .author("Brion Vibber <brion@pobox.com>")
        .arg(Arg::with_name("input")
            .help("Input filename, must be .jxr or .png as saved by NVIDIA capture overlay.")
            .required(true)
            .index(1))
        .arg(Arg::with_name("output")
            .help("Output filename, must be .png.")
            .required(true)
            .index(2))
        .arg(Arg::with_name("sdr-white")
            .help("SDR white point in nits, used to scale the HDR input linearly so that input value of standard 80 nits is scaled up to this value. Defaults to 80 nits, which is standard SDR calibration for a darkened room.")
            .long("sdr-white")
            .default_value("80.0"))
        .arg(Arg::with_name("tone-map")
            .help("Method for mapping HDR into SDR domain.")
            .long("tone-map")
            .possible_values(&["linear", "reinhard"])
            .default_value("reinhard"))
        .arg(Arg::with_name("hdr-max")
            .help("Max HDR luminance level for Reinhard algorithm, in nits.")
            .long("hdr-max")
            .default_value("10000"))
        .arg(Arg::with_name("levels-min")
            .help("Minimum output level to save when expanding final SDR output for saving. May be an absolute value in 0..1 range or a percentile from 0% to 100%.")
            .long("levels-min")
            .default_value("0.0"))
        .arg(Arg::with_name("levels-max")
            .help("Maximum output level to save when expanding final SDR output for saving. May be an absolute value in 0..1 range or a percentile from 0% to 100%.")
            .long("levels-max")
            .default_value("1.0"))
        .arg(Arg::with_name("color-map")
            .help("Method for mapping and fixing out of gamut colors.")
            .long("color-map")
            .possible_values(&["clip", "desaturate"])
            .default_value("desaturate"))
        .get_matches();

    match hdrfix(args) {
        Ok(outfile) => println!("Saved: {}", outfile),
        Err(e) => eprintln!("Error: {}", e),
    }
}
