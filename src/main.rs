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

struct Options {
    pre_scale: f32,
    pre_gamma: f32,
    hdr_max: f32,
    desaturation_coeff: f32,
    mantiuk_c: f32,
    tone_map: fn(Vec3, &Options) -> Vec3,
    post_gamma: f32,
    post_scale: f32,
    color_map: fn(Vec3) -> Vec3,
    histogram: bool,
    histogram_min: f32,
    histogram_max: f32,
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

    fn par_iter_rgb<'a>(&'a self) -> impl 'a + IndexedParallelIterator<Item = Vec3> {
        self.par_iter_bytes().map(self.read_rgb_func)
    }

    fn par_fill<T>(&mut self, in_iter: T)
    where T: IndexedParallelIterator<Item = Vec3>
    {
        let writer = self.write_rgb_func;

        let out_iter = self.par_iter_bytes_mut();
        let iter = in_iter.zip(out_iter);

        iter.for_each(|(input_rgb, bytes_out)| {
            writer(bytes_out, input_rgb);
        });
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
    let scale = REC2100_MAX / 80.0;
    matrix.mul_vec3(val * scale)
}

const KR: f32 = 0.2126;
const KG: f32 = 0.7152;
const KB: f32 = 0.0722;

fn luma_scrgb(val: Vec3) -> f32 {
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
    let luma_in = luma_scrgb(c_in);
    let white = options.pre_scale * options.hdr_max / 80.0;
    let white2 = white * white;
    let luma_out = luma_in * (1.0 + luma_in / white2) / (1.0 + luma_in);

    // simple version, no desaturation correction
    //let c_out = c_in * (luma_out / luma_in);

    // correcting version based on equation 2 from Mantiuk 2009
    // https://vccimaging.org/Publications/Mantiuk2009CCT/Mantiuk2009CCT.pdf
    //let s = options.desaturation_coeff;
    //let c_out = (c_in / luma_in).powf(s) * luma_out;

    // correcting version based on equation 3 from Mantiuk 2009
    let s = options.desaturation_coeff;
    // https://vccimaging.org/Publications/Mantiuk2009CCT/Mantiuk2009CCT.pdf
    let c_out = (((c_in / luma_in) - Vec3::ONE) * s + Vec3::ONE) * luma_out;

    c_out
}

fn tonemap_reinhard_rgb(c_in: Vec3, options: &Options) -> Vec3 {
    // Variant that maps R, G, and B channels separately.
    // This should desaturate very bright colors gradually, but will
    // possible cause some color shift.
    let white = options.pre_scale * options.hdr_max / 80.0;
    let white2 = white * white;
    let c_out = c_in * (Vec3::ONE + c_in / white2) / (Vec3::ONE + c_in);
    c_out
}

fn tonemap_mantiuk(c_in: Vec3, options: &Options) -> Vec3 {
    // https://vccimaging.org/Publications/Mantiuk2009CCT/Mantiuk2009CCT.pdf

    // Equation 4: tone-mapping
    let c = options.mantiuk_c;
    let b = 1.0 / (options.hdr_max / 80.0);
    let l_in = luma_scrgb(c_in);
    let l_out = (l_in * b).powf(c);

    // Equation 3: color preservation
    let s = c;
    let c_out = (((c_in / l_in) - Vec3::ONE) * s + Vec3::ONE) * l_out;
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
    // or clip anything with luminance out of bounds to white
    let luma_in = luma_scrgb(c_in);
    if luma_in > 1.0 {
        Vec3::ONE
    } else {
        let max = c_in.max_element();
        if max > 1.0 {
            let white = Vec3::splat(luma_in);
            let diff = c_in - white;
            let ratio = (max - 1.0) / max;
            let desaturated = c_in - diff * ratio;
            clip(desaturated)
        } else {
            c_in
        }
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

fn hdr_to_sdr_pixel(rgb_scrgb: Vec3, options: &Options) -> Vec3
{
    let mut val = rgb_scrgb;
    val = val * options.pre_scale;
    val = apply_gamma(val, options.pre_gamma);
    val = (options.tone_map)(val, &options);
    val = apply_gamma(val, options.post_gamma);
    val = val * options.post_scale;
    val = (options.color_map)(val);
    val
}

fn hdr_to_sdr(in_data: &PixelBuffer, out_data: &mut PixelBuffer, options: &Options)
{
    out_data.par_fill(in_data.par_iter_rgb().map(|rgb| hdr_to_sdr_pixel(rgb, options)));
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
        source.par_iter_rgb().map(|rgb| luma_scrgb(rgb)).collect_into_vec(&mut luma_vals);
        luma_vals.par_sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        Self {
            luma_vals: luma_vals,
        }
    }

    fn percentile(&self, target: f32) -> f32{
        let max_index = self.luma_vals.len() - 1;
        let target_index = (max_index as f32 * target) as usize;
        self.luma_vals[target_index]
    }

    fn apply_levels(&self, source: &PixelBuffer, dest: &mut PixelBuffer, options: &Options) {
        let level_min = self.percentile(options.histogram_min);
        let level_max = self.percentile(options.histogram_max);
        let offset = level_min;
        let scale = level_max - level_min;
        dest.par_fill(source.par_iter_rgb().map(|rgb| {
            let luma_in = luma_scrgb(rgb);
            let luma_out = (luma_in - offset) / scale;
            // Note: these can go out of gamut, but applying
            // gamut mapping just pushes them to solid white,
            // and it's only at the extremes. Could take it
            // or leave it.
            rgb * (luma_out / luma_in)
        }))
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

    let options = Options {
        pre_scale: args.value_of("pre-scale").unwrap().parse::<f32>()?,
        pre_gamma: args.value_of("pre-gamma").unwrap().parse::<f32>()?,
        tone_map: match args.value_of("tone-map").unwrap() {
            "linear" => tonemap_linear,
            "reinhard-luma" => tonemap_reinhard_luma,
            "reinhard-rgb" => tonemap_reinhard_rgb,
            "mantiuk" => tonemap_mantiuk,
            _ => unreachable!("bad tone-map option")
        },
        hdr_max: args.value_of("hdr-max").unwrap().parse::<f32>()?,
        desaturation_coeff: args.value_of("desaturation-coeff").unwrap().parse::<f32>()?,
        mantiuk_c: args.value_of("mantiuk-c").unwrap().parse::<f32>()?,
        color_map: match args.value_of("color-map").unwrap() {
            "clip" => color_clip,
            "darken" => color_darken,
            "desaturate" => color_desaturate,
            _ => unreachable!("bad color-map option")
        },
        post_gamma: args.value_of("post-gamma").unwrap().parse::<f32>()?,
        post_scale: args.value_of("post-scale").unwrap().parse::<f32>()?,
        histogram: args.is_present("histogram"),
        histogram_min: args.value_of("histogram-min").unwrap().parse::<f32>()?,
        histogram_max: args.value_of("histogram-max").unwrap().parse::<f32>()?,
    };

    let width = source.width as usize;
    let height = source.height as usize;
    let format = if options.histogram {
        HDRFloat32
    } else {
        SDR8bit
    };
    let mut tone_mapped = PixelBuffer::new(width, height, format);
    time_func("hdr_to_sdr", || {
        Ok(hdr_to_sdr(&source, &mut tone_mapped, &options))
    })?;

    // apply histogram expansion
    let dest = if options.histogram {
        time_func("histogram", || {
            let mut dest = PixelBuffer::new(width, height, SDR8bit);
            let histogram = Histogram::new(&tone_mapped);
            histogram.apply_levels(&tone_mapped, &mut dest, &options);
            Ok(dest)
        })?
    } else {
        tone_mapped
    };

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
            .help("Input filename, must be .png as saved by Nvidia capture overlay.")
            .required(true)
            .index(1))
        .arg(Arg::with_name("output")
            .help("Output filename, must be .png.")
            .required(true)
            .index(2))
        .arg(Arg::with_name("pre-scale")
            .help("Multiplicative scaling on linear input.")
            .long("pre-scale")
            .default_value("1.0"))
        .arg(Arg::with_name("pre-gamma")
            .help("Gamma power to apply on linear input, after scaling.")
            .long("pre-gamma")
            .default_value("1.0"))
        .arg(Arg::with_name("tone-map")
            .help("Method for mapping HDR into SDR domain.")
            .long("tone-map")
            .possible_values(&["linear", "reinhard-luma", "reinhard-rgb", "mantiuk"])
            .default_value("reinhard-luma"))
        .arg(Arg::with_name("hdr-max")
            .help("Max HDR luminance level for Reinhard and Mantiuk algorithms, in nits.")
            .long("hdr-max")
            .default_value("10000"))
        .arg(Arg::with_name("desaturation-coeff")
            .help("Desaturation coefficient for Reinhard luminance to chroma mapping.")
            .long("desaturation-coeff")
            .default_value("1.0"))
        .arg(Arg::with_name("mantiuk-c")
            .help("The 'c' contrast compression factor for Mantiuk tone-mapping.")
            .long("mantiuk-c")
            .default_value("1.0"))
        .arg(Arg::with_name("post-gamma")
            .help("Gamma curve to apply on tone-mapped luminance values.")
            .long("post-gamma")
            .default_value("1.0"))
        .arg(Arg::with_name("post-scale")
            .help("Multiplicative scaling on linear output.")
            .long("post-scale")
            .default_value("1.0"))
        .arg(Arg::with_name("histogram")
            .help("Expand the final output contrast based on a histogram. \
            Use --histogram-min and --histogram-max to set luminance percentiles \
            (in 0..1 space) to expand from.")
            .long("histogram"))
        .arg(Arg::with_name("histogram-min")
            .help("Minimum portion of histogram levels to retain in output.")
            .long("histogram-min")
            .default_value("0.0"))
        .arg(Arg::with_name("histogram-max")
            .help("Maximum portion of histogram levels to retain in output.")
            .long("histogram-max")
            .default_value("1.0"))
        .arg(Arg::with_name("color-map")
            .help("Method for mapping and fixing out of gamut colors.")
            .long("color-map")
            .possible_values(&["clip", "darken", "desaturate"])
            .default_value("desaturate"))
        .get_matches();

    match hdrfix(args) {
        Ok(outfile) => println!("Saved: {}", outfile),
        Err(e) => eprintln!("Error: {}", e),
    }
}
