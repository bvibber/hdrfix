mod jpegxr_sys;
mod jpegxr;

use std::fs::File;
use std::io;
use std::num;

// Math bits
use glam::f32::{Mat3, Vec3};

// CLI bits
use clap::{Arg, App, ArgMatches};

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

fn apply_gamma(val: Vec3, gamma: f32) -> Vec3 {
    let luma_in = luma_srgb(val);
    let luma_out = luma_in.powf(gamma);
    val * luma_out / luma_in
}

fn reinhold_tonemap(val: Vec3, white: f32) -> Vec3 {
    // TMO_reinhardext​(C) = C(1 + C/C_white^2​) / (1 + C)
    //
    // Do the Reinhold tone mapping on luminance, then scale the RGB
    // values according to it. Note we may end up out of gamut.
    let luma = luma_srgb(val);
    let white2 = white * white;
    let scaled_luma = luma * (1.0 + luma / white2) / (1.0 + luma);
    let scaled_rgb = val * Vec3::splat(scaled_luma / luma);
    scaled_rgb
}

fn clamp_colors(val: Vec3) -> Vec3 {
    // If any color elements went outside the color gamut, desaturate them.
    // This will preserve contrast at the cost of color
    let max = val.max_element();
    if max > 1.0 {
        let luma = Vec3::splat(luma_srgb(val));
        val + ((max - 1.0) / max) * (luma - val)
    } else {
        val
    }.min(Vec3::ONE).max(Vec3::ZERO)
}

fn linear_to_srgb(val: Vec3) -> Vec3 {
    // fixme make sure all the splats are efficient constants
    let min = Vec3::splat(0.0031308);
    let linear = val * Vec3::splat(12.92);
    let gamma = (val * Vec3::splat(1.055)).powf(1.0 / 2.4) - Vec3::splat(0.055);
    Vec3::select(val.cmple(min), linear, gamma)
}

fn hdr_to_sdr(width: u32, height: u32, data: &mut [u8], sdr_white: f32, hdr_max: f32, gamma: f32)
{
    let scale_8bpp = 255.0;
    let bt2100_max = 10000.0; // the 1.0 value for BT.2100 linear
    let scrgb_max = bt2100_max / sdr_white;
    let luminance_max = hdr_max / sdr_white;
    // todo: paralellize these loops
    for y in 0..height {
        for x in 0..width {
            // Read the original pixel value
            let index = ((x + y * width) * 3) as usize;
            let r1 = data[index] as f32;
            let g1 = data[index + 1] as f32;
            let b1 = data[index + 2] as f32;
            let mut val = Vec3::new(r1, g1, b1) / scale_8bpp;
            val = pq_to_linear(val);
            val = val * scrgb_max;
            val = bt2020_to_srgb(val);
            val = apply_gamma(val, gamma);
            val = reinhold_tonemap(val, luminance_max);
            val = clamp_colors(val);
            val = linear_to_srgb(val);
            val = val * scale_8bpp;
            data[index] = val.x as u8;
            data[index + 1] = val.y as u8;
            data[index + 2] = val.z as u8;
        }
    }
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
    let (width, height, mut data) = read_png(input_filename)?;

    let sdr_white = args.value_of("sdr-white").unwrap().parse::<f32>()?;
    let hdr_max = args.value_of("hdr-max").unwrap().parse::<f32>()?;
    let gamma = args.value_of("gamma").unwrap().parse::<f32>()?;
    hdr_to_sdr(width, height, &mut data, sdr_white, hdr_max, gamma);

    let output_filename = args.value_of("output").unwrap();
    write_png(output_filename, width, height, &data)?;

    return Ok(output_filename.to_string());
}

fn main() {
    let args = App::new("hdrfix converter for HDR screenshots")
        .version("0.1.0")
        .author("Brion Vibber <brion@pobox.com>")
        .arg(Arg::with_name("input")
            .help("Input filename, must be .png as saved by Nvidia capture overlay")
            .required(true)
            .index(1))
        .arg(Arg::with_name("output")
            .help("Output filename, must be .png.")
            .required(true)
            .index(2))
        .arg(Arg::with_name("sdr-white")
            .help("SDR white point, in nits")
            .long("sdr-white")
            // 80 nits is the nominal SDR white point in a dark room.
            // Bright rooms often set SDR balance point brighter!
            .default_value("80"))
        .arg(Arg::with_name("hdr-max")
            .help("Max HDR luminance level to preserve, in nits")
            .long("hdr-max")
            .default_value("10000"))
        .arg(Arg::with_name("gamma")
            .help("Gamma curve to apply on linear luminance values")
            .long("gamma")
            .default_value("1.4"))
        .get_matches();

    match hdrfix(args) {
        Ok(outfile) => println!("Saved: {}", outfile),
        Err(e) => eprintln!("Error: {}", e),
    }
}
