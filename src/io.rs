use std::{fs::File, io::Write, path::Path};

use crate::{LocalError, PixelBuffer, PixelFormat};

// Read an input PNG and return its size and contents
// It must be a certain format (8bpp true color no alpha)
pub fn read_png(filename: &Path) -> Result<PixelBuffer, LocalError> {
    use png::Decoder;
    use png::Transformations;

    let mut decoder = Decoder::new(File::open(filename)?);
    decoder.set_transformations(Transformations::IDENTITY);

    let mut reader = decoder.read_info()?;
    let info = reader.info();

    if info.bit_depth != png::BitDepth::Eight {
        return Err(LocalError::PNGFormatError);
    }
    if info.color_type != png::ColorType::Rgb {
        return Err(LocalError::PNGFormatError);
    }

    let mut buffer = PixelBuffer::new(
        info.width as usize,
        info.height as usize,
        PixelFormat::HDR8bit,
    );
    reader.next_frame(buffer.bytes_mut())?;

    Ok(buffer)
}

pub fn read_jxr(filename: &Path) -> Result<PixelBuffer, LocalError> {
    use jpegxr::ImageDecode;
    use jpegxr::PixelFormat::*;
    use jpegxr::Rect;

    let input = File::open(filename)?;
    let mut decoder = ImageDecode::with_reader(input)?;

    let (width, height) = decoder.get_size()?;
    let format = decoder.get_pixel_format()?;
    let (bytes_per_pixel, buf_fmt) = match format {
        PixelFormat128bppRGBAFloat => (16, PixelFormat::HDRFloat32),
        PixelFormat64bppRGBAHalf => (8, PixelFormat::HDRFloat16),
        _ => {
            println!("Pixel format: {:?}", format);
            return Err(LocalError::UnsupportedPixelFormat);
        }
    };

    let stride = width as usize * bytes_per_pixel;
    let mut buffer = PixelBuffer::new(width as usize, height as usize, buf_fmt);

    let rect = Rect::new(0, 0, width, height);
    decoder.copy(&rect, buffer.bytes_mut(), stride)?;

    Ok(buffer)
}

pub fn write_png(filename: &Path, data: &PixelBuffer) -> Result<(), LocalError> {
    use mtpng::encoder::{Encoder, Options};
    use mtpng::ColorType;
    use mtpng::{CompressionLevel, Header};

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

pub fn write_jpeg(filename: &Path, data: &PixelBuffer) -> Result<(), LocalError> {
    // @todo allow setting jpeg quality
    // mozjpeg is much faster than image crate's encoder
    std::panic::catch_unwind(|| {
        use mozjpeg::{ColorSpace, Compress};
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
        let data = c
            .data_as_mut_slice()
            .expect("error accessing JPEG output buffer");
        writer.write_all(data).expect("error writing output file");
    })
    .map_err(|_| LocalError::JpegWriteFailure)
}
