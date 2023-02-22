use thiserror::Error;

#[derive(Error, Debug)]
pub enum LocalError {
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("numeric format error: {0}")]
    ParseFloatError(#[from] core::num::ParseFloatError),
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
    RecvError(#[from] std::sync::mpsc::RecvError),
    #[error("Image format error")]
    ImageError(#[from] image::ImageError),
    #[error("JPEG write failure")]
    JpegWriteFailure,
}
