#![allow(dead_code)]
#![allow(non_upper_case_globals)]

use std::io::{self};
use std::ffi::{CString, NulError};

use thiserror::Error;

use crate::jpegxr_sys::*;


pub type Result<T> = std::result::Result<T, JXRError>;

#[derive(Error, Debug)]
pub enum JXRError {
    // Rust-side errors
    #[error("I/O error")]
    IoError(#[from] io::Error),
    #[error("null byte in string")]
    NulError(#[from] NulError),
    #[error("invalid data")]
    InvalidData,

    // C-side errors
    #[error("unknown error")]
    UnknownError,
    #[error("fail")]
    Fail,
    #[error("not yet implemented")]
    NotYetImplemented,
    #[error("abstract method")]
    AbstractMethod,
    #[error("out of memory")]
    OutOfMemory,
    #[error("file I/O")]
    FileIO,
    #[error("buffer overflow")]
    BufferOverflow,
    #[error("invalid parameter")]
    InvalidParameter,
    #[error("invalid argument")]
    InvalidArgument,
    #[error("unsupported format")]
    UnsupportedFormat,
    #[error("incorrect codec version")]
    IncorrectCodecVersion,
    #[error("index not found")]
    IndexNotFound,
    #[error("out of sequence")]
    OutOfSequence,
    #[error("not initialized")]
    NotInitialized,
    #[error("must be multiple of 16 lines until last call")]
    MustBeMultipleOf16LinesUntilLastCall,
    #[error("planar alpha banded enc requires temp file")]
    PlanarAlphaBandedEncRequiresTempFile,
    #[error("alpha mode cannot be transcoded")]
    AlphaModeCannotBeTranscoded,
    #[error("incorrect codec sub-version")]
    IncorrectCodecSubVersion
}
use JXRError::*;

fn call(err: ERR) -> Result<()> {
    if err >= 0 {
        return Ok(());
    }
    return Err(match err {
        WMP_errFail => Fail,
        WMP_errNotYetImplemented => NotYetImplemented,
        WMP_errAbstractMethod => AbstractMethod,
        WMP_errOutOfMemory => OutOfMemory,
        WMP_errFileIO => FileIO,
        WMP_errBufferOverflow => BufferOverflow,
        WMP_errInvalidParameter => InvalidParameter,
        WMP_errInvalidArgument => InvalidArgument,
        WMP_errUnsupportedFormat => UnsupportedFormat,
        WMP_errIncorrectCodecVersion => IncorrectCodecVersion,
        WMP_errIndexNotFound => IndexNotFound,
        WMP_errOutOfSequence => OutOfSequence,
        WMP_errNotInitialized => NotInitialized,
        WMP_errMustBeMultipleOf16LinesUntilLastCall => MustBeMultipleOf16LinesUntilLastCall,
        WMP_errPlanarAlphaBandedEncRequiresTempFile => PlanarAlphaBandedEncRequiresTempFile,
        WMP_errAlphaModeCannotBeTranscoded => AlphaModeCannotBeTranscoded,
        WMP_errIncorrectCodecSubVersion => IncorrectCodecSubVersion,
        _ => UnknownError
    });
}

#[derive(Debug, Eq, PartialEq, Clone, Copy)]
pub enum Channel {
    Luminance,
    Red,
    Green,
    Blue,
    Alpha
}

#[derive(Debug, Eq, PartialEq, Clone, Copy)]
pub struct PixelFormatHash {
    raw: u8
}

#[derive(Debug, Eq, PartialEq, Clone, Copy)]
pub enum PixelFormat {
    DontCare,

    // Indexed
    BlackWhite,
    SDR8bppGray,

    // sRGB - uses sRGB gamma
    SDR16bppRGB555,
    SDR16bppRGB565,
    SDR16bppGray,
    SDR24bppBGR,
    SDR24bppRGB,
    SDR32bppBGR,
    SDR32bppBGRA,
    SDR32bppPBGRA,
    SDR32bppGrayFloat,
    SDR32bppRGB,
    SDR32bppRGBA,
    SDR32bppPRGBA,
    SDR48bppRGBFixedPoint,

    // scRGB - linear, HDR-capable
    HDR16bppGrayFixedPoint,
    HDR32bppRGB101010,
    HDR48bppRGB,
    HDR64bppRGBA,
    HDR64bppPRGBA,
    HDR96bppRGBFixedPoint,
    HDR96bppRGBFloat
}
use PixelFormat::*;

static GUID_MAP: [(&'static GUID, PixelFormat); 23] = unsafe {
    [
        (&GUID_PKPixelFormatDontCare, DontCare),
        (&GUID_PKPixelFormatBlackWhite, BlackWhite),
        (&GUID_PKPixelFormat8bppGray, SDR8bppGray),
        (&GUID_PKPixelFormat16bppRGB555, SDR16bppRGB555),
        (&GUID_PKPixelFormat16bppRGB565, SDR16bppRGB565),
        (&GUID_PKPixelFormat16bppGray, SDR16bppGray),
        (&GUID_PKPixelFormat24bppBGR, SDR24bppBGR),
        (&GUID_PKPixelFormat24bppRGB, SDR24bppRGB),
        (&GUID_PKPixelFormat32bppBGR, SDR32bppBGR),
        (&GUID_PKPixelFormat32bppBGRA, SDR32bppBGRA),
        (&GUID_PKPixelFormat32bppPBGRA, SDR32bppPBGRA),
        (&GUID_PKPixelFormat32bppGrayFloat, SDR32bppGrayFloat),
        (&GUID_PKPixelFormat32bppRGB, SDR32bppRGB),
        (&GUID_PKPixelFormat32bppRGBA, SDR32bppRGBA),
        (&GUID_PKPixelFormat32bppPRGBA, SDR32bppPRGBA),
        (&GUID_PKPixelFormat48bppRGBFixedPoint, SDR48bppRGBFixedPoint),
        (&GUID_PKPixelFormat16bppGrayFixedPoint, HDR16bppGrayFixedPoint),
        (&GUID_PKPixelFormat32bppRGB101010, HDR32bppRGB101010),
        (&GUID_PKPixelFormat48bppRGB, HDR48bppRGB),
        (&GUID_PKPixelFormat64bppRGBA, HDR64bppRGBA),
        (&GUID_PKPixelFormat64bppPRGBA, HDR64bppPRGBA),
        (&GUID_PKPixelFormat96bppRGBFixedPoint, HDR96bppRGBFixedPoint),
        (&GUID_PKPixelFormat96bppRGBFloat, HDR96bppRGBFloat)
    ]
};


impl PixelFormat {

    fn from_guid(&guid: &GUID) -> Option<Self> {
        for (&map_guid, map_val) in &GUID_MAP {
            if guid == map_guid {
                return Some(*map_val);
            }
        }
        None
    }

    pub fn from_hash(hash: u8) -> Option<PixelFormat> {
        unsafe {
            let guid = GetPixelFormatFromHash(hash);
            PixelFormat::from_guid(&*guid)
        }
    }

}

pub enum ColorFormat {
    YOnly,
    YUV420,
    YUV422,
    YUV444,
    CMYK,
    NComponent,
    RGB,
    RGBE
}

impl ColorFormat {
    pub fn from_raw(raw: COLORFORMAT) -> Result<ColorFormat> {
        match raw {
            COLORFORMAT_Y_ONLY => Ok(ColorFormat::YOnly),
            COLORFORMAT_YUV_420 => Ok(ColorFormat::YUV420),
            COLORFORMAT_YUV_422 => Ok(ColorFormat::YUV422),
            COLORFORMAT_YUV_444 => Ok(ColorFormat::YUV444),
            COLORFORMAT_CMYK => Ok(ColorFormat::CMYK),
            COLORFORMAT_NCOMPONENT => Ok(ColorFormat::NComponent),
            COLORFORMAT_CF_RGB => Ok(ColorFormat::RGB),
            COLORFORMAT_CF_RGBE => Ok(ColorFormat::RGBE),
            _ => Err(InvalidData)
        }
    }
}

struct IID {
    raw: PKIID
}

pub struct PixelInfo {
    raw: PKPixelInfo
}

/*
impl PixelInfo {

    pub fn format_lookup(lookup_type: PixelFormatHash) -> Result<PixelInfo> {
        let mut info = PixelInfo {
            raw: mem::zeroed()
        };
        call(PixelFormatLookup(&mut info.raw, lookup_type.raw))?;
        Ok(info)
    }

    pub fn format(&self) -> PixelFormat {
        PixelFormat::from_guid(self.raw.guid).unwrap()
    }

    pub fn count_channel(&self) -> usize {
        self.raw.cChannel
    }

    pub fn color_format(&self) -> Option<ColorFormat> {
        ColorFormat::from_raw(self.raw.cfColorFormat).unwrap()
    }
    
    pub fn bit_depth(&self) -> BitDepthBits {
        BitDepthBits::from_raw(self.raw.bdBitDepth)
    }

    pub fn bit_unit(&self) -> u32 {
        self.raw.cbitUnit;
    }

    pub fn gr_bit(&self) -> u32 {
        // implemented as LONG for some reason
        self.raw.grBit as u32
    }

    // todo add the tiff properties
}
*/

struct Factory {
    raw: *mut PKFactory
}

impl Factory {
    fn create() -> Result<Factory> {
        unsafe {
            let mut ptr = std::ptr::null_mut();
            call(PKCreateFactory(&mut ptr, PK_SDK_VERSION))?;
            Ok(Factory {
                raw: ptr
            })
        }
    }

    /*
    pub fn stream_from_filename(&mut self, filename: &str, mode: &str) -> Result<Stream> {
        let filename_bytes = CString::new(filename)?;
        let mode_bytes = CString::new(mode)?;
        let mut ptr = std::ptr::null_mut();
        call((*self.raw).CreateStreamFromFilename.unwrap()(
            &mut ptr,
            filename_bytes.as_ptr(),
            mode_bytes.as_ptr()
        ))?;
        Ok(Stream {
            raw: ptr
        })
    }
    */

    //pub fn stream_from_memory(&mut self)
}

impl Drop for Factory {
    fn drop(&mut self) {
        unsafe {
            (*self.raw).Release.unwrap()(&mut self.raw);
        }
    }
}


struct CodecFactory {
    raw: *mut PKCodecFactory
}

impl CodecFactory {
    fn create() -> Result<Self> {
        unsafe {
            let mut ptr = std::ptr::null_mut();
            call(PKCreateCodecFactory(&mut ptr, WMP_SDK_VERSION))?;
            Ok(Self {
                raw: ptr
            })
        }
    }

    fn create_decoder(&self, _iid: IID) -> Result<ImageDecode> {
        Err(NotYetImplemented)
    }

    fn create_decoder_from_file(&self, filename: &str) -> Result<ImageDecode> {
        unsafe {
            let bytes = CString::new(filename)?;
            let mut ptr = std::ptr::null_mut();
            call((*self.raw).CreateDecoderFromFile.unwrap()(bytes.as_ptr(), &mut ptr))?;
            Ok(ImageDecode {
                raw: ptr
            })
        }
    }
}

impl Drop for CodecFactory {
    fn drop(&mut self) {
        unsafe {
            (*self.raw).Release.unwrap()(&mut self.raw);
        }
    }
}


struct ImageDecode {
    raw: *mut PKImageDecode
}

// @todo implement ImageEncode
// @todo implement FormatConverter

struct Stream {
    raw: *mut WMPStream
}

impl Stream {
    /*
    fn eos(&mut self) -> bool {
        (*self.raw).EOS.unwrap()(self.raw) != 0
    }

    // Note using the std::io::Write interface for dest would
    // introduce an extra copy if you're reading into a large
    // memory buffer.
    //
    // note also this interface has no way to report the
    // number of bytes read, so if asked to read beyond the
    // end of the file there may be uninitialized bytes at
    // end
    fn read(&mut self, dest: &mut [u8]) -> Result<()> {
        call((*self.raw).Read.unwrap()(
            self.raw,
            dest.as_mut_ptr() as *mut c_void,
            dest.len()
        ))?;
        Ok(())
    }

    fn write(&mut self, source: &[u8]) -> Result<()> {
        call((*self.raw).Write.unwrap()(
            self.raw,
            source.as_ptr() as *const c_void,
            source.len()
        ))?;
        Ok(())
    }

    fn set_pos(&mut self, off_pos: usize) -> Result<()> {
        call((*self.raw).SetPos.unwrap()(
            self.raw,
            off_pos
        ))?;
        Ok(())
    }

    fn get_pos(&mut self) -> Result<usize> {
        let mut off_pos: usize;
        call((*self.raw).GetPos.unwrap()(
            self.raw,
            &mut off_pos
        ))?;
        Ok(off_pos)
    }
    */
}

/*
impl Drop for Stream {
    fn drop(&mut self) {
        (*self.raw).Close.unwrap()(&mut self.raw);
    }
}
*/
