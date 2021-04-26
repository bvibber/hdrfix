#![allow(dead_code)]
#![allow(non_upper_case_globals)]

use std::convert::TryFrom;
use std::io::{self, Read, Seek, SeekFrom};
use std::ffi::{CString, NulError, c_void};

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

    fn from_guid(&guid: &GUID) -> Result<Self> {
        for (&map_guid, map_val) in &GUID_MAP {
            if guid == map_guid {
                return Ok(*map_val);
            }
        }
        Err(InvalidData)
    }

    pub fn from_hash(hash: u8) -> Result<PixelFormat> {
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


struct InputStream<R: Read + Seek> {
    raw: WMPStream,
    reader: R,
    length: u64
}

impl<R> InputStream<R> where R: Read + Seek {
    fn create(mut reader: R) -> Result<*mut Self> {
        // Get length so we can calculate EOS. Assumes static files.
        let start = reader.stream_position()?;
        reader.seek(SeekFrom::End(0))?;
        let length = reader.stream_position()?;
        reader.seek(SeekFrom::Start(start))?;

        unsafe {
            let state = Box::into_raw(Box::new(Self {
                raw: WMPStream {
                    state: WMPStream__bindgen_ty_1 {
                        pvObj: std::ptr::null_mut(),
                    },
                    fMem: 0,
                    Close: Some(Self::input_stream_close),
                    EOS: Some(Self::input_stream_eos),
                    Read: Some(Self::input_stream_read),
                    Write: Some(Self::input_stream_write),
                    SetPos: Some(Self::input_stream_set_pos),
                    GetPos: Some(Self::input_stream_get_pos),
                },
                reader: reader,
                length: length
            }));
            (*state).raw.state.pvObj = std::mem::transmute(state);
            Ok(state)
        }
    }

    unsafe fn get_state(me: *mut WMPStream) -> *mut Self {
        std::mem::transmute((*me).state.pvObj)
    }

    unsafe extern "C" fn input_stream_close(me: *mut *mut WMPStream) -> ERR {
        let state = Self::get_state(*me);
        let boxed = Box::from_raw(state);
        drop(boxed);
        *me = std::ptr::null_mut();
        WMP_errSuccess as ERR
    }

    unsafe extern "C" fn input_stream_eos(me: *mut WMPStream) -> i32 {
        let state = Self::get_state(me);
        match (*state).reader.stream_position() {
            Ok(pos) => (pos == (*state).length) as i32,
            Err(_) => true as i32
        }
    }

    unsafe extern "C" fn input_stream_read(me: *mut WMPStream, dest: *mut c_void, cb: usize) -> ERR {
        let state = Self::get_state(me);
        let bytes: *mut u8 = std::mem::transmute(dest);
        let dest_slice = std::slice::from_raw_parts_mut(bytes, cb);
        match (*state).reader.read_exact(dest_slice) {
            Ok(_) => WMP_errSuccess as ERR,
            Err(_) => WMP_errFileIO as ERR
        }
    }

    unsafe extern "C" fn input_stream_write(_me: *mut WMPStream, _dest: *const c_void, _cb: usize) -> ERR {
        WMP_errFileIO as ERR
    }

    unsafe extern "C" fn input_stream_set_pos(me: *mut WMPStream, off_pos: usize) -> ERR {
        let state = Self::get_state(me);
        match (*state).reader.seek(SeekFrom::Start(off_pos as u64)) {
            Ok(_) => WMP_errSuccess as ERR,
            Err(_) => WMP_errFileIO as ERR
        }
    }

    unsafe extern "C" fn input_stream_get_pos(me: *mut WMPStream, off_pos: *mut usize) -> ERR {
        let state = Self::get_state(me);
        match (*state).reader.stream_position() {
            Ok(pos) => {
                match usize::try_from(pos) {
                    Ok(out) => {
                        *off_pos = out;
                        WMP_errSuccess as ERR
                    },
                    Err(_) => WMP_errFileIO as ERR
                }
            },
            Err(_) => WMP_errFileIO as ERR
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

    fn create_decoder<R: Read + Seek>(&self, reader: R) -> Result<ImageDecode<R>> {
        unsafe {
            let stream = InputStream::create(reader)?;
            let mut codec: *mut PKImageDecode = std::ptr::null_mut();
            call((*self.raw).CreateCodec.unwrap()(&IID_PKImageWmpDecode, std::mem::transmute(&mut codec)))?;
    
            call((*codec).Initialize.unwrap()(codec, &mut (*stream).raw))?;
            Err(NotYetImplemented)
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


pub struct Rect {
    raw: PKRect
}

impl Rect {
    pub fn new(x: i32, y: i32, width: i32, height: i32) -> Self {
        Self {
            raw: PKRect {
                X: x,
                Y: y,
                Width: width,
                Height: height
            }
        }
    }

    pub fn get_x(&self) -> i32 {
        return self.raw.X;
    }

    pub fn get_y(&self) -> i32 {
        return self.raw.Y;
    }

    pub fn get_width(&self) -> i32 {
        return self.raw.Width;
    }

    pub fn get_height(&self) -> i32 {
        return self.raw.Height;
    }
}

pub struct ImageDecode<R: Read + Seek> {
    raw: *mut PKImageDecode,
    stream: *mut InputStream<R>,
}

impl<R> ImageDecode<R> where R: Read + Seek {
    pub fn get_pixel_format(&mut self) -> Result<PixelFormat> {
        unsafe {
            let mut guid: GUID = std::mem::zeroed();
            call((*self.raw).GetPixelFormat.unwrap()(self.raw, &mut guid))?;
            PixelFormat::from_guid(&guid)
        }
    }

    pub fn get_size(&mut self) -> Result<(i32, i32)> {
        unsafe {
            let mut width: i32 = 0;
            let mut height: i32 = 0;
            call((*self.raw).GetSize.unwrap()(self.raw, &mut width, &mut height))?;
            Ok((width, height))
        }
    }

    pub fn get_resolution(&mut self) -> Result<(f32, f32)> {
        unsafe {
            let mut horiz: f32 = 0.0;
            let mut vert: f32 = 0.0;
            call((*self.raw).GetResolution.unwrap()(self.raw, &mut horiz, &mut vert))?;
            Ok((horiz, vert))
        }
    }

    pub fn get_color_context(&mut self) -> Result<(u8, u32)> {
        // ???
        unsafe {
            let mut a: u8 = 0;
            let mut b: u32 = 0;
            call((*self.raw).GetColorContext.unwrap()(self.raw, &mut a, &mut b))?;
            Ok((a, b))
        }
    }

    pub fn get_descriptive_metadata(&mut self) -> Result<()> {
        Err(NotYetImplemented)
    }

    pub fn get_raw_stream(&mut self) -> Result<&mut R> {
        unsafe {
            let mut stream: *mut WMPStream = std::ptr::null_mut();
            call((*self.raw).GetRawStream.unwrap()(self.raw, &mut stream))?;
            let state = InputStream::<R>::get_state(stream);
            Ok(&mut (*state).reader)
        }
    }

    pub fn copy(&mut self, rect: Rect, dest: &mut [u8], stride: u32) -> Result<()> {
        unsafe {
            call((*self.raw).Copy.unwrap()(self.raw, &rect.raw, dest.as_mut_ptr(), stride))?;
            Ok(())
        }
    }

    pub fn get_frame_count(&mut self) -> Result<u32> {
        unsafe {
            let mut frames: u32 = 0;
            call((*self.raw).GetFrameCount.unwrap()(self.raw, &mut frames))?;
            Ok(frames)
        }
    }

    pub fn select_frame(&mut self, frame: u32) -> Result<()> {
        unsafe {
            call((*self.raw).SelectFrame.unwrap()(self.raw, frame))?;
            Ok(())
        }
    }
}

impl<R> Drop for ImageDecode<R> where R: Read + Seek {
    fn drop(&mut self) {
        unsafe {
            // This will call through to close the stream.
            (*self.raw).Release.unwrap()(&mut self.raw);
        }
    }
}
