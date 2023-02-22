use time::OffsetDateTime;

mod pixelbuffer;
pub use pixelbuffer::*;

mod histogram;
pub use histogram::Histogram;

pub mod io;

mod error;
pub use error::LocalError;

pub mod transforms;
use transforms::*;

mod tonemap;
pub use tonemap::Tonemap;

mod colormap;
pub use colormap::ColorMap;

/// Hdrfix options
#[derive(Clone, Debug, PartialEq, clap::Parser)]
pub struct Hdrfix {
    /// Auto-exposure.
    /// Input level or percentile of input data to average to re-expose to neutral 50% mid-tone on input. Default is 0.5, which passes input through unchanged.
    #[clap(default_value = "0.5")]
    pub auto_exposure: Level,

    /// Exposure adjustment in stops, applied after any auto exposure adjustment. May be positive or negative in stops; defaults to 0, which does not change the exposure.
    #[clap(default_value = "0")]
    pub exposure: f32,

    /// Method for mapping HDR into SDR domain.
    #[clap(long, default_value = "hable")]
    pub tone_map: Tonemap,

    /// Max HDR luminance level for Reinhard algorithm, in nits or a percentile to be calculated from input data. The default is 100%, which represents the highest input value.
    #[clap(long, default_value = "100%")]
    pub hdr_max: Level,

    /// Coefficient for how to scale saturation in tone mapping. 1.0 will desaturate linearly to the compression ratio; smaller values will desaturate more aggressively.
    #[clap(long, default_value = "1")]
    pub saturation: f32,

    /// Method for mapping and fixing out of gamut colors.
    #[clap(long, default_value = "clip")]
    pub color_map: ColorMap,

    /// Gamma power applied on input.
    #[clap(long, default_value = "1.0")]
    pub pre_gamma: f32,

    /// Minimum input level to normalize to 0 when expanding input for processing. May be an absolute value in -infinity..infinity range or a percentile from 0% to 100%.
    #[clap(long, default_value = "0.0")]
    pub pre_levels_min: Level,

    #[clap(long, default_value = "0")]
    #[clap(long, default_value = "1.0")]
    pub pre_levels_max: Level,

    /// Gamma power applied on output.
    #[clap(long, default_value = "1.0")]
    pub post_gamma: f32,

    /// Minimum output level to save when expanding final SDR output for saving. May be an absolute value in 0..1 range or a percentile from 0% to 100%.
    #[clap(long, default_value = "0.0")]
    pub post_levels_min: Level,

    /// Maximum output level to save when expanding final SDR output for saving. May be an absolute value in 0..1 range or a percentile from 0% to 100%.")
    #[clap(long, default_value = "1.0")]
    pub post_levels_max: Level,
}

/// Context for hdr conversion functions
pub struct Context {
    pub scale: f32,
    pub hdr_max: f32,
    pub saturation: f32,
    pub tone_map: Tonemap,
    pub color_map: ColorMap,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Level {
    Scalar(f32),
    Percentile(f32),
}

impl std::str::FromStr for Level {
    type Err = LocalError;

    fn from_str(source: &str) -> Result<Self, Self::Err> {
        match source.strip_suffix('%') {
            Some(val) => Ok(Self::Percentile(val.parse()?)),
            None => Ok(Self::Scalar(source.parse::<f32>()?)),
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub enum PixelFormat {
    SDR8bit,
    HDR8bit,
    HDRFloat16,
    HDRFloat32,
}

impl Hdrfix {
    pub fn apply(&self, source: PixelBuffer) -> Result<PixelBuffer, LocalError> {
        use rayon::prelude::*;

        let width = source.width as usize;
        let height = source.height as usize;

        let mut pre_histogram = Lazy::new(|| Histogram::new(&source));
        let pre_levels_min = pre_histogram.level(self.pre_levels_min);
        let pre_levels_max = pre_histogram.level(self.pre_levels_min);
        let source = {
            let mut dest = PixelBuffer::new(width, height, PixelFormat::HDRFloat32);
            dest.fill(
                source
                    .pixels()
                    .map(|rgb| apply_levels(rgb, pre_levels_min, pre_levels_max, self.pre_gamma)),
            );
            dest
        };

        let mut input_histogram =
            Lazy::new(|| time_func("input histogram", || Ok(Histogram::new(&source))).unwrap());

        let scale = exposure_scale(self.exposure) * 0.5
            / match self.auto_exposure {
                Level::Scalar(level) => level,
                Level::Percentile(percent) => {
                    input_histogram.force().average_below_percentile(percent)
                }
            };

        let hdr_max = match self.hdr_max {
            // hdr_max input is in nits if scalar, so scale it to scrgb
            Level::Scalar(nits) => nits / SDR_WHITE,

            // If given a percentile for hdr_max, detect from input histogram.
            Level::Percentile(val) => input_histogram.force().percentile(val),
        } * scale;

        let options = Context {
            scale,
            hdr_max,
            saturation: self.saturation,
            tone_map: self.tone_map,
            color_map: self.color_map,
        };

        let mut tone_mapped = PixelBuffer::new(width, height, PixelFormat::HDRFloat32);
        time_func("hdr_to_sdr", || {
            tone_mapped.fill(source.pixels().map(|rgb| hdr_to_sdr_pixel(rgb, &options)));
            Ok(())
        })?;

        // apply histogram expansion and color gamut correction to output
        let mut lazy_histogram = Lazy::new(|| {
            time_func("levels histogram", || Ok(Histogram::new(&tone_mapped))).unwrap()
        });
        let post_levels_min = lazy_histogram.level(self.post_levels_min);
        let post_levels_max = lazy_histogram.level(self.post_levels_max);

        let mut dest = PixelBuffer::new(width, height, PixelFormat::SDR8bit);
        time_func("output mapping", || {
            dest.fill(tone_mapped.pixels().map(|rgb| {
                // We have to color map again
                // in case the histogram pushed things back out of gamut.
                clip(options.color_map.map(apply_levels(
                    rgb,
                    post_levels_min,
                    post_levels_max,
                    self.post_gamma,
                )))
            }));
            Ok(())
        })?;

        Ok(dest)
    }
}

struct Lazy<T, F>
where
    F: (FnOnce() -> T),
{
    value: Option<T>,
    func: Option<F>,
}

impl<T, F> Lazy<T, F>
where
    F: (FnOnce() -> T),
{
    fn new(func: F) -> Self {
        Lazy {
            value: None,
            func: Some(func),
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

impl<F> Lazy<Histogram, F>
where
    F: (FnOnce() -> Histogram),
{
    fn level(&mut self, level: Level) -> f32 {
        match level {
            Level::Scalar(val) => val,
            Level::Percentile(val) => self.force().percentile(val),
        }
    }
}

pub fn time_func<F, G>(msg: &str, func: F) -> Result<G, LocalError>
where
    F: FnOnce() -> Result<G, LocalError>,
{
    let start = OffsetDateTime::now_utc();
    let result = func()?;
    let delta = OffsetDateTime::now_utc() - start;
    println!("{} in {} ms", msg, delta.as_seconds_f64() * 1000.0);
    Ok(result)
}
