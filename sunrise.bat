rem @echo off

call :convert 1-linear "--color-map=clip" "--tone-map=linear"
call :convert 2-dark   "--color-map=clip" "--tone-map=linear" "--sdr-white=2560"
call :convert 3-mapped "--color-map=clip"
call :convert 4-gamut
call :convert 5-gamma "--pre-gamma=0.5"
call :convert 6-scale-gamma "--pre-gamma=0.5" "--sdr-white=400"
call :convert 7-levels "--pre-gamma=0.5" "--sdr-white=400" "--levels-min=0%%%%" "--levels-max=100%%%%"

exit /b %ERRORLEVEL%


:convert
cargo run --release -- ^
    samples\sunrise-hdr.jxr ^
    samples\sunrise-test-%1.png ^
    %2 %3 %4 %5 %6 %7 %8 %9
exit /b 0
