rem @echo off

call :convert 1-linear "--color-map=clip" "--tone-map=linear"
call :convert 2-dark   "--color-map=clip" "--tone-map=linear" "--sdr-white=2560"
call :convert 3-mapped "--color-map=clip"
call :convert 4-mapped "--color-map=clip" "--tone-map=reinhard-rgb"
call :convert 5-gamut  
call :convert 6-gamut-oklab  "--color-map=oklab"
call :convert 7-oklab        "--color-map=clip" "--tone-map=reinhard-oklab"
call :convert 8-oklab-gamut  "--color-map=oklab" "--tone-map=reinhard-oklab"

exit /b %ERRORLEVEL%


:convert
cargo run --release -- ^
    samples\sunrise-hdr.jxr ^
    samples\sunrise-test-%1.png ^
    %2 %3 %4 %5 %6 %7 %8 %9
exit /b 0
