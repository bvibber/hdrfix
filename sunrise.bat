rem @echo off

call :convert 1-dark   "--color-map=clip" "--tone-map=linear" "--sdr-white=2560"
call :convert 2-linear "--color-map=clip" "--tone-map=linear"
call :convert 3-maprgb "--color-map=clip" "--tone-map=reinhard-rgb"
call :convert 4-mapped "--color-map=clip"
call :convert 5-desat

exit /b %ERRORLEVEL%


:convert
cargo run --release -- ^
    samples\sunrise-hdr.jxr ^
    samples\sunrise-test-%1.png ^
    %2 %3 %4 %5 %6 %7 %8 %9
exit /b 0
