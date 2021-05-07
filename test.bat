@echo off

call :convert burbank
call :convert closeup
call :convert cloud
call :convert ikea
call :convert newyork
call :convert oregon
call :convert panel
call :convert portland
call :convert sanfran
call :convert spitfire
call :convert sunrise
call :convert usbank

exit /b %ERRORLEVEL%


:convert
cargo run --release -- ^
    --hdr-max=100%% ^
    --sdr-white=120 ^
    --tone-map=reinhard-oklab ^
    --color-map=oklab ^
    --levels-min=0%% ^
    --levels-max=100%% ^
    samples\%1-hdr.jxr ^
    samples\%1-sdr.png

exit /b 0
