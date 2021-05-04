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
    --hdr-max=4000 ^
    --sdr-white=400 ^
    --desaturation-coeff=0.96 ^
    --levels-max=99%% ^
    samples\%1-hdr.jxr ^
    samples\%1-sdr.png
exit /b 0
