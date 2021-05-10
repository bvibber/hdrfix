@echo off

call :convert burbank
call :convert closeup
call :convert cloud
call :convert green
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
    --auto-exposure=99.9%% ^
    --hdr-max=99.9%% ^
    samples\%1-hdr.jxr ^
    samples\%1-sdr.jpg

exit /b 0
