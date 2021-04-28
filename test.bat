@echo off

call :convert burbank
call :convert closeup
call :convert cloud
call :convert newyork
call :convert oregon
call :convert panel
call :convert portland
call :convert sanfran
call :convert spitfire
call :convert usbank

exit /b %ERRORLEVEL%


:convert
cargo run --release -- --sdr-white=200 --hdr-max=2000 samples\%1.png samples\%1-sdr.png
exit /b 0
