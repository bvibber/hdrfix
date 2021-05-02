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
call :convert usbank

exit /b %ERRORLEVEL%


:convert
cargo run --release -- --pre-scale=0.3333 --hdr-max=1000 --tone-map=reinhard-luma --desaturation-coeff=0.75 --histogram --histogram-max=0.995 samples\%1.png samples\%1-sdr.png
exit /b 0
