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
cargo run --release -- --sdr-white=100 --gamma=1.2 --color=clip samples\%1.png samples\%1-sdr.png
cargo run --release -- --sdr-white=100 --gamma=1.2 --color=desaturate samples\%1.png samples\%1-sdr-desat.png
exit /b 0
