@echo off

call :convert burbank
call :convert closeup
call :convert cloud
call :convert eagle
call :convert green
call :convert ikea
call :convert newyork
call :convert oregon
call :convert overcast1
call :convert overcast2
call :convert panel
call :convert portland
call :convert sanfran
call :convert spitfire
call :convert sunrise
call :convert usbank

exit /b %ERRORLEVEL%


:convert

cargo run --release -- ^
    samples\%1-hdr.jxr ^
    samples\%1-sdr.jpg

exit /b 0
