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
cargo -q run --release -- --sdr-white=100 --gamma=1.2 samples\%1.png samples\%1-sdr.png
exit /b 0
