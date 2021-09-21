@rem Watches for new .jxr files and converts them as they appear.

hdrfix ^
    --pre-gamma=2 ^
    --exposure=-4 ^
    --saturation=1.5 ^
    --post-gamma=0.5 ^
    --color-map=clip ^
    --watch=.
