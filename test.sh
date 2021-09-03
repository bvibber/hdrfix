convert() {
    cargo run --release -- \
        --pre-gamma=2 \
        --exposure=-4 \
        --tone-map=reinhard \
        --saturation=1.5 \
        --post-gamma=0.5 \
        --color-map=clip \
        samples/"$1"-hdr.jxr \
        samples/"$1"-sdr.jpg
}

convert burbank
convert closeup
convert cloud
convert green
convert ikea
convert newyork
convert oregon
convert overcast1
convert overcast2
convert panel
convert portland
convert sanfran
convert spitfire
convert sunrise
convert usbank
