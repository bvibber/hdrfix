convert() {
    cargo run --release -- \
        --hdr-max=4000 \
        --pre-scale=0.2 \
        --desaturation-coeff=0.96 \
        --histogram \
        --histogram-max=0.99 \
        samples/"$1"-hdr.jxr \
        samples/"$1"-sdr.png
}

convert burbank
convert closeup
convert cloud
convert ikea
convert newyork
convert oregon
convert panel
convert portland
convert sanfran
convert spitfire
convert sunrise
convert usbank
