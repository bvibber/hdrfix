convert() {
    cargo run --release -- \
        --pre-scale=0.2 \
        --tone-map=reinhard-luma \
        --desaturation-coeff=0.96 \
        --histogram \
        --histogram-min=0.00 \
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
