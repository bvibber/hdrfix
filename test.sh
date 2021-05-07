convert() {
    cargo run --release -- \
        --hdr-max=100% \
        --sdr-white=120 \
        --tone-map=reinhard-oklab \
        --color-map=oklab \
        --levels-min=0% \
        --levels-max=100% \
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
