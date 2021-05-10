convert() {
    cargo run --release -- \
        --auto-exposure=99.9% \
        --hdr-max=99.9% \
        --levels-min=0% \
        --levels-max=99.9% \
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
convert panel
convert portland
convert sanfran
convert spitfire
convert sunrise
convert usbank
