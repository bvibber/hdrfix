convert() {
    cargo run --release -- \
        --exposure=-1.5 \
        --hdr-max=100% \
        --levels-min=0.1% \
        --levels-max=99.9% \
        samples/"$1"-hdr.jxr \
        samples/"$1"-sdr.jpg
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
