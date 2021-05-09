convert() {
    cargo run --release -- \
        --exposure=-1.5 \
        --hdr-max=1000 \
        --levels-min=0.1% \
        --levels-max=99.9% \
        samples/"$1"-hdr.jxr \
        samples/"$1"-sdr.png

    # Roughly equivalent to MSFS SDR
    #cargo run --release -- \
    #    --exposure=-1.5 \
    #    --tone-map=reinhard-rgb \
    #    --hdr-max=640 \
    #    samples/"$1"-hdr.jxr \
    #    samples/"$1"-sim.png
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
