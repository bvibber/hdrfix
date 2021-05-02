convert() {
    cargo run --release -- --pre-scale=0.3333 --hdr-max=1000 --tone-map=reinhard-rgb samples/"$1"-hdr.jxr samples/"$1"-sdr.png

    # this is not bad
    # but is a bit too saturated still
    cargo run --release -- --pre-scale=1.5 --hdr-max=10000 --tone-map=reinhard-blend --post-gamma=1.8 --post-scale=1.0  samples/"$1"-hdr.jxr samples/"$1"-sdr-blend.png
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
convert usbank
