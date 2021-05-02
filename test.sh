convert() {
    cargo run --release -- --pre-scale=0.3333 --hdr-max=1000 --tone-map=reinhard-luma --desaturation-coeff=0.75 samples/"$1"-hdr.jxr samples/"$1"-sdr.png
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
