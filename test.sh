convert() {
    cargo run --release --sdr-white=200 --hdr-max=2000 samples/"$1".png samples/"$1"-sdr.png
}

convert burbank
convert closeup
convert cloud
convert newyork
convert oregon
convert panel
convert portland
convert sanfran
convert spitfire
convert usbank
