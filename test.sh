convert() {
    cargo run --release samples/"$1".png samples/"$1"-sdr.png
}

convert burbank
convert closeup
convert cloud
convert newyork
convert oregon
convert sanfran
convert spitfire
convert usbank
