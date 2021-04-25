convert() {
    cargo -q run --release samples/"$1".png samples/"$1"-sdr.png
}

convert burbank
convert closeup
convert cloud
convert newyork
convert oregon
convert portland
convert sanfran
convert spitfire
convert usbank
