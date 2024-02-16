set -e

convert() {
    cargo run --release -- \
        samples/"$1"-hdr.jxr \
        samples/"$1"-sdr.jpg
}

convert burbank
convert closeup
convert cloud
convert eagle
convert green
convert ikea
convert newyork
convert oregon
convert overcast1
convert overcast2
convert panel
convert portland
convert sanfran
convert spitfire
convert sunrise
convert usbank
convert xbox
