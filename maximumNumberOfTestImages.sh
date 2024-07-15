max_count=0
max_dir=""

for dir in /home/sfmt/PycharmProjects/vq-vae-2-pytorch/MVTec/*/test; do
    count=$(find "$dir" -type f | wc -l)
    if [ "$count" -gt "$max_count" ]; then
        max_count="$count"
        max_dir="$dir"
    fi
done

echo "Directory with the maximum file count: $max_dir"
echo "Maximum File Count: $max_count"
