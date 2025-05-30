# !bin bash

./build/bin/llama-speculative \
        -m /root/autodl-tmp/model/prosparse-llama-2-7b-gguf/prosparse-llama-2-7b.gguf \
        -md /root/autodl-tmp/models/llama-160m/llama-160m.fp16.gguf \
        -c 0 -co -ngl 99 -ngld 99 -fa \
        --draft-max 7 --draft-min 3 --draft-p-min 0.0 \
        -p "# Dijkstra's shortest path algorithm in Python (4 spaces indentation) + complexity analysis:\n\n" \
        --n-predict 512 --seed 42 \
        --sampling-seq k \
        --top-k 4 --temp 0.0 -np 4