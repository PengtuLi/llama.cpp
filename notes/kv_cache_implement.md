# llama.cpp 实现 KV Cache 的方式

依旧是通过图构建和图计算的方式实现

## 图构建阶段

1. 计算当前 token 的 `q`、`k`、`v` 向量。
2. 将 `k` 和 `v` 存入 KV Cache：
    ```cpp
    // store to KV cache
    {
        ggml_build_forward_expand(gf, kv_self->cpy_k(ctx0, k_cur, il));
        ggml_build_forward_expand(gf, kv_self->cpy_v(ctx0, v_cur, il));
    }
    ```
    查看 `cpy_k` 函数的实现：
    ```cpp
    ggml_tensor * llama_kv_cache_unified::cpy_k(ggml_context * ctx, ggml_tensor * k_cur, int32_t il) const {
        const int32_t ikv = map_layer_ids.at(il);

        auto * k = layers[ikv].k;

        const int64_t n_tokens = k_cur->ne[2];

        ggml_tensor * k_view = ggml_view_1d(ctx, k,
                n_tokens * hparams.n_embd_k_gqa(il),
                ggml_row_size(k->type, hparams.n_embd_k_gqa(il)) * head);

        return ggml_cpy(ctx, k_cur, k_view);
    }
    ```
    **关键点**：`k_cur` 是当前 token 的张量，`k_view` 是 `k_cache` 向量向后偏移到一个与 `k_cur` 大小相同的连续空白张量。随后在 `ggml_cpy` 中将该节点的操作设置为 `GGML_OP_CPY`。

## 图计算阶段

3. 对于 CUDA 后端，当读取到当前节点的操作为 `GGML_OP_CPY` 时，会执行 `ggml_cuda_cpy`。该函数在 CUDA 后端完成 D2D 的内存拷贝。

## 对于 Reload 节点的启发

`reload` 并不是简单的 KV Cache-like view 加 `cpy`，我认为需要新建 `GGML_OP_RELOAD`，并在 CPU 计算后端完成 `reload` 的具体跨后端内存拷贝操作。
