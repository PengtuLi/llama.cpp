# Sparkinfer Cache Manager

`sparkinfer_cache_manager` 是 `sparkinfer inference` 中的类，实现在以下文件中：
- `./src/llama-sparkinfer.h`
- `./src/llama-sparkinfer.cpp`

## 规范
- 实例化的命名规范为：`spif_cache`
- 张量指：`struct ggml_tensor`
- 时刻记住当前的步骤是在**构建计算图**还是在**计算计算图**

## 职责
1. **加载阶段**  
    按照 `gpu_neu_idx` 把 CPU 上的神经元加载到 GPU 上，更新 `spif_cache` 中的 GPU 张量指针  
    **[TODO]**:  
    - `lpt` 之前在 `llama-model.cpp` 中实现了一个 `sparkInfer_layer_cache`，用于加载阶段的神经元加载到 GPU 上。  
    - 需要将此类与 `sparkinfer_cache_manager` 合并，共同放在 `llama-sparkinfer.h/cpp` 中。

2. **跨 decodes 张量管理**  
    负责保存、维护跨 decodes 的张量，在构建计算图时能够调用函数获取这些张量。

3. **(optional) 构建 reload 的计算图**  
    - 这部分无所谓，因为在 `build-graph` 函数中，如果能从 `spif_cache` 中获取跨 decodes 张量，直接原地构建或返回即可。

4. **(不建议实现)**  
    不负责 reload 的具体实现（此时已经是计算计算图阶段）。  
    **[TODO]**:  
    - 参考 `llama-kv-cache` 的具体实现，完成计算计算图阶段的设计。

## 内部成员
`sparkinfer_cache_manager` 包含一个 `vector` 列表 `layer_caches`，其中存储 `layer_cache` 类。  
`layer_cache` 类用于保存 cache 中需要动态变化的、跨 decodes 的张量：
1. GPU 上需要动态调整的神经元组：`ffn_gpu_up`，`ffn_gpu_gate`，`ffn_gpu_down_t`
2. 指导 GPU 神经元映射、reload 的张量：`gpu_neu_idx/mask`，`DFR_score`
3. 一些 hyperparameter (`n_layers`, `dfr_decay`...)
4. **(optional)** 构建 reload 的计算图

## 归属与初始化
目前的设计思路：
- `sparkinfer_cache_manager` 作为 `struct llama_context` 的成员（`llama_context` 是一个单例，跨 decodes 存在）。
- 在 `llama_context` 初始化时初始化，与 `kv-cache` 同级。  
**[TODO]**:  
- `lpt` 加载阶段的类是在 model 加载时调用，早于 `llama_context` 初始化。  
- 只能在 model 加载阶段就初始化 `sparkinfer_cache_manager` 然后在 `llama_context` 初始化的时候再放进`llama_context`
