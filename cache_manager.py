from matplotlib import pyplot as plt
import numpy as np

NUM_LAYERS = 32
INTERMEDIATE_SIZE = 11008  
HIDDEN_SIZE = 4096
BYTES_PER_NEURONS = HIDDEN_SIZE*2*3 # up gate down

SPARSE_IDX_PATH = "/root/autodl-tmp/datasets/prosparse-llama-2-7b-sparse-idx" 
ACTIVATION_PATH = "/root/autodl-tmp/datasets/prosparse-llama-2-7b-activation"

GPU_MEMORY_GB = 6.3
FORWARD_STEPS = 500
DECAY_ALPHA = 0.75
need_plot = True


def load_sparse_idx(layer_id):
    mmap_file = f"{SPARSE_IDX_PATH}/sparse_idx_{layer_id}.mmap"
    mmap = np.memmap(mmap_file, dtype=np.uint8, mode="r", shape=(10240, INTERMEDIATE_SIZE // 8))
    mmap_segment = mmap[0:FORWARD_STEPS]
    sparse_idx = np.unpackbits(mmap_segment, axis=1).reshape(FORWARD_STEPS, INTERMEDIATE_SIZE)
    return sparse_idx

def load_activation(layer_id):
    mmap_file = f"{ACTIVATION_PATH}/mlp_label_{layer_id}.mmap"
    mmap = np.memmap(mmap_file, dtype=np.uint8, mode="r", shape=(10240, INTERMEDIATE_SIZE // 8))
    activation = np.unpackbits(mmap, axis=1).reshape(10240, INTERMEDIATE_SIZE)
    return activation

def load_random_sparse_idx(layer_id):
    real_sparse = load_sparse_idx(layer_id).astype(bool)
    T, N = real_sparse.shape

    activation_freq = real_sparse.mean(axis=0)  

    random_sparse = np.random.rand(T, N) < activation_freq  
    return random_sparse


def neurons_to_bytes(num_neurons):
    return num_neurons * BYTES_PER_NEURONS

def neuron_partition(activation_list, gpu_memory_bytes, num_layers):
    """Partition neurons across layers based on activation frequency.
    
    Args:
        activation_list (list): List of activation tensors, each (T, N).
        gpu_memory_bytes (float): Total GPU memory in bytes.
        num_layers (int): Number of layers in the network.
    
    Returns:
        list: neurons_splits, list of neuron indices per layer on GPU.
    """
    # Compute activation frequency per layer
    activation_freq = [np.sum(sparse_idx, axis=0) for sparse_idx in activation_list]
    total_freq_per_layer = [np.sum(freq) for freq in activation_freq]
    total_freq = sum(total_freq_per_layer)
    
    
    memory_proportions = [freq / total_freq for freq in total_freq_per_layer]
    
    # Allocate memory per layer
    allowed_memory_per_layer = [prop * gpu_memory_bytes for prop in memory_proportions]
    neurons_per_layer = [min(int(mem / BYTES_PER_NEURONS), INTERMEDIATE_SIZE) for mem in allowed_memory_per_layer]
    memory_per_layer = [neu*BYTES_PER_NEURONS for neu in neurons_per_layer]

    for layer_id in range(NUM_LAYERS):
        print(f"layer {layer_id}, load {neurons_per_layer[layer_id]} neurons, used {memory_per_layer[layer_id]/(1024*1024)} MB")
    print(f"{sum(neurons_per_layer)} neurons were loaded in total")
    print(f"{sum(memory_per_layer)/(1024*1024)} MB VRAM were used in total")


    # Select top-activated neurons per layer
    neurons_splits = []
    for layer_id in range(num_layers):
        freq = activation_freq[layer_id]
        # Sort neurons by activation frequency (descending)
        top_neurons = np.argsort(freq)[::-1][:neurons_per_layer[layer_id]]
        neurons_splits.append(top_neurons)
    
    return neurons_splits

class CacheManager:
    def __init__(self, gpu_memory_gb, neurons_splits):
        """初始化 CacheManager, 设置 GPU 内存和神经元分配。
        
        Args:
            gpu_memory_gb (float): GPU 内存大小(单位: GB)。
            neurons_splits (list): 每层在 GPU 上的神经元索引列表。
            bytes_per_neuron (int): 每个神经元占用的字节数。
        """

        self.gpu_memory_bytes = gpu_memory_gb * (1024 ** 3)  
        self.neurons_idx = [set(split) for split in neurons_splits]
        self.bytes_per_neuron = BYTES_PER_NEURONS
        self.neurons_score = [np.zeros(INTERMEDIATE_SIZE) for _ in range(NUM_LAYERS)]
        for i in range(NUM_LAYERS):
            self.neurons_score[i][list(self.neurons_idx[i])] = 1

        # for recording 
        self.hit_rates = []  
        self.reload_list = [[] for _ in range(NUM_LAYERS)]  
        if need_plot:
            self.hot_hit_rates = [[] for _ in range(NUM_LAYERS)] 
            self.cold_hit_rates = [[] for _ in range(NUM_LAYERS)]  
    
    def simulate_forward(self, sparse_idx_list, reload_type):
        """模拟 forward 过程，计算所有层的热点和冷点命中率。
        
        Args:
            sparse_idx_list (list): 每层的稀疏索引张量列表，每个张量形状为 (T, N)。
            reload_type (str) : powerinfer or DFR reload (decay frequency reload)
        
        Returns:
            tuple: (hot_hit_ratio, cold_hit_ratio, reload_list)，包含每层的平均命中率和 reload 次数。
        """
        if reload_type == "DFR-reload":
            print(f"using DFR reload with decay alpha: {DECAY_ALPHA}")

        hot_hit_ratio = []
        cold_hit_ratio = []

        for layer_id in range(NUM_LAYERS):
            sparse_idx = sparse_idx_list[layer_id]
            T, N = sparse_idx.shape
            layer_hot_hits = []
            layer_cold_hits = []

            for t in range(T):
                # 在每个时间步的 forward 之前进行 reload
                if reload_type == "DFR-reload":
                    self.DFR_reload(layer_id, sparse_idx[t], decay_alpha=DECAY_ALPHA)
                    

                gpu_neurons = self.neurons_idx[layer_id]
                num_gpu_neurons = len(gpu_neurons)
                num_cpu_neurons = INTERMEDIATE_SIZE - num_gpu_neurons

                activated_neurons = np.where(sparse_idx[t])[0]
                if len(activated_neurons) == 0:
                    continue

                hot_hits = sum(1 for neuron in activated_neurons if neuron in gpu_neurons)
                cold_hits = len(activated_neurons) - hot_hits

                hot_hit_rate = hot_hits / num_gpu_neurons if num_gpu_neurons > 0 else 0.0
                cold_hit_rate = cold_hits / num_cpu_neurons if num_cpu_neurons > 0 else 0.0

                if need_plot:
                    self.hot_hit_rates[layer_id].append(hot_hit_rate)
                    self.cold_hit_rates[layer_id].append(cold_hit_rate)
                layer_hot_hits.append(hot_hit_rate)
                layer_cold_hits.append(cold_hit_rate)

            avg_hot_hit_rate = np.mean(layer_hot_hits) if layer_hot_hits else 0.0
            avg_cold_hit_rate = np.mean(layer_cold_hits) if layer_cold_hits else 0.0
            avg_reload = np.mean(self.reload_list[layer_id]) if self.reload_list[layer_id] else 0.0

            hot_hit_ratio.append(avg_hot_hit_rate)
            cold_hit_ratio.append(avg_cold_hit_rate)
            self.hit_rates.append((avg_hot_hit_rate, avg_cold_hit_rate))

            print(f"Layer {layer_id}: Hot hit ratio = {avg_hot_hit_rate:.4f}, Cold hit ratio = {avg_cold_hit_rate:.4f}, average reload neurons = {avg_reload:.2f}")

        return hot_hit_ratio, cold_hit_ratio, self.reload_list

    def DFR_reload(self, layer_id, current_activation, decay_alpha=0.85):
        """Decay frequency reload:
        1. 更新 neurons_score, 记录每个神经元的启动频率分数
        2. 根据 neurons_score 驱逐分数低的神经元, 加载分数高的神经元
        3. 维护 reload 列表，记录每层的 reload 次数
        
        Args:
            layer_id (int): 当前层 ID
            current_activation (np.ndarray): 当前时间步的激活情况，形状为 (N,)
            decay_alpha (float): 衰减因子
        """
        # 1. 更新 neurons_score
        self.neurons_score[layer_id] = self.neurons_score[layer_id] * decay_alpha + current_activation

        # 2. 驱逐和加载神经元
        scores = self.neurons_score[layer_id]
        current_gpu_neurons = self.neurons_idx[layer_id]
        num_gpu_neurons = len(current_gpu_neurons)

        # 选择分数最高的 num_gpu_neurons 个神经元
        new_gpu_neurons = set(np.argsort(scores)[-num_gpu_neurons:])

        # 计算需要加载和驱逐的神经元
        to_load = new_gpu_neurons - current_gpu_neurons
        reload_count = len(to_load)  # 加载和驱逐数量相等

        # 更新 neurons_idx
        self.neurons_idx[layer_id] = new_gpu_neurons

        # 3. 记录 reload 次数
        self.reload_list[layer_id].append(reload_count)

    def plot_metrics(self, layers_to_plot=None):
        """绘制每层的热点命中率、冷点命中率和重新加载数量的折线图。
        
        Args:
            layers_to_plot (list): 要绘制的层 ID 列表，默认为 [0, 7, 15, 23, 31]。
        """
        if layers_to_plot is None:
            layers_to_plot = [0, 7, 15, 23, 31]

        for layer_id in layers_to_plot:
            plt.figure(figsize=(15, 10))
            time_steps = range(len(self.hot_hit_rates[layer_id]))
            plt.subplot(3, 1, 1)
            plt.plot(time_steps, self.hot_hit_rates[layer_id], label=f'layer {layer_id}')
            plt.title('hot-hit ratio')
            plt.xlabel('forward step')
            plt.ylabel('hot-hit ratio')
            plt.legend()

            plt.subplot(3, 1, 2)
            plt.plot(time_steps, self.cold_hit_rates[layer_id], label=f'layer {layer_id}')
            plt.title('cold-hit ratio')
            plt.xlabel('forward step')
            plt.ylabel('cold-hit ratio')
            plt.legend()

            plt.subplot(3, 1, 3)
            plt.plot(time_steps, self.reload_list[layer_id], label=f'layer {layer_id}')
            plt.title('neurons reloads numbers')
            plt.xlabel('forward step')
            plt.ylabel('neurons reloads numbers')
            plt.legend()

            plt.tight_layout()
            plt.savefig(f'metrics_plot_{layer_id}.svg',format='svg')
            plt.close()
            print(f"save as 'metrics_plot_{layer_id}.svg'")

def main():
    # Offline profile
    activation_list = [load_activation(layer_id) for layer_id in range(NUM_LAYERS)]
    neurons_splits = neuron_partition(activation_list, GPU_MEMORY_GB * (1024 ** 3), NUM_LAYERS)

    # Online inference simulation
    cache_manager = CacheManager(GPU_MEMORY_GB, neurons_splits)
    sparse_idx_test = [load_sparse_idx(layer_id) for layer_id in range(NUM_LAYERS)]
    hot_hit_ratio, cold_hit_ratio, reload_list = cache_manager.simulate_forward(sparse_idx_test, reload_type= "DFR-reload")
    
    if need_plot:
        cache_manager.plot_metrics()
    
if __name__ == "__main__":
    main()