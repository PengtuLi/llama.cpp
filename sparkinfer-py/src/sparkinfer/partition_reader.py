import os
import sys
from collections import defaultdict
from typing import Any, Dict, Tuple
import concurrent.futures
import json

import numpy as np


def load_partition(path: str):
    """
    Read the partition file.
    """
    partition = []
    group_vertices = defaultdict(list)

    with open(path, 'r', buffering=1 << 20) as f:  # 1 MB buffer
        append = partition.append
        for vid, line in enumerate(f):
            gid = int(line)
            append(gid)
            group_vertices[gid].append(vid)

    return partition, group_vertices


def parse_graph(path: str, partition: list[int], num_groups: int):
    """
    Compute vertex counts and intra-group edge-weight sums.
    Assumes METIS-like format with 1-based vertex ids and
    (neighbor, weight) pairs.
    """
    vertex_count = [0] * num_groups
    for gid in partition:
        vertex_count[gid] += 1

    edge_weight_sum = [0] * num_groups

    with open(path, 'r', buffering=1 << 20) as f:
        num_vertices = int(next(f).split()[0])  # first line
        if num_vertices != len(partition):
            raise ValueError("Vertex count mismatch between graph and partition")

        for vid, line in enumerate(f):
            if not line.strip():
                continue
            nums = [int(x) for x in line.split()]
            neighbors = nums[::2]
            weights = nums[1::2]

            gid = partition[vid]
            for nid, w in zip(neighbors, weights):
                if partition[nid - 1] == gid:  # METIS ids are 1-based
                    edge_weight_sum[gid] += w

    # Each undirected edge counted twice
    return vertex_count, [w // 2 for w in edge_weight_sum]

def process_single_layer(args: Tuple[int, str, str]) -> Tuple[int, Dict[str, Any]]:
    """
    处理单个层（一对图和分区文件）的完整逻辑。
    这个函数将被每个线程独立调用。
    
    返回: (层ID, 包含该层所有数据的字典)
    """
    layer_id, graph_path, part_path = args
    print(f"[Thread] 开始处理第 {layer_id} 层...")
    try:
        partition, group_vertices = load_partition(part_path)
        if not partition: # 如果分区文件为空
            num_groups = 0
        else:
            num_groups = max(partition) + 1
        
        v_cnt, e_sum = parse_graph(graph_path, partition, num_groups)
        
        neuron_to_group_map = np.full(sum(v_cnt), -1, dtype=np.int32)
        for group_id, vertices in group_vertices.items():
            # 将列表转换为numpy数组以便进行高级索引
            v_indices = np.array(vertices, dtype=np.int32)
            # 在这些神经元的位置上，写入它们所属的组ID
            neuron_to_group_map[v_indices] = group_id
        
        result_data = {
            "vertex_count": v_cnt,
            "edge_weight_sum": e_sum,
            "group_vertices": dict(group_vertices),
            "neuron_to_group_map": [int(x) for x in neuron_to_group_map],
            "status": "success"
        }
        
        print(f"[Thread] 第 {layer_id} 层处理完成。")
        return layer_id, result_data
    except Exception as e:
        print(f"[Thread] 处理第 {layer_id} 层时发生错误: {e}")
        return layer_id, {"status": "error", "message": str(e)}

def load_all_layers_concurrently(
    graph_dir: str, 
    partition_dir: str, 
    num_layers: int,
    num_workers: int = 8
) -> Dict[int, Dict[str, Any]]:
    """
    使用多线程并发地从两个独立的文件夹中读取所有的图和其对应的分割。
    
    参数:
    - graph_dir: 包含所有图文件 (.metis) 的文件夹路径。
    - partition_dir: 包含所有分割文件 (.part) 的文件夹路径。
    - num_layers: 总共要处理的层数。
    
    返回:
    一个字典，键是层ID，值是该层的所有处理结果。
    """
    tasks = []
    # 从两个不同的目录构建文件路径
    for i in range(num_layers):
        graph_path = os.path.join(graph_dir, f"layer_{i}.graph")
        part_path = os.path.join(partition_dir, f"layer_{i}.part")
        tasks.append((i, graph_path, part_path))

    all_layer_data = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_layer = executor.map(process_single_layer, tasks)
        for layer_id, result in future_to_layer:
            all_layer_data[layer_id] = result
            
    return dict(sorted(all_layer_data.items()))

# =================================================================
#  新增的缓存读写函数
# =================================================================

def save_results_to_json(data: Dict, filepath: str):
    """将处理结果保存为JSON文件。"""
    print(f"正在将结果保存到缓存文件: {filepath} ...")
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        print("保存成功。")
    except IOError as e:
        print(f"错误: 无法写入文件 {filepath}。原因: {e}")

def load_results_from_json(filepath: str) -> Dict[int, Any]:
    """从JSON缓存文件中加载结果。"""
    
    def deep_convert_keys_to_int(d):
        if not isinstance(d, dict):
            return d
        new_dict = {}
        for k, v in d.items():
            try:
                key = int(k)
            except ValueError:
                key = k  # 保留原字符串键，如果无法转成 int
            # 递归处理值
            new_dict[key] = deep_convert_keys_to_int(v)
        return new_dict

    print(f"发现缓存文件，正在从中加载数据: {filepath} ...")
    try:
        with open(filepath, 'r') as f:
            loaded_data = json.load(f)
            # JSON的键总是字符串，需要将其转换回整数的层ID
            return deep_convert_keys_to_int(loaded_data)
    except (IOError, json.JSONDecodeError) as e:
        print(f"错误: 无法读取或解析缓存文件 {filepath}。原因: {e}")
        return {}

def get_all_layer_data(
    graph_dir: str,
    partition_dir: str,
    num_layers: int,
    use_cache: bool = True,
    cache_filepath: str = "neuron_partition_cache.json"
) -> Dict[int, Dict[str, Any]]:
    """
    获取所有层的处理数据，按需使用缓存或并发处理。

    参数:
        graph_dir (str): 包含图文件的文件夹路径。
        partition_dir (str): 包含分区文件的文件夹路径。
        num_layers (int): 要处理的总层数。默认为 32。
        use_cache (bool): 是否使用缓存。默认为 True。
        force_rerun (bool): 是否强制重新处理，忽略现有缓存。默认为 False。
        cache_filepath (str): 缓存文件的路径。默认为 "processed_data_cache.json"。

    返回:
        Dict[int, Dict[str, Any]]: 一个字典，键是层ID，值是该层的处理结果。
                                     如果处理失败，则返回 None。
    """
    all_data = None

    if use_cache and os.path.exists(cache_filepath):
        print(f"发现缓存文件，正在从 '{cache_filepath}' 加载数据...")
        all_data = load_results_from_json(cache_filepath)
        if all_data:
            print("从缓存中成功加载数据！")
            return all_data

    if not os.path.isdir(graph_dir) or not os.path.isdir(partition_dir):
        print(f"错误: 请确保图文件夹 '{graph_dir}' 和分区文件夹 '{partition_dir}' 都存在。")
        return {}

    print(f"开始从以下文件夹并发读取 {num_layers} 层数据...")
    print(f"  - 图文件来源: '{graph_dir}'")
    print(f"  - 分区文件来源: '{partition_dir}'")
    
    all_data = load_all_layers_concurrently(graph_dir, partition_dir, num_layers)
    
    if use_cache:
        print(f"正在将新处理的结果保存到缓存文件: {cache_filepath}")
        save_results_to_json(all_data, cache_filepath)
        
    return all_data

# =================================================================
#  命令行执行入口 (作为使用示例)
# =================================================================

if __name__ == "__main__":
    # 这个部分现在只作为演示，展示如何调用 get_all_layer_data 函数
    
    # 1. 解析命令行参数
    args = sys.argv[1:]

    if len(args) < 4:
        print("用法: python graph_processor.py <graph_directory> <partition_directory> num_layers cache_filepath")
        sys.exit(1)
        
    graph_data_dir = args[0]
    partition_data_dir = args[1]
    num_layers = int(args[2])
    cache_filepath = args[3]

    # 2. 调用核心函数
    processed_data = get_all_layer_data(
        graph_dir=graph_data_dir,
        partition_dir=partition_data_dir,
        num_layers=num_layers,
        use_cache=True,
        cache_filepath=cache_filepath,
    )

    # 3. 处理返回结果
    if processed_data:
        print("\n========================================")
        print("数据准备就绪。")
        print("========================================\n")
        
        # 打印摘要
        successful_layers = sum(1 for data in processed_data.values() if data.get("status") == "success")
        
        for layer_id, data in processed_data.items():
            if data.get("status") == "success":
                print(f"--- 第 {layer_id} 层 (成功) ---")
                total_internal_weight = sum(data.get('edge_weight_sum', []))
                print(f"  分区数量: {len(data.get('vertex_count', []))}")
                print(f"  总内部边权重和: {total_internal_weight}")
        
        print(f"\n--- 整体统计 ---")
        print(f"成功处理层数: {successful_layers}/{len(processed_data)}")
    else:
        print("\n处理失败，未能获取任何数据。")