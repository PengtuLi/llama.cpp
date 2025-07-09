import sys
from typing import Any, Dict, List
from gguf.constants import GGMLQuantizationType
from gguf.gguf_writer import GGUFWriter
from pathlib import Path
import gguf
import numpy as np
from ordered_set import OrderedSet


class NeuronPartition:
    def __init__(self, partition: Dict[int, Dict[str, Any]], neuron: int):
        self.partition = partition
        self.layer_buffer_neuron_size: List[int] = []
        self.total_neurons_in_layer = neuron
        self.group_size = partition[0]["vertex_count"][0]
        self.group_count = int(neuron / self.group_size)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        return self.partition[item]

    # GTODO: we use powerinfer activation like data to split layer buffer or just use partition graph weights?
    def cal_layer_buffer_size(self, neuron_capacity):
        result = []
        for i, partition in self.partition.items():
            non_zero_count = sum(partition["edge_weight_sum"])
            result.append(
                non_zero_count,
            )
        result = [x / sum(result) for x in result]
        print("partitioned layer buffer size propotion:", [round(x, 3) for x in result])
        # calculate the layer buffer size based on neuron and neuron_capacity
        result = [
            min(int(x * neuron_capacity), self.total_neurons_in_layer) for x in result
        ]
        print("partitioned layer buffer size:", result)
        self.layer_buffer_neuron_size = result

    def append_gpu_idx(self, gguf_writer: GGUFWriter, layer_idx: int) -> None:
        """
        根据分组权重选择神经元放入GPU，并将所有相关信息写入GGUF文件。
        """
        layer_data = self.partition[layer_idx]

        if layer_data.get("status") != "success":
            raise ValueError(
                f"Layer {layer_idx} data is not ready or has failed: {layer_data.get('status', 'unknown status')}"
            )

        neuron_capacity_for_layer = self.layer_buffer_neuron_size[layer_idx]

        if neuron_capacity_for_layer == 0 or self.total_neurons_in_layer == 0:
            raise ValueError(
                f"Layer {layer_idx} has zero neuron capacity or total neurons: {neuron_capacity_for_layer}, {self.total_neurons_in_layer}"
            )

        group_weights = layer_data["edge_weight_sum"]
        group_vertices_map = layer_data["group_vertices"]

        # select groups based on their weights
        groups_with_info = []
        for group_id, weight in enumerate(group_weights):
            if group_id in group_vertices_map.keys():
                vertices = group_vertices_map[group_id]
                if vertices:
                    groups_with_info.append((weight, group_id, vertices))
            else:
                raise ValueError(
                    f"Group {group_id} not found in group_vertices_map for layer {layer_idx}"
                )
        groups_with_info.sort(key=lambda x: x[0], reverse=True)

        # offloaded neurons Idx and group IDs
        gpu_neurons_set = OrderedSet([])  # 有序集合
        gpu_group_set = OrderedSet([])
        for weight, group_id, vertices in groups_with_info:
            if len(gpu_neurons_set) + len(vertices) <= neuron_capacity_for_layer:
                gpu_neurons_set.update(vertices)
                gpu_group_set.add(group_id)

        print(f"\n--- Layer {layer_idx} ---")
        print(
            f"Target GPU neurons: {neuron_capacity_for_layer}, Selected: {len(gpu_neurons_set)}"
        )

        # --- GGUF write ---

        # a) ffn_gpu_neu_mask (位图)
        ffn_gpu_neu_mask = np.zeros(self.total_neurons_in_layer, dtype=np.int32)
        if gpu_neurons_set:
            selected_indices = list(gpu_neurons_set)
            ffn_gpu_neu_mask[selected_indices] = 1
        key_gpu_neu_mask = f"blk.{layer_idx}.ffn_gpu_neu_mask"
        gguf_writer.add_tensor(
            name=key_gpu_neu_mask,
            tensor=ffn_gpu_neu_mask,
            raw_shape=ffn_gpu_neu_mask.shape[::-1],
            raw_dtype=GGMLQuantizationType.I32,
        )
        print(
            f"  {key_gpu_neu_mask} => shape: {ffn_gpu_neu_mask.shape}, dtype: GGMLQuantizationType.I32, {ffn_gpu_neu_mask.nbytes / 1024 / 1024:.3f} MiB"
        )

        # b) ffn_gpu_neu_idx (索引列表)
        ffn_gpu_neu_idx = np.sort(np.array(list(gpu_neurons_set), dtype=np.int32))
        key_gpu_neuron_idx = f"blk.{layer_idx}.ffn_gpu_neu_idx"
        gguf_writer.add_tensor(
            name=key_gpu_neuron_idx,
            tensor=ffn_gpu_neu_idx,
            raw_shape=ffn_gpu_neu_idx.shape[::-1],
            raw_dtype=GGMLQuantizationType.I32,
        )
        print(
            f"  {key_gpu_neuron_idx} => shape: {ffn_gpu_neu_idx.shape}, dtype: GGMLQuantizationType.I32, {ffn_gpu_neu_idx.nbytes / 1024 / 1024:.3f} MiB"
        )

        # c) ffn_gpu_group_mask
        ffn_gpu_group_mask = np.zeros(self.group_count, dtype=np.int32)
        if gpu_group_set:
            selected_indices = list(gpu_group_set)
            ffn_gpu_group_mask[selected_indices] = 1
        key_ffn_gpu_group_mask = f"blk.{layer_idx}.ffn_gpu_group_mask"
        gguf_writer.add_tensor(
            name=key_ffn_gpu_group_mask,
            tensor=ffn_gpu_group_mask,
            raw_shape=ffn_gpu_group_mask.shape[::-1],
            raw_dtype=GGMLQuantizationType.I32,
        )
        print(
            f"  {key_ffn_gpu_group_mask} => shape: {ffn_gpu_group_mask.shape}, dtype: GGMLQuantizationType.I32, {ffn_gpu_group_mask.nbytes / 1024 / 1024:.3f} MiB"
        )

        # d) ffn_gpu_group_idx (组ID列表)
        ffn_gpu_group_idx = np.array(list(gpu_group_set), dtype=np.int32)
        key_ffn_gpu_group_idx = f"blk.{layer_idx}.ffn_gpu_group_idx"
        gguf_writer.add_tensor(
            name=key_ffn_gpu_group_idx,
            tensor=ffn_gpu_group_idx,
            raw_shape=ffn_gpu_group_idx.shape[::-1],
            raw_dtype=GGMLQuantizationType.I32,
        )
        print(
            f"  {key_ffn_gpu_group_idx} => shape: {ffn_gpu_group_idx.shape}, dtype: GGMLQuantizationType.I32, {ffn_gpu_group_idx.nbytes / 1024 / 1024:.3f} MiB"
        )

        # e) ffn_neuron_to_group_map (神经元 -> 组ID 的映射)
        ffn_neuron_to_group_map = np.array(
            layer_data["neuron_to_group_map"], dtype=np.int32
        )
        key_ffn_neuron_to_group_map = f"blk.{layer_idx}.ffn_neuron_to_group_map"
        print(
            f"  {key_ffn_neuron_to_group_map} => shape: {ffn_neuron_to_group_map.shape}, dtype: GGMLQuantizationType.I32, {ffn_neuron_to_group_map.nbytes / 1024 / 1024:.3f} MiB"
        )
        gguf_writer.add_tensor(
            name=key_ffn_neuron_to_group_map,
            tensor=ffn_neuron_to_group_map,
            raw_shape=ffn_neuron_to_group_map.shape[::-1],
            raw_dtype=GGMLQuantizationType.I32,
        )


def export_split(
    output_path: str,
    neuron_partition: Dict[int, Dict[str, Any]],
    neuron: int,
    vram_capacity: int,
    neuron_capacity: int,
):
    sparkinfer_np = NeuronPartition(neuron_partition, neuron)
    sparkinfer_np.cal_layer_buffer_size(neuron_capacity)

    gguf_out = GGUFWriter(output_path, "sparkinfer.gpu_index")
    for i, partition in neuron_partition.items():
        sparkinfer_np.append_gpu_idx(gguf_out, i)

    # set kvs
    gguf_out.add_block_count(len(neuron_partition))
    gguf_out.add_uint64(gguf.Keys.Split.VRAM_CAPACITY, vram_capacity)
    gguf_out.add_uint64(gguf.Keys.Split.LAYER_NEURON_COUNT, neuron)
    gguf_out.add_uint64(gguf.Keys.Split.LAYER_GROUP_COUNT, sparkinfer_np.group_count)

    gguf_out.write_header_to_file()
    gguf_out.write_kv_data_to_file()
    gguf_out.write_tensors_to_file()
    gguf_out.close()

    # post-process: write another unique file header to distinguish from the origianl GGUF file
    # with open(output_path, "r+b") as fout:
    #     GGUF_MAGIC = int.from_bytes(b"GGUF", "little")
    #     fout.write(struct.pack("<I", GGUF_MAGIC))
    #     fout.write(struct.pack("<I", 3))

    print(f"exported GPU index to {output_path}")
