import argparse
import os
from pathlib import Path

from .export_split import export_split
from .partition_reader import get_all_layer_data

'''
python -m sparkinfer \
    --model-path=/share/models/prosparse-7b-gguf-w-our-predictor \
    --neuron=11008 \
    --neuron-capacity=314184 \
    --layer=32 \
    --vram-capacity=5187846144 \
    --output=/share/models/prosparse-7b-gguf-w-our-predictor/prosparse-7b.gguf.sparkinfer_split_idx
'''
if __name__ == "__main__":

    # Set up command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path',
                        type=str,
                        required=True,
                        help='Path to the model folder.')
    parser.add_argument('--neuron', type=int, default=8192 * 4, help='Total number of neurons in the network.')
    parser.add_argument('--neuron-capacity',
                        type=int,
                        default=int(8192 * 4 * 32 * 0.1),
                        help='Total VRAM capacity for the model.')
    parser.add_argument('--layer', type=int, required=True, help='Total number of layers in the neural network.')
    parser.add_argument('--vram-capacity', type=int, help='Total VRAM capacity (Bytes) available for splitting')
    parser.add_argument('--output', type=str, required=True, help='File path for the output gguf file.')

    args = parser.parse_args()

    print("args:", args)
    
    # first apply graph construct to the neurons
    
    # then apply partition to the neurons based on graph edge weight
    
    # then export the partitioned neurons to the output file
    
    # read partitioned neurons, and generate the offload strategy based on vram_capacity and layer-wise neuron activation count
    
    neuron_partition=get_all_layer_data(
        graph_dir="/share/datasets/graph_partitioning/prosparse-llama-2-7b/raw_metis_graph/",
        partition_dir="/share/datasets/graph_partitioning/prosparse-llama-2-7b/partitioned_graph/",
        num_layers=args.layer,
        use_cache=True,
        cache_filepath=os.path.join(args.model_path, "neuron_partition_cache.json")
    )
    
    export_split(output_path=args.output,
                 neuron_partition=neuron_partition,
                 neuron=args.neuron,
                 vram_capacity=args.vram_capacity,
                 neuron_capacity=args.neuron_capacity)

    print(f"Exported to {args.output}")
