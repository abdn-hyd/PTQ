from typing import Dict
import json
from pathlib import Path
from tqdm import tqdm
from safetensors import safe_open
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
import itertools
from Model.Llama_3.Meta_model import Transformer, ModelArgs
from Model.Llama_3.Meta_tokenizer import Tokenizer


# Llama-3 path
model_path = {
    "Llama_3_1B": "/Users/gunneo/.cache/modelscope/hub/models/LLM-Research/Llama-3___2-1B-Instruct/",
    "Llama_3_3B": "/Users/gunneo/.cache/modelscope/hub/models/LLM-Research/Llama-3___2-3B-Instruct/",
    "Llama_3_8B": "/Users/gunneo/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3___1-8B-Instruct/",
}

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_args(config: Dict, hf: bool = False):
    if hf:
        model_args_dict = {
            "hf": True,
            "dim": config["hidden_size"],
            "n_layers": config["num_hidden_layers"],
            "n_heads": config["num_attention_heads"],
            "n_kv_heads": config.get("num_key_value_heads", None),
            "vocab_size": config["vocab_size"],
            "multiple_of": config.get("multiple_of", 256),
            "ffn_dim_multiplier": config.get("ffn_dim_multiplier", None),
            "intermediate_size": config["intermediate_size"],
            "norm_eps": config["rms_norm_eps"],
            "rope_theta": float(config.get("rope_theta", 500000)),
        }
        model_args = ModelArgs(**model_args_dict)
    else:
        model_args = ModelArgs(**config)
    return model_args


def rename_hf_keys_to_meta(hf_state_dict):
    """
    hf's format into Meta's format
    """
    meta_state_dict = {}
    for key, value in hf_state_dict.items():
        new_key = key
        if new_key.startswith('model.'):
            new_key = new_key[6:]

        # Hugging Face's self_attn -> Meta's attention
        if 'self_attn' in new_key:
            new_key = new_key.replace('self_attn.q_proj', 'attention.wq')
            new_key = new_key.replace('self_attn.k_proj', 'attention.wk')
            new_key = new_key.replace('self_attn.v_proj', 'attention.wv')
            new_key = new_key.replace('self_attn.o_proj', 'attention.wo')

        # Hugging Face's mlp -> Meta's feed_forward
        if 'mlp' in new_key:
            new_key = new_key.replace('mlp.gate_proj', 'feed_forward.w1')
            new_key = new_key.replace('mlp.up_proj', 'feed_forward.w3')
            new_key = new_key.replace('mlp.down_proj', 'feed_forward.w2')

        # Hugging Face's layernorm -> Meta's norm
        if 'input_layernorm' in new_key:
            new_key = new_key.replace('input_layernorm', 'attention_norm')
        if 'post_attention_layernorm' in new_key:
            new_key = new_key.replace('post_attention_layernorm', 'ffn_norm')

        # --- Special naming system ---
        # Eg. embed_tokens.weight -> tok_embeddings.weight
        if new_key == 'embed_tokens.weight':
            new_key = 'tok_embeddings.weight'
        elif new_key == 'lm_head.weight':
            new_key = 'output.weight'

        meta_state_dict[new_key] = value

    return meta_state_dict


def load_weights(model: nn.Module, model_dir: Path, hf: bool = False):
    if hf:
        index_path = model_dir / "model.safetensors.index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"File not found: {index_path}")

        with open(index_path, 'r') as f:
            index_data = json.load(f)

        weight_map = index_data['weight_map']
        open_files = {}
        hf_loaded_state_dict = {}

        print("Start to load the weight!")
        progress_bar = tqdm(weight_map.items(),
                            desc="Loading Tensors", unit="tensor")

        for tensor_name, shard_filename in progress_bar:
            shard_path = model_dir / shard_filename
            if shard_filename not in open_files:
                progress_bar.write(f"Opening new shard file: {shard_filename}")
                open_files[shard_filename] = safe_open(
                    shard_path, framework="pt", device="cpu")
            tensor = open_files[shard_filename].get_tensor(tensor_name)
            hf_loaded_state_dict[tensor_name] = tensor
        print(f"\nAll {len(weight_map)} tensors have been successfully loaded!")

        meta_state_dict = rename_hf_keys_to_meta(hf_loaded_state_dict)

        try:
            # ignore cache k and v
            model.load_state_dict(meta_state_dict, strict=False)
            print("Model loaded successfully！")
        except RuntimeError as e:
            print("\nLoad state_dict with error.")
            print(f"Error: {e}")
    else:
        shard_files = sorted(model_dir.glob("consolidated.*.pth"))

        if not shard_files:
            raise FileNotFoundError(
                f"No 'consolidated.*.pth' files found in {model_dir}. "
                "Please make sure you have the correct path to the official Meta weights."
            )

        print(f"Found {len(shard_files)} model shard files.")
        meta_state_dict = {}
        progress_bar = tqdm(
            shard_files, desc="Loading Model Shards", unit="shard")

        for shard_path in progress_bar:
            progress_bar.write(f"Loading shard: {shard_path.name}")
            shard_state_dict = torch.load(shard_path, map_location="cpu")
            meta_state_dict.update(shard_state_dict)
            del shard_state_dict

        print(
            f"\nAll {len(meta_state_dict)} tensors from {len(shard_files)} shards have been loaded!")

        try:
            model.load_state_dict(meta_state_dict, strict=False)
            print("Model loaded successfully!")
        except RuntimeError as e:
            print("\nLoad state_dict encountered an error.")
            print(f"Error: {e}")
    return model


def visualize_weights(model: nn.Module, type: str, save_dir: Path, sample_rate: int = 4):
    print("\nStarting weights visualization...")
    save_dir.mkdir(parents=True, exist_ok=True)

    # determine the weight type
    type_map = {
        "query": "wq",
        "key": "wk",
        "value": "wv",
    }

    model.eval()
    for i, layer in enumerate(tqdm(model.layers, desc="Visualizing Layers")):

        weights_component = getattr(layer.attention, type_map[type])
        weights = weights_component.weight.detach().cpu().float().numpy()
        out_features, in_features = weights.shape

        x = np.arange(0, in_features, sample_rate)
        y = np.arange(0, out_features, sample_rate)
        X, Y = np.meshgrid(x, y)
        Z = np.abs(weights[::sample_rate, ::sample_rate])

        fig = plt.figure(figsize=(24, 16))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
        ax.set_xlabel('Input Features Dimension')
        ax.set_ylabel('Output Features Dimension')
        ax.set_zlabel('Weight Value')
        ax.set_title(f'Layer {i}: Query Weight Distribution (3D Sampled)')

        save_path_3d = save_dir / f"layer_{i:02d}_{type}_weights_3d.png"
        plt.savefig(save_path_3d, dpi=300, bbox_inches="tight")

        plt.close(fig)

    print(f"\nVisualization complete. All images saved in '{save_dir}'.")


def visualize_activations(model: nn.Module, tokenizer: Tokenizer, dataset_samples: list, save_dir: Path, max_seq_len: int = 512, batch_size: int = 32, sample_rate: int = 4):
    print("\nStarting activations visualization...")
    save_dir.mkdir(parents=True, exist_ok=True)

    texts = [sample['text'] for sample in dataset_samples]

    # --- Tokenization (remains the same) ---
    all_tokens = []
    for text in tqdm(texts, desc="Tokenizing samples"):
        tokens = tokenizer.encode(text, bos=True, eos=False)
        tokens = tokens[:max_seq_len]
        padding_needed = max_seq_len - len(tokens)
        tokens.extend([tokenizer.eos_id] * padding_needed)
        all_tokens.append(tokens)

    # --- Hook Setup (remains the same) ---
    activations = {}

    def get_activation(name):
        def hook(_module, _input, output):
            # This will capture the activation for the CURRENT batch
            activations[name] = output.detach()
        return hook

    hooks = []
    for i, layer in enumerate(model.layers):
        hook = layer.attention.wq.register_forward_hook(
            get_activation(f"layer_{i}"))
        hooks.append(hook)

    model.eval()

    # --- BATCH PROCESSING & ACTIVATION ACCUMULATION ---
    activation_sums = {}
    num_samples_processed = 0

    with torch.no_grad():
        # Loop through all tokens in batches
        for i in tqdm(range(0, len(all_tokens), batch_size), desc="Forward pass in batches"):
            # Create a tensor for the current batch
            batch_tokens = all_tokens[i:i + batch_size]
            current_batch_size = len(batch_tokens)
            if current_batch_size == 0:
                continue

            input_tensor = torch.tensor(
                batch_tokens, dtype=torch.long, device=device)

            # Run the forward pass for this batch
            model(input_tensor, start_pos=0)

            # After the forward pass, the `activations` dict is populated
            # with the activations for the current batch. Now, accumulate them.
            for name, batch_activation in activations.items():
                # Sum the activations across the batch dimension and move to CPU to save VRAM
                batch_sum = batch_activation.sum(dim=0).cpu()

                if name not in activation_sums:
                    activation_sums[name] = batch_sum
                else:
                    activation_sums[name] += batch_sum

            num_samples_processed += current_batch_size

    print(
        f"\nForward pass complete. Total samples processed: {num_samples_processed}.")

    # --- Cleanup Hooks (remains the same) ---
    for hook in hooks:
        hook.remove()

    # --- VISUALIZATION OF AVERAGED ACTIVATIONS ---
    print("Visualizing averaged activations...")
    for i, (name, total_activation_sum) in enumerate(tqdm(activation_sums.items(), desc="Generating 3D Surfaces")):
        # Calculate the mean activation (same as before)
        mean_activation = (total_activation_sum /
                           num_samples_processed).float().numpy()

        num_tokens, num_channels = mean_activation.shape

        x = np.arange(0, num_channels, sample_rate)
        y = np.arange(0, num_tokens, sample_rate)
        X, Y = np.meshgrid(x, y)
        Z = np.abs(mean_activation[::sample_rate, ::sample_rate])

        fig = plt.figure(figsize=(24, 16))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

        ax.set_xlabel('Channels')
        ax.set_ylabel('Tokens')
        ax.set_zlabel('Activation Value')
        ax.set_title(f'Layer {i}: Mean Query Activation 3D Surface (Batched)')

        save_path = save_dir / f"layer_{i:02d}_query_activation_3d_surface.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    print(
        f"\nActivation visualization complete. All images saved in '{save_dir}'.")


def main(model_id: str, vis_res: str, hf: bool = False):
    model_dir = Path(model_path[model_id])
    # load the config file
    if hf:
        config_path = model_dir / "config.json"
    else:
        config_path = model_dir / "original/params.json"
        model_dir = model_dir / "original/"
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{config_path}' was not found.")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    # load model args
    model_args = load_args(config, hf)

    # load model with model_args
    model = Transformer(model_args).to(torch.bfloat16)

    suffix = "hf" if hf else "meta"

    # load the weights into model
    load_weights(model, model_dir, hf)

    print("Moving model to GPU...")
    model.to(device)
    print("Model successfully moved to GPU.")

    # visualization query weight
    query_save_dir = Path(vis_res) / f"Query_{suffix}/"
    visualize_weights(model, "query", query_save_dir)

    # visualization key weight
    key_save_dir = Path(vis_res) / f"Key_{suffix}/"
    visualize_weights(model, "key", key_save_dir)

    # visualization value weight
    value_save_dir = Path(vis_res) / f"Value_{suffix}/"
    visualize_weights(model, "value", value_save_dir)

    # load a piece of pile datasets
    pile_stream = load_dataset(
        "monology/pile-uncopyrighted",
        split="train",
        streaming=True
    )
    shuffled_stream = pile_stream.shuffle(seed=42, buffer_size=1000)
    random_samples = list(itertools.islice(shuffled_stream, 512))
    print(f"Download {len(random_samples)} pieces!")

    # load tokenizer
    tokenizer_path = model_dir / \
        "original/tokenizer.model" if hf else model_dir / "tokenizer.model"
    tokenizer = Tokenizer(model_path=str(tokenizer_path))

    # visualize activations
    activation_save_dir = Path(vis_res) / f"Activations_{suffix}/"

    visualize_activations(
        model=model,
        tokenizer=tokenizer,
        dataset_samples=random_samples,
        save_dir=activation_save_dir
    )


if __name__ == "__main__":
    main("Llama_3_8B", "/path/to/vis_res", hf=True)
