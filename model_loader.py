from typing import Dict
import json
from pathlib import Path
from tqdm import tqdm
from safetensors import safe_open
import torch
import torch.nn as nn
from Model.Llama_3.Meta_model import Transformer, ModelArgs


# checkpoint path
try:
    with open("checkpoint.json", "r") as file:
        data = json.load(file)
        print("Checkpoint path loaded.")
except FileNotFoundError:
    print("Error: The file checkpoint path file was not found.")
except json.JSONDecodeError as e:
    print(f"Error decoding JSON: {e}")


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
        if new_key == 'embed_tokens.weight':
            new_key = 'tok_embeddings.weight'
        elif new_key == 'lm_head.weight':
            new_key = 'output.weight'

        meta_state_dict[new_key] = value

    return meta_state_dict


def load_weights(model: nn.Module, model_dir: Path, hf: bool):
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


def load(model_id: str, hf: bool = False):
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

    # load the weights into model
    load_weights(model, model_dir, hf)

    print("Moving model to GPU...")
    model.to(device)
    print("Model successfully moved to GPU.")


if __name__ == "__main__":
    load("Llama_3_8B", hf=False)
