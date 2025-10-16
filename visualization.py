from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
import itertools
from Model.Llama_3.Meta_tokenizer import Tokenizer
from model_loader import load

device = "cuda" if torch.cuda.is_available() else "cpu"


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
    load(model_id=model_id, hf=hf)

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
    main("Llama_3_8B", "/path/to/vis_res", hf=False)
