import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.font_manager as fm
import os

MODELS = [
	"Qwen/Qwen1.5-7B",
	"Qwen/Qwen1.5-14B",
	"allenai/OLMo-2-0425-1B",
	"allenai/OLMo-2-1124-13B",
]

PROMPT_ADD = "general-cot"
PROMPT_SUB = "general-direct"
DATA_NAMES = ["gsm8k", "math"]  # , "svamp"


def compute_von_neumann_entropy(K):
	"""Von Neumann entropy: S(rho) = -Tr(rho log rho), rho = K / tr(K)."""
	eigenvalues = torch.linalg.eigvalsh(K)
	eigenvalues = torch.sort(eigenvalues, dim=0, descending=True).values
	positive_evals = eigenvalues[eigenvalues > 1e-10]
	trace_K = torch.sum(positive_evals)
	normalized_eigenvalues = positive_evals / (trace_K)

	entropy = torch.where(
		normalized_eigenvalues > 0,
		normalized_eigenvalues * torch.log(normalized_eigenvalues),
		torch.zeros_like(normalized_eigenvalues),
	)

	von_neumann_entropy = -torch.sum(entropy)
	print(von_neumann_entropy)

	return von_neumann_entropy.item()


def main():
	fm.fontManager.addfont(os.path.expanduser("~/.fonts/JuliaMono-Regular.ttf"))
	fm.fontManager.addfont(os.path.expanduser("~/.fonts/NotoSansMono-Regular.ttf"))
	fm.fontManager.addfont(os.path.expanduser("~/.fonts/NotoSansMono-Bold.ttf"))

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device: {device}")

	plt.rcParams.update(
		{
			"text.usetex": False,
			"font.family": "monospace",
			"font.monospace": ["Noto Sans Mono", "JuliaMono", "DejaVu Sans Mono"],
			"mathtext.fontset": "dejavusans",
			"figure.dpi": 1000,
			"savefig.dpi": 1000,
			"axes.titlesize": 23,
			"axes.titleweight": "bold",
			"axes.labelsize": 22,
			"xtick.labelsize": 20,
			"axes.labelweight": "bold",
			"ytick.labelsize": 20,
			"legend.fontsize": 16,
			"hatch.linewidth": 0.5,
		}
	)

	for m in MODELS:
		OUTPUT_FILE = f"./figures/spectral_entropy/spectral_entropy_add_{m.split('/')[-1]}_sub_{m.split('/')[-1]}.pdf"

		fig, axes = plt.subplots(1, 2, figsize=(12, 5))

		sample_sizes = [4, 16, 64, 128, 256, 512]
		markers = ["o", "X", "s", "^", "D", "v", "P", "H"]
		cmap = plt.get_cmap("viridis")
		n_samples_list = len(sample_sizes)
		t = np.linspace(0.1, 0.95, n_samples_list)
		colors = [cmap(v) for v in t]

		for dataset_idx, data_name in enumerate(DATA_NAMES):
			add_file = f"/projects/llms-lab/transfer_compare/spectral_entropy/{m.replace('/', '_')}_{data_name}_test_{PROMPT_ADD}.pth"
			sub_file = f"/projects/llms-lab/transfer_compare/spectral_entropy/{m.replace('/', '_')}_{data_name}_test_{PROMPT_SUB}.pth"

			print(f"\nLoading data for {data_name}...")
			add_data = torch.load(add_file, map_location=device)
			sub_data = torch.load(sub_file, map_location=device)

			num_ex = min(add_data.shape[1], sub_data.shape[1])

			matrix = (add_data[:, :num_ex] - sub_data[:, :num_ex]).to(torch.float64)
			print(f"Matrix shape: {matrix.shape}")

			n_layers = matrix.shape[0]
			ax = axes[dataset_idx]

			for sample_idx, n_samples in enumerate(sample_sizes):
				print(
					f"\nComputing kernel matrices for {data_name}, {n_layers} layers and {n_samples} samples..."
				)

				spectral_entropies = []

				for layer_idx in range(n_layers):
					layer_activations = matrix[layer_idx, :n_samples]

					K = torch.matmul(
						layer_activations.squeeze(1).T, layer_activations.squeeze(1)
					)

					entropy = compute_von_neumann_entropy(K)
					spectral_entropies.append(entropy)

					if layer_idx % 5 == 0:
						print(f"Layer {layer_idx}: Spectral Entropy = {entropy:.4f}")

				layers = list(range(n_layers))
				ax.plot(
					layers,
					spectral_entropies,
					marker=markers[sample_idx],
					linewidth=3,
					markersize=9,
					markeredgecolor=colors[sample_idx],
					color=colors[sample_idx],
					label=r"$\mathrm{n}$" + f"={n_samples}",
				)

			ax.set_xlabel("Layer")
			if dataset_idx == 0:
				ax.set_ylabel("Spectral Entropy (nats)")
			leg = ax.legend(loc="upper left", title="# Samples", title_fontsize=20)
			leg.get_frame().set_alpha(0.2)
			ax.set_title(f"{data_name.upper()}")
			ax.grid(True, alpha=0.3)
			ax.set_axisbelow(True)

			plt.subplots_adjust(left=0.1, right=0.98, top=0.9, bottom=0.19)

		torch.cuda.empty_cache()

		Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(OUTPUT_FILE, dpi=200)
		print(f"\nPlot saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
	main()
