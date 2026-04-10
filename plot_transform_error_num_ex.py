import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.font_manager as fm
import os

prompt = "general-direct"
rank = 4

MODEL_PAIRS = [
	("Qwen/Qwen1.5-7B", "Qwen/Qwen1.5-14B"),
	("Qwen/Qwen1.5-14B", "Qwen/Qwen1.5-7B"),
	("allenai/OLMo-2-1124-7B", "allenai/OLMo-2-0425-1B"),
]

DATA_NAMES = ["gsm8k", "math"]

sample_sizes = [4, 16, 64, 128, 256, 512]
markers = ["o", "X", "s", "^", "D", "v", "P", "H"]
cmap = plt.get_cmap("viridis")
n_samples_list = len(sample_sizes)
t = np.linspace(0.1, 0.95, n_samples_list)
colors = [cmap(v) for v in t]


def load_hidden_states(file_path, device):
	"""Load hidden states from file."""
	print(f"Loading {file_path}...")
	data = torch.load(file_path, map_location=device)

	# if isinstance(data, dict):
	# 	print(f"Keys: {data.keys()}")
	# 	for key in ["hidden_states", "activations", "hs"]:
	# 		if key in data:
	# 			return data[key]
	# 	return data[list(data.keys())[0]]
	return data


def load_transformation(file_path, device):
	"""Load transformation matrices from file."""
	print(f"Loading transformation from {file_path}...")
	data = torch.load(file_path, map_location=device).to(torch.bfloat16)

	# if isinstance(data, dict):
	# 	print(f"Keys: {data.keys()}")
	# 	for key in ["transformation", "transform", "W", "weight"]:
	# 		if key in data:
	# 			return data[key]
	# 	return data[list(data.keys())[0]]
	return data


def compute_l2_error(student, teacher, transform=None):
	"""
	Compute L2 error between transformed student and teacher.
	If transform is None, compute optimal transformation using least squares.
	"""
	# if transform is not None:
	student_transformed = torch.matmul(student, transform)

	print(student.shape, transform.shape, student_transformed.shape, teacher.shape)

	# else:
	# student_transformed, _ = torch.lstsq(teacher, student)
	# student_transformed = torch.matmul(student, student_transformed.solution)

	num_ex = min(student.shape[0], teacher.shape[0])

	error = torch.norm(
		teacher[:num_ex] - student_transformed[:num_ex], p=2
	) / torch.norm(teacher[:num_ex], p=2)
	return error.item()


def main():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device: {device}")

	fm.fontManager.addfont(os.path.expanduser("~/.fonts/JuliaMono-Regular.ttf"))
	fm.fontManager.addfont(os.path.expanduser("~/.fonts/NotoSansMono-Regular.ttf"))
	fm.fontManager.addfont(os.path.expanduser("~/.fonts/NotoSansMono-Bold.ttf"))

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

	for base_model, sub_model in MODEL_PAIRS:
		OUTPUT_FILE = f"./figures/transform_error_num_ex/transformation_error_base_rank_{rank}_{base_model.split('/')[-1]}_sub_{sub_model.split('/')[-1]}_data_gsm8k_math_prompt_{prompt}.pdf"

		BASE_FILES = {
			data: f"/projects/llms-lab/transfer_compare/spectral_entropy/{base_model.replace('/', '_')}_{data}_test_{prompt}.pth"
			for data in DATA_NAMES
		}
		SUB_FILES = {
			data: f"/projects/llms-lab/transfer_compare/spectral_entropy/{sub_model.replace('/', '_')}_{data}_test_{prompt}.pth"
			for data in DATA_NAMES
		}
		TRANSFORM_FILES = {
			data: [
				f"/projects/llms-lab/transfer_compare/spectral_entropy/num_train_samples_4/base_{base_model.split('/')[-1]}_sub_{sub_model.split('/')[-1]}_pca_data_{data}_rank_{rank}_num_train_samples_4.pth",
				f"/projects/llms-lab/transfer_compare/spectral_entropy/num_train_samples_16/base_{base_model.split('/')[-1]}_sub_{sub_model.split('/')[-1]}_pca_data_{data}_rank_{rank}_num_train_samples_16.pth",
				f"/projects/llms-lab/transfer_compare/spectral_entropy/num_train_samples_64/base_{base_model.split('/')[-1]}_sub_{sub_model.split('/')[-1]}_pca_data_{data}_rank_{rank}_num_train_samples_64.pth",
				f"/projects/llms-lab/transfer_compare/spectral_entropy/num_train_samples_128/base_{base_model.split('/')[-1]}_sub_{sub_model.split('/')[-1]}_pca_data_{data}_rank_{rank}_num_train_samples_128.pth",
				f"/projects/llms-lab/transfer_compare/spectral_entropy/num_train_samples_256/base_{base_model.split('/')[-1]}_sub_{sub_model.split('/')[-1]}_pca_data_{data}_rank_{rank}_num_train_samples_256.pth",
				f"/projects/llms-lab/transfer_compare/spectral_entropy/num_train_samples_512/base_{base_model.split('/')[-1]}_sub_{sub_model.split('/')[-1]}_pca_data_{data}_rank_{rank}_num_train_samples_512.pth",
			]
			for data in DATA_NAMES
		}

		fig, axes = plt.subplots(1, 2, figsize=(12, 5))

		for dataset_idx, data_name in enumerate(DATA_NAMES):
			base_hs = load_hidden_states(BASE_FILES[data_name], device)
			sub_hs = load_hidden_states(SUB_FILES[data_name], device)

			print(f"\n{data_name.upper()}:")
			print(f"Base shape: {base_hs.shape}")
			print(f"Sub shape: {sub_hs.shape}")

			base_layers = base_hs.shape[0]
			sub_layers = sub_hs.shape[0]

			print(f"Base layers: {base_layers}")
			print(f"Sub layers: {sub_layers}")

			ax = axes[dataset_idx]

			for idx, transform_file in enumerate(TRANSFORM_FILES[data_name]):
				transform_data = load_transformation(transform_file, device)
				print(f"\nTransform {idx + 1} shape: {transform_data.shape}")

				num_samples_str = transform_file.split("num_train_samples_")[-1].split(
					".pth"
				)[0]
				num_samples = int(num_samples_str)

				color = colors[
					len(sample_sizes) - len(TRANSFORM_FILES[data_name]) + idx
				]

				l2_errors = []

				print(f"Computing L2 errors for {base_layers} base layers...")
				for Lb in range(base_layers):
					Ls = min(
						((sub_layers - 1) * Lb) // (base_layers - 1),
						sub_layers - 1,
					)

					base_layer = base_hs[Lb]
					sub_layer = sub_hs[Ls]

					transform_layer = transform_data[Lb]

					error = compute_l2_error(sub_layer, base_layer, transform_layer)
					l2_errors.append(error)

					if Lb % 5 == 0:
						print(f"Layer {Lb} (from sub {Ls}): L2 Error = {error:.6f}")

				label = (
					r"$\mathrm{n}$" + f"={num_samples}"
					if num_samples
					else f"Transform {idx + 1}"
				)
				layers = list(range(base_layers))
				ax.plot(
					layers,
					l2_errors,
					marker=markers[
						len(sample_sizes) - len(TRANSFORM_FILES[data_name]) + idx
					],
					linewidth=3,
					markersize=9,
					markeredgecolor=colors[
						len(sample_sizes) - len(TRANSFORM_FILES[data_name]) + idx
					],
					color=color,
					label=label,
				)

			ax.set_xlabel("Layer")
			if dataset_idx == 0:
				ax.set_ylabel(r"Normalized $\ell_2$ Error")
			ax.set_title(f"{data_name.upper()}")
			leg = ax.legend(loc="upper left", title="# Samples", title_fontsize=20)
			leg.get_frame().set_alpha(0.2)
			ax.grid(True, alpha=0.3)
			ax.set_axisbelow(True)

			plt.subplots_adjust(left=0.1, right=0.98, top=0.9, bottom=0.19)

		torch.cuda.empty_cache()

		Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(OUTPUT_FILE, dpi=200)
		print(f"\nPlot saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
	main()
