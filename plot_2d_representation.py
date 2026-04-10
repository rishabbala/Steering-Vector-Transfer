import numpy as np
import plotly.graph_objs as go


def _sample_points_2d(n: int, seed: int, lim: float) -> np.ndarray:
	rng = np.random.default_rng(seed)
	return rng.uniform(-lim, lim, size=(n, 2))


def _add_wireframe_2d(traces, *, lim: float, step: float, color: str, width: int):
	vals = np.arange(-lim, lim + 1e-9, step)
	for v in vals:
		# vertical
		traces.append(
			go.Scatter(
				x=[v, v],
				y=[-lim, lim],
				mode="lines",
				line=dict(color=color, width=width),
				hoverinfo="skip",
			)
		)
		# horizontal
		traces.append(
			go.Scatter(
				x=[-lim, lim],
				y=[v, v],
				mode="lines",
				line=dict(color=color, width=width),
				hoverinfo="skip",
			)
		)


def plot_2d(
	*,
	seed: int,
	output_path: str,
	n_points: int = 6,
	lim: float = 1.65,
	point_lim: float = 1.15,
 	point_size: int = 28,
	fill_rgba: str,
	wire_rgba: str,
	wire_step: float = 0.25,
	wire_width: int = 3,
):
	# Keep consistent with the 3D palette file.
	colors = [
		"#006D2C",
		"#7A4E00",
		"#00695C",
		"#542788",
		"#BF0A92",
		"#4D4D4D",
	][:n_points]

	pts = _sample_points_2d(n_points, seed=seed, lim=point_lim)
	xs, ys = pts[:, 0], pts[:, 1]

	traces = []
	_add_wireframe_2d(traces, lim=lim, step=wire_step, color=wire_rgba, width=wire_width)

	# Points
	traces.append(
		go.Scatter(
			x=xs,
			y=ys,
			mode="markers",
			marker=dict(size=point_size, color=colors, line=dict(width=0)),
			hoverinfo="skip",
		)
	)

	fig = go.Figure(data=traces)
	fig.update_layout(
		showlegend=False,
		margin=dict(l=0, r=0, b=0, t=0),
		paper_bgcolor="rgba(0,0,0,0)",
		plot_bgcolor="rgba(0,0,0,0)",
		xaxis=dict(
			visible=False,
			range=[-lim, lim],
			scaleanchor="y",
			scaleratio=1,
			fixedrange=True,
		),
		yaxis=dict(visible=False, range=[-lim, lim], fixedrange=True),
		shapes=[
			dict(
				type="rect",
				xref="x",
				yref="y",
				x0=-lim,
				x1=lim,
				y0=-lim,
				y1=lim,
				fillcolor=fill_rgba,
				line=dict(width=0),
				layer="below",
			)
		],
	)

	# High-res export
	fig.write_image(output_path, width=1600, height=1600, scale=3)
	return fig


plot_2d(
	seed=2025,
	output_path="representation_2d_red.pdf",
	point_size=120,
	fill_rgba="rgba(235,120,120,0.55)",
	wire_rgba="rgba(255,32,32,0.75)",
)

plot_2d(
	seed=2026,
	output_path="representation_2d_blue.pdf",
	point_size=120,
	fill_rgba="rgba(120,140,245,0.55)",
	wire_rgba="rgba(32,32,255,0.75)",
)


