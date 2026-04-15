import numpy as np
import plotly.graph_objs as go


def _hide_3d_axes(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        showlegend=False,
        margin=dict(l=0, r=0, b=0, t=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            xaxis=dict(visible=False, showbackground=False, showgrid=False, zeroline=False),
            yaxis=dict(visible=False, showbackground=False, showgrid=False, zeroline=False),
            zaxis=dict(visible=False, showbackground=False, showgrid=False, zeroline=False),
            aspectmode="data",
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    return fig


def _sample_points_2d(n: int, seed: int = 2025, lim: float = 0.95):
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-lim, lim, size=(n, 2))
    return pts


def _add_arrow_2d(traces, x, y, z, color: str, *, line_width: int = 3):
    """
    Add a 2D arrow (lying in the plane z=0) from origin -> (x,y,z):
    a shaft + a simple V-shaped arrowhead (no 3D cone).
    """
    # Shaft
    traces.append(
        go.Scatter3d(
            x=[0, x],
            y=[0, y],
            z=[0, z],
            mode="lines",
            line=dict(color=color, width=line_width),
            opacity=1,
        )
    )

    # 2D arrowhead in the plane (two short segments)
    v2 = np.array([x, y], dtype=float)
    n2 = float(np.linalg.norm(v2))
    if n2 == 0:
        return
    u = v2 / n2
    p = np.array([-u[1], u[0]])

    head_len = 0.12 * n2
    head_w = 0.06 * n2
    tip = np.array([x, y], dtype=float)
    left = tip - head_len * u + head_w * p
    right = tip - head_len * u - head_w * p

    traces.append(
        go.Scatter3d(
            x=[tip[0], left[0], None, tip[0], right[0]],
            y=[tip[1], left[1], None, tip[1], right[1]],
            z=[0, 0, None, 0, 0],
            mode="lines",
            line=dict(color=color, width=line_width),
            opacity=1,
        )
    )


def _add_plane_wireframe(
    traces,
    *,
    plane_lim: float,
    step: float = 0.25,
    color: str = "rgba(0,0,0,0.35)",
    width: int = 3,
):
    """
    Draw a wireframe on the z=0 plane using line traces (works even with axes hidden).
    """
    vals = np.arange(-plane_lim, plane_lim + 1e-9, step)
    for v in vals:
        # vertical line: x=v, y in [-lim, lim]
        traces.append(
            go.Scatter3d(
                x=[v, v],
                y=[-plane_lim, plane_lim],
                z=[0, 0],
                mode="lines",
                line=dict(color=color, width=width),
                hoverinfo="skip",
                opacity=1,
            )
        )
        # horizontal line: y=v, x in [-lim, lim]
        traces.append(
            go.Scatter3d(
                x=[-plane_lim, plane_lim],
                y=[v, v],
                z=[0, 0],
                mode="lines",
                line=dict(color=color, width=width),
                hoverinfo="skip",
                opacity=1,
            )
        )

def plot_2d_surface_with_points_and_arrows(
    *,
    n_points: int = 6,
    seed: int = 2025,
    point_size: int = 4,
    arrow_line_width: int = 3,
    plane_lim: float = 1.65,
    plane_fill_rgba: str = "rgba(200,200,200,1)",
    plane_opacity: float = 0.25,
    wireframe_rgba: str = "rgba(0,0,0,0.35)",
    wireframe_step: float = 0.25,
    wireframe_width: int = 3,
):
    # Same 6 colors used in the 3D version.
    circle_colors = [
        "#006D2C",
        "#7A4E00",
        "#00695C",
        "#542788",
        "#BF0A92",
        "#4D4D4D",
    ]
    circle_colors = circle_colors[:n_points]

    # Keep points well within the plane.
    # Scatter points more widely while staying well within the plane.
    pts2d = _sample_points_2d(n_points, seed=seed, lim=1.15)
    xs = pts2d[:, 0]
    ys = pts2d[:, 1]
    zs = np.zeros(n_points)

    # 2D plane (z=0) rendered in a 3D scene.
    # Higher resolution grid => crisper exports.
    grid = np.linspace(-plane_lim, plane_lim, 90)
    xx, yy = np.meshgrid(grid, grid)
    zz = np.zeros_like(xx)
    plane = go.Surface(
        x=xx,
        y=yy,
        z=zz,
        opacity=plane_opacity,
        showscale=False,
        colorscale=[[0, plane_fill_rgba], [1, plane_fill_rgba]],
        hoverinfo="skip",
    )

    traces = [plane]
    _add_plane_wireframe(
        traces,
        plane_lim=plane_lim,
        step=wireframe_step,
        color=wireframe_rgba,
        width=wireframe_width,
    )

    # Points as circles on the plane.
    traces.append(
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="markers",
            marker=dict(
                size=point_size,
                color=circle_colors,
                opacity=1,
                line=dict(width=0),
                symbol="circle",
            ),
            hoverinfo="skip",
        )
    )

    # Arrows from origin to each point
    for (x, y, z, c) in zip(xs, ys, zs, circle_colors):
        _add_arrow_2d(traces, float(x), float(y), float(z), c, line_width=arrow_line_width)

    fig = go.Figure(data=traces)
    fig = _hide_3d_axes(fig)

    # Camera to emphasize 3D while keeping the surface clearly 2D.
    fig.update_layout(scene_camera=dict(eye=dict(x=1.85, y=1.85, z=1.55)))
    return fig


fig_red = plot_2d_surface_with_points_and_arrows(
    seed=2025,
    point_size=8,
    arrow_line_width=16,
    plane_fill_rgba="rgba(235,120,120,1)",  # darker red fill
    plane_opacity=0.48,
    wireframe_rgba="rgba(255,32,32,0.75)",  # darker red wireframe (matches 3D)
    wireframe_step=0.25,
    wireframe_width=3,
    plane_lim=1.65,
)
fig_red.write_image("representation_2d_red.pdf", width=1200, height=1200, scale=3)

fig_blue = plot_2d_surface_with_points_and_arrows(
    seed=2026,
    point_size=8,
    arrow_line_width=16,
    plane_fill_rgba="rgba(120,140,245,1)",  # darker blue fill
    plane_opacity=0.46,
    wireframe_rgba="rgba(32,32,255,0.75)",  # darker blue wireframe (matches 3D)
    wireframe_step=0.25,
    wireframe_width=3,
    plane_lim=1.65,
)
fig_blue.write_image("representation_2d_blue.pdf", width=1200, height=1200, scale=3)

