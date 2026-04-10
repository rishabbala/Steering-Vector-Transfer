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


def _append_polyline(xs, ys, zs, x, y, z):
    xs.extend(x)
    ys.extend(y)
    zs.extend(z)
    xs.append(None)
    ys.append(None)
    zs.append(None)


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n == 0:
        return v
    return v / n


def _tangent_ring(
    xyz_fn,
    phi0: float,
    theta0: float,
    *,
    ring_radius: float,
    n: int = 90,
    eps: float = 1e-3,
):
    """
    Small ring centered at a surface point, lying on the local tangent plane.
    """
    p = xyz_fn(phi0, theta0)
    # Finite-difference tangents in parameter space.
    t_phi = xyz_fn(phi0 + eps, theta0) - p
    t_theta = xyz_fn(phi0, theta0 + eps) - p

    u = _normalize(t_theta)
    if float(np.linalg.norm(u)) == 0:
        u = _normalize(t_phi)
    v = t_phi - float(np.dot(t_phi, u)) * u
    v = _normalize(v)
    if float(np.linalg.norm(v)) == 0:
        # Fallback: pick any perpendicular direction to u
        a = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(a, u))) > 0.9:
            a = np.array([0.0, 1.0, 0.0])
        v = _normalize(np.cross(u, a))

    ang = np.linspace(0.0, 2.0 * np.pi, n, endpoint=True)
    ring = p + ring_radius * (
        np.cos(ang)[:, None] * u[None, :] + np.sin(ang)[:, None] * v[None, :]
    )
    return ring[:, 0], ring[:, 1], ring[:, 2]


def _sample_circle_locations(n: int, seed: int = 123):
    rng = np.random.default_rng(seed)
    # Avoid poles for numerical stability.
    phis = rng.uniform(0.25, 2.0 * np.pi, size=n)
    thetas = rng.uniform(0.0, 2.0 * np.pi, size=n)
    return list(zip(phis.tolist(), thetas.tolist()))


def plot_wireframe(
    kind: int,
    *,
    color: str = "#0066FF",
    linewidth: int = 1,
    n_lat: int = 50,
    n_lon: int = 50,
    n_pts: int = 1000,
    seed: int = 0,
    circle_locations=None,
    circle_colors=None,
    point_size: int = 5,
    scale: float = 1.0,
):
    """
    Draw a dense, sphere-like "latent space" wireframe.

    `kind` controls the deformation frequency:
      - kind=6  -> lower-frequency structure
      - kind=10 -> higher-frequency structure
    """
    rng = np.random.default_rng(seed + kind)
    # Smooth "noise" via a small random Fourier-like basis (keeps curves smooth).
    p1, p2, p3 = rng.uniform(0, 2 * np.pi, size=3)
    k1, k2 = rng.integers(2, 5, size=2)
    k3, k4 = rng.integers(2, 5, size=2)

    # Dense latitudes/longitudes
    phis = np.linspace(0.0, np.pi, n_lat, endpoint=True)  # polar angle
    thetas = np.linspace(0.0, 2.0 * np.pi, n_lon, endpoint=False)  # azimuth

    def radius(phi, theta):
        # Base sphere + smooth spherical-ish deformation + tiny roughness.
        # Chosen to look "latent manifold"-like rather than a geometric solid.
        a = 0.16 if kind == 7 else 0.22
        b = 0.08 if kind == 7 else 0.12
        # Use frequencies tied to kind to visually differentiate the two spaces.
        r = 1.0
        r += a * np.sin((kind // 2) * theta) * np.cos((kind // 2) * phi)
        r += b * np.cos((kind) * theta + 0.6 * np.sin(phi)) * np.sin((kind // 2) * phi)
        eps = 0.03 if kind == 4 else 0.04
        r += eps * (
            0.55 * np.sin(k1 * theta + k2 * phi + p1)
            + 0.30 * np.sin(k3 * theta - k4 * phi + p2)
            + 0.15 * np.sin((k1 + 1) * theta + (k2 + 1) * phi + p3)
        )
        return r

    def xyz(phi, theta):
        r = radius(phi, theta)
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        return scale * np.array([x, y, z], dtype=float)

    xs, ys, zs = [], [], []

    # Latitude rings (theta varies)
    for phi0 in phis[1:-1]:
        theta = np.linspace(0.0, 2.0 * np.pi, n_pts, endpoint=True)
        r = radius(phi0, theta)
        x = scale * (r * np.sin(phi0) * np.cos(theta))
        y = scale * (r * np.sin(phi0) * np.sin(theta))
        z = scale * (r * np.cos(phi0) * np.ones_like(theta))
        _append_polyline(xs, ys, zs, x.tolist(), y.tolist(), z.tolist())

    # Longitude meridians (phi varies)
    for theta0 in thetas:
        phi = np.linspace(0.0, np.pi, n_pts, endpoint=True)
        r = radius(phi, theta0)
        x = scale * (r * np.sin(phi) * np.cos(theta0))
        y = scale * (r * np.sin(phi) * np.sin(theta0))
        z = scale * (r * np.cos(phi))
        _append_polyline(xs, ys, zs, x.tolist(), y.tolist(), z.tolist())

    traces = [
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines",
            line=dict(color=color, width=linewidth),
            opacity=1,
        )
    ]

    # Overlay points: same 6 colors across both plots, placed at shared surface points.
    if circle_locations is not None and circle_colors is not None:
        pts = [xyz(float(phi0), float(theta0)) for (phi0, theta0) in circle_locations]
        pts = np.stack(pts, axis=0)
        traces.append(
            go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode="markers",
                marker=dict(
                    size=point_size,
                    color=list(circle_colors),
                    opacity=1,
                    line=dict(width=0),
                ),
            )
        )

    fig = go.Figure(data=traces)
    fig = _hide_3d_axes(fig)
    # Slightly closer camera so the shape fills more of the canvas.
    fig.update_layout(scene_camera=dict(eye=dict(x=1.25, y=1.25, z=0.85)))
    return fig


# 6 distinct colors for circles (shared across the two plots), chosen to avoid the base wire colors.
CIRCLE_COLORS = [
	"#2FC490",
	"#FF7A00",
	"#00B800",
	"#542788",
	"#BF0A92",
	"#4D4D4D",
]


# Shared circle placements across both plots.
CIRCLE_LOCS = _sample_circle_locations(6, seed=2025)

fig_6 = plot_wireframe(
    4,
    color="#2020FF",
    seed=3,
    circle_locations=CIRCLE_LOCS,
    circle_colors=CIRCLE_COLORS,
)
fig_6.write_image("representation_4_sided.png")

fig_10 = plot_wireframe(
    7,
    color="#FF2020",
    circle_locations=CIRCLE_LOCS,
    circle_colors=CIRCLE_COLORS,
)
fig_10.write_image("representation_7_sided.png")

# Optional interactive HTML:
# If you want HTML output, import `plotly.io as pio` and uncomment:
# pio.write_html(fig_6, "representation_6_sided.html", include_plotlyjs="cdn", full_html=False)
# pio.write_html(fig_10, "representation_10_sided.html", include_plotlyjs="cdn", full_html=False)