import marimo

__generated_with = "0.22.0"
app = marimo.App(width="medium", sql_output="pandas")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from miniball.config import (
        BALL_DECEL,
        MAX_STRIKE_SPEED,
        STANDARD_PITCH_HEIGHT,
        STANDARD_PITCH_WIDTH,
        STRIKE_WEIGHT_CIRCLE,
        STRIKE_WEIGHT_CROSS,
        STRIKE_WEIGHT_SQUARE,
        STRIKE_WEIGHT_TRIANGLE,
    )

    return (
        BALL_DECEL,
        MAX_STRIKE_SPEED,
        STANDARD_PITCH_HEIGHT,
        STANDARD_PITCH_WIDTH,
        STRIKE_WEIGHT_CIRCLE,
        STRIKE_WEIGHT_CROSS,
        STRIKE_WEIGHT_SQUARE,
        STRIKE_WEIGHT_TRIANGLE,
        mo,
        np,
        plt,
    )


@app.cell
def _(
    BALL_DECEL,
    MAX_STRIKE_SPEED,
    STRIKE_WEIGHT_CIRCLE,
    STRIKE_WEIGHT_CROSS,
    STRIKE_WEIGHT_SQUARE,
    STRIKE_WEIGHT_TRIANGLE,
    mo,
):
    strike_speed_slider = mo.ui.slider(
        start=10,
        stop=120,
        step=1,
        value=MAX_STRIKE_SPEED,
        label="Max strike speed (units/s)",
        show_value=True,
    )
    ball_decel_slider = mo.ui.slider(
        start=1.0,
        stop=50.0,
        step=0.5,
        value=round(BALL_DECEL, 1),
        label="Ball deceleration (units/s²)",
        show_value=True,
    )
    weight_x_slider = mo.ui.slider(
        start=0.01,
        stop=1.0,
        step=0.01,
        value=STRIKE_WEIGHT_CROSS,
        label="✕ X weight",
        show_value=True,
    )
    weight_square_slider = mo.ui.slider(
        start=0.01,
        stop=1.0,
        step=0.01,
        value=STRIKE_WEIGHT_SQUARE,
        label="□ Square weight",
        show_value=True,
    )
    weight_triangle_slider = mo.ui.slider(
        start=0.01,
        stop=1.0,
        step=0.01,
        value=STRIKE_WEIGHT_TRIANGLE,
        label="△ Triangle weight",
        show_value=True,
    )
    weight_circle_slider = mo.ui.slider(
        start=0.01,
        stop=1.0,
        step=0.01,
        value=STRIKE_WEIGHT_CIRCLE,
        label="○ Circle weight",
        show_value=True,
    )
    mo.vstack(
        [
            mo.hstack([strike_speed_slider, ball_decel_slider]),
            mo.hstack(
                [
                    weight_square_slider,
                    weight_circle_slider,
                    weight_triangle_slider,
                    weight_x_slider,
                ]
            ),
        ]
    )
    return (
        ball_decel_slider,
        strike_speed_slider,
        weight_circle_slider,
        weight_square_slider,
        weight_triangle_slider,
        weight_x_slider,
    )


@app.cell
def _(np):
    # Reference ball speed measurements from Rematch
    # distance, time in seconds
    rematch_reference = np.array(
        [[20, 0.5], [40, 1.3], [60, 2.1], [80, 3.2], [90, 5.0]]
    )
    return (rematch_reference,)


@app.cell
def _(
    STANDARD_PITCH_WIDTH,
    ball_decel_slider,
    mo,
    np,
    plt,
    rematch_reference,
    strike_speed_slider,
    weight_circle_slider,
    weight_square_slider,
    weight_triangle_slider,
    weight_x_slider,
):
    v_max = strike_speed_slider.value
    a = ball_decel_slider.value

    # Button definitions: (label, weight_slider, color)
    buttons = [
        ("□ Square", weight_square_slider, "#dc3296"),
        ("○ Circle", weight_circle_slider, "#dc2828"),
        ("△ Triangle", weight_triangle_slider, "#32be5a"),
        ("✕ X", weight_x_slider, "#3264d2"),
    ]

    fig, ax = plt.subplots(figsize=(9, 5))

    for btn_label, w_slider, color in buttons:
        w = w_slider.value
        # weight is a linear range fraction → speed = sqrt(weight) * v_max
        v0 = v_max * w**0.5
        t_stop = v0 / a
        t = np.linspace(0, t_stop, 500)
        displacement = v0 * t - 0.5 * a * t**2
        total_distance = v0**2 / (2 * a)  # = w * v_max² / (2a)
        ax.plot(
            t,
            displacement,
            linewidth=2,
            color=color,
            label=f"{btn_label}  w={w:.2f}  range={total_distance:.1f}",
        )
        ax.axhline(total_distance, color=color, linestyle=":", linewidth=1, alpha=0.4)

    # Pitch width reference
    ax.axhline(
        STANDARD_PITCH_WIDTH,
        color="gray",
        linestyle="--",
        linewidth=1,
        label=f"Pitch width ({STANDARD_PITCH_WIDTH})",
    )

    # Rematch reference observations
    ax.scatter(
        rematch_reference[:, 1],
        rematch_reference[:, 0],
        marker="x",
        color="black",
        zorder=5,
        label="Rematch reference",
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Displacement (game units)")
    ax.set_title(
        f"Ball displacement by strike weight  |  v_max = {v_max} u/s,  decel = {a:.1f} u/s²"
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    mo.mpl.interactive(fig)
    return


@app.function
def STRIKE_ANGULAR_ERROR_DEGREES_FN(weight: float) -> float:
    """Max angular error in degrees for a strike at the given weight (linear: 0 → 4)."""
    return weight**3 * 6.0


@app.cell
def _(
    STANDARD_PITCH_HEIGHT,
    STANDARD_PITCH_WIDTH,
    ball_decel_slider,
    np,
    plt,
    strike_speed_slider,
    weight_circle_slider,
    weight_square_slider,
    weight_triangle_slider,
    weight_x_slider,
):
    from miniball.ai.utils.vis import plot_pitch

    GK_X, GK_Y = 5.0, STANDARD_PITCH_HEIGHT / 2
    facing = 0.0  # shooting rightward (+x)

    v_max2 = strike_speed_slider.value
    a2 = ball_decel_slider.value

    buttons2 = [
        ("□ Square", weight_square_slider, "#dc3296"),
        ("○ Circle", weight_circle_slider, "#dc2828"),
        ("△ Triangle", weight_triangle_slider, "#32be5a"),
        ("✕ X", weight_x_slider, "#3264d2"),
    ]

    def _ray_t(gx, gy, angle, max_range):
        """Distance to pitch boundary or max_range along angle from (gx, gy)."""
        dx, dy = np.cos(angle), np.sin(angle)
        candidates = []
        if abs(dx) > 1e-9:
            candidates.append(((STANDARD_PITCH_WIDTH if dx > 0 else 0) - gx) / dx)
        if abs(dy) > 1e-9:
            candidates.append(((STANDARD_PITCH_HEIGHT if dy > 0 else 0) - gy) / dy)
        valid = [t for t in candidates if t > 1e-9]
        return min(min(valid) if valid else max_range, max_range)

    ax2 = plot_pitch()

    # Draw cones heaviest → lightest so lighter ones render on top
    n_steps = 64
    for btn_label2, w_slider2, color2 in buttons2:
        w2 = w_slider2.value
        error_rad = np.radians(STRIKE_ANGULAR_ERROR_DEGREES_FN(w2))
        max_range2 = w2 * v_max2**2 / (2 * a2)
        angle_lo = facing - error_rad
        angle_hi = facing + error_rad

        pts = [(GK_X, GK_Y)]
        for i in range(n_steps + 1):
            angle = angle_lo + (angle_hi - angle_lo) * i / n_steps
            t2 = _ray_t(GK_X, GK_Y, angle, max_range2)
            pts.append((GK_X + np.cos(angle) * t2, GK_Y + np.sin(angle) * t2))

        xs, ys = zip(*pts)
        ax2.fill(xs, ys, color=color2, alpha=0.25)
        ax2.plot(xs[1:], ys[1:], color=color2, linewidth=1, alpha=0.6)

        # Centre aim arrow
        t_c = _ray_t(GK_X, GK_Y, facing, max_range2)
        ax2.annotate(
            "",
            xy=(GK_X + np.cos(facing) * t_c, GK_Y + np.sin(facing) * t_c),
            xytext=(GK_X, GK_Y),
            arrowprops=dict(arrowstyle="-|>", color=color2, lw=1.5),
        )

        # Invisible line for legend
        ax2.plot(
            [],
            [],
            color=color2,
            linewidth=3,
            label=(
                f"{btn_label2}  w={w2:.2f}"
                f"  range={max_range2:.1f}"
                f"  err=±{np.degrees(error_rad):.1f}°"
            ),
        )

    # GK marker
    ax2.scatter([GK_X], [GK_Y], color="white", edgecolors="black", s=100, zorder=10)
    ax2.annotate(
        "GK",
        (GK_X, GK_Y),
        textcoords="offset points",
        xytext=(6, 6),
        color="white",
        fontsize=9,
    )

    ax2.set_xlim(-5, STANDARD_PITCH_WIDTH + 5)
    ax2.set_ylim(-5, STANDARD_PITCH_HEIGHT + 5)
    ax2.set_aspect("equal")
    ax2.legend(fontsize=9, loc="upper right", framealpha=0.8)
    ax2.set_title(
        f"Strike uncertainty cones from GK"
        f"  |  v_max = {v_max2} u/s,  decel = {a2:.1f} u/s²"
    )
    ax2.set_xlabel("x (game units)")
    ax2.set_ylabel("y (game units)")
    plt.tight_layout()

    ax2
    return


if __name__ == "__main__":
    app.run()
