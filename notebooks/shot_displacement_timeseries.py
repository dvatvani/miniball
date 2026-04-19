import marimo

__generated_with = "0.22.0"
app = marimo.App(width="medium", sql_output="pandas")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from miniball.config import STRIKE_SPEED, BALL_DECEL, STANDARD_PITCH_WIDTH

    return BALL_DECEL, STANDARD_PITCH_WIDTH, STRIKE_SPEED, mo, np, plt


@app.cell
def _(BALL_DECEL, STRIKE_SPEED, mo):
    strike_speed_slider = mo.ui.slider(
        start=10,
        stop=120,
        step=1,
        value=STRIKE_SPEED,
        label="Strike speed (units/s)",
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
    mo.vstack([strike_speed_slider, ball_decel_slider])
    return ball_decel_slider, strike_speed_slider


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
):
    v0 = strike_speed_slider.value
    a = ball_decel_slider.value

    # Ball decelerates linearly: speed(t) = max(0, v0 - a*t)
    t_stop = v0 / a
    t = np.linspace(0, t_stop, 500)

    # Displacement: integral of (v0 - a*t) dt = v0*t - a*t²/2
    displacement = v0 * t - 0.5 * a * t**2
    total_distance = v0**2 / (2 * a)  # exact stop distance

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(t, displacement, linewidth=2, color="#4a90d9")

    # Reference lines
    ax.axhline(
        STANDARD_PITCH_WIDTH,
        color="gray",
        linestyle="--",
        linewidth=1,
        label=f"Pitch width ({STANDARD_PITCH_WIDTH})",
    )
    ax.axhline(
        total_distance,
        color="#e07040",
        linestyle=":",
        linewidth=1.5,
        label=f"Total distance = {total_distance:.1f}",
    )

    # Rematch reference observations
    ax.scatter(
        rematch_reference[:, 1], rematch_reference[:, 0], marker="x", label="reference"
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Displacement (game units)")
    ax.set_title(f"Ball displacement  |  v₀ = {v0} u/s,  decel = {a:.1f} u/s²")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    mo.mpl.interactive(fig)
    return


if __name__ == "__main__":
    app.run()
