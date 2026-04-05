import marimo

__generated_with = "0.22.0"
app = marimo.App(width="medium", sql_output="pandas")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from miniball.config import STRIKE_SPEED, BALL_DRAG, STANDARD_PITCH_WIDTH

    return BALL_DRAG, STANDARD_PITCH_WIDTH, STRIKE_SPEED, mo, np, plt


@app.cell
def _(BALL_DRAG, STRIKE_SPEED, mo):
    strike_speed_slider = mo.ui.slider(
        start=10,
        stop=120,
        step=1,
        value=STRIKE_SPEED,
        label="Strike speed (units/s)",
        show_value=True,
    )
    ball_drag_slider = mo.ui.slider(
        start=0.1,
        stop=2.0,
        step=0.01,
        value=round(BALL_DRAG, 2),
        label="Ball drag (1/s)",
        show_value=True,
    )
    mo.vstack([strike_speed_slider, ball_drag_slider])
    return ball_drag_slider, strike_speed_slider


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
    ball_drag_slider,
    mo,
    np,
    plt,
    rematch_reference,
    strike_speed_slider,
):
    v0 = strike_speed_slider.value
    k = ball_drag_slider.value

    # Time axis: run until ball has lost 99 % of its initial speed
    t_stop = -np.log(0.01) / k
    t = np.linspace(0, t_stop, 500)

    # Displacement: integral of v0*exp(-k*t) dt = v0/k * (1 - exp(-k*t))
    displacement = (v0 / k) * (1 - np.exp(-k * t))
    total_distance = v0 / k  # asymptotic limit

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
        label=f"Total distance ≈ {total_distance:.1f}",
    )

    # Rematch reference observations
    ax.scatter(
        rematch_reference[:, 1], rematch_reference[:, 0], marker="x", label="reference"
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Displacement (game units)")
    ax.set_title(f"Ball displacement  |  v₀ = {v0} u/s,  drag = {k:.2f} /s")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    mo.mpl.interactive(fig)
    return


if __name__ == "__main__":
    app.run()
