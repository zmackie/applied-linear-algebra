import marimo

__generated_with = "0.9.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    return mo, np, plt


@app.cell
def _(mo):
    mo.md(
        r"""
        # Applied Linear Algebra

        Drag the sliders to change the matrix $A$ and see how it transforms the unit square.
        """
    )
    return


@app.cell
def _(mo):
    a = mo.ui.slider(-2, 2, step=0.1, value=1, label="a")
    b = mo.ui.slider(-2, 2, step=0.1, value=0.5, label="b")
    c = mo.ui.slider(-2, 2, step=0.1, value=0, label="c")
    d = mo.ui.slider(-2, 2, step=0.1, value=1, label="d")
    mo.hstack([a, b, c, d])
    return a, b, c, d


@app.cell
def _(a, b, c, d, mo, np, plt):
    A = np.array([[a.value, b.value], [c.value, d.value]])
    square = np.array([[0, 1, 1, 0, 0], [0, 0, 1, 1, 0]])
    image = A @ square

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(*square, label="unit square")
    ax.plot(*image, label="A · square")
    ax.set_aspect("equal")
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.legend()

    mo.vstack([mo.md(f"$\\det A = {np.linalg.det(A):.2f}$"), ax])
    return A, ax, fig, image, square


if __name__ == "__main__":
    app.run()
