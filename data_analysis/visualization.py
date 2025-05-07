from enum import StrEnum
from typing import Union
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("ggplot")


class AxisNames(StrEnum):
    X = "X"
    Y = "Y"


class DiagramTypes(StrEnum):
    Violin = "violin"
    Hist = "hist"
    Boxplot = "box"


def visualize_distribution(
    points: np.ndarray,
    diagram_type: Union[DiagramTypes, list[DiagramTypes]],
    diagram_axis: Union[AxisNames, list[AxisNames]],
    path_to_save: str = "",
) -> None:
    """Визуализация распределения данных"""
    if not isinstance(diagram_type, list):
        diagram_type = [diagram_type]
    if not isinstance(diagram_axis, list):
        diagram_axis = [diagram_axis]

    fig, axes = plt.subplots(
        len(diagram_type),
        len(diagram_axis),
        figsize=(8, 6),
        squeeze=False,
    )

    for i, d_type in enumerate(diagram_type):
        for j, axis in enumerate(diagram_axis):
            ax = axes[i, j]
            data = points[:, 0] if axis == AxisNames.X else points[:, 1]

            if d_type == DiagramTypes.Violin:
                violin_parts = ax.violinplot(
                    data,
                    showmeans=True,
                    showmedians=True,
                    vert=False,
                )
                for body in violin_parts["bodies"]:
                    body.set_facecolor("cornflowerblue")
                    body.set_edgecolor("blue")

            elif d_type == DiagramTypes.Hist:
                ax.hist(
                    data,
                    bins=20,
                    alpha=0.7,
                    color="cornflowerblue",
                )

            elif d_type == DiagramTypes.Boxplot:
                ax.boxplot(
                    data,
                    vert=False,
                    patch_artist=True,
                    boxprops=dict(facecolor="lightsteelblue"),
                    medianprops=dict(color="k")
                )

            ax.set_title(f"{d_type} plot for {axis} axis")
            ax.grid(True)


    plt.tight_layout()
    if path_to_save:
        plt.savefig(path_to_save)
    plt.show()
