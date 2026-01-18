import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os


def heatmaps(scores: dict, title: str, save_path: str, normalize: bool = True):  # dla LSTM trzeba inaczej
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(20, 16))
    plt.suptitle(title, fontsize=20)
    for i, (second, overlaps) in enumerate(scores.items()):
        for j, (overlap, scores) in enumerate(overlaps.items()):
            heatmap_kwargs = {
                True: {
                    "data": MinMaxScaler().fit_transform(scores),
                    "vmin": 0,
                    "vmax": 1,
                    "cmap": "Greys",
                    "ax": ax[i][j]
                },
                False: {
                    "data": scores,
                    "cmap": "Greys",
                    "ax": ax[i][j]
                }
            }
            sns.heatmap(**heatmap_kwargs.get(normalize))
            if i == 2:
                ax[i][j].set_xlabel("Numer segmentu", fontsize=16)
            if j == 0:
                ax[i][j].set_ylabel("Numer sekwencji", fontsize=16)
            ax[i][j].set_title(f"Wielkość okna = {second}, Overlap = {overlap}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if not "figures" in os.listdir(os.getcwd()) and save_path:
        os.mkdir("figures")
    plt.savefig(os.path.join("figures", save_path), dpi=300, bbox_inches="tight")
    plt.close()

def plot_fragment(
        true_segments: np.typing.NDArray,
        pred_segments: np.typing.NDArray,
        ax,
        title: str,
        place: tuple[int, int] = (0, 0),
        part_of_confusion_matrix: str = "tp"
):
    colors = {
        "tp": ["#1f77b4", "#ff7f0e", "#ffbb78"],
        "tn": ["#1f77b4", "#2ca02c", "#98df8a"],
        "fp": ["#1f77b4", "#d62728", "#ff9896"],
        "fn": ["#1f77b4", "#9467bd", "#c5b0d5"],
    }
    for i, (true, pred) in enumerate(zip(true_segments, pred_segments)):
        timesteps = np.arange(0, len(true))
        color = colors.get(part_of_confusion_matrix)
        ax[i + place[0]][place[1]].plot(
            timesteps, true,
            label="Prawdziwy sygnał", linewidth=2,
            color=color[0], alpha=0.8
        )
        ax[i + place[0]][place[1]].plot(
            timesteps, pred,
            label="Odtworzony sygnał", linewidth=2,
            linestyle="--", color=color[1], alpha=0.8
        )
        ax[i + place[0]][place[1]].fill_between(
            timesteps, true, pred,
            alpha=0.3, color=color[2]
        )

        ax[i + place[0]][place[1]].set_title(title, fontweight="bold")
        ax[i + place[0]][place[1]].set_ylabel("Amplituda", fontsize=11)
        ax[i + place[0]][place[1]].set_xlabel("Timestep", fontsize=11)
        ax[i + place[0]][place[1]].legend(loc="upper right", fontsize=10)
        ax[i + place[0]][place[1]].grid(True, alpha=0.3)

def segments_reconstruction(
        X_test: np.typing.NDArray,
        X_pred: np.typing.NDArray,
        y_true: np.typing.NDArray,
        y_pred: np.typing.NDArray,
        title: str,
        save_path: str,
        n: int = 3,
):
    np.random.seed(42)

    conf_matrix = {
        "tp": np.argwhere((y_true == -1) & (y_pred == -1)),
        "fp": np.argwhere((y_true == 1) & (y_pred == -1)),
        "fn": np.argwhere((y_true == -1) & (y_pred == 1)),
        "tn": np.argwhere((y_true == 1) & (y_pred == 1))
    }

    choosen = {k: v[np.random.choice(len(v), n)] for k, v in conf_matrix.items() if len(v) != 0}

    segments = {
        k: [X_test[v[:, 0], v[:, 1]], X_pred[v[:, 0], v[:, 1]]]
        for k, v in choosen.items()
    }

    plot_info = {
        "tp": ["True Positive — poprawnie wykryta anomalia", (0, 0)],
        "fp": ["False Positive — fałszywy alarm", (0, 1)],
        "fn": ["False Negative — pominięta anomalia", (n, 0)],
        "tn": ["True Negative — poprawnie rozpoznany sygnał normalny", (n, 1)],
    }

    fig, ax = plt.subplots(nrows=2 * n, ncols=2, figsize=(18, 3.5 * 2 * n))
    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.995)

    for k, v in segments.items():
        title, place = plot_info.get(k)
        plot_fragment(
            v[0], v[1],
            ax, place=place,
            title=title,
            part_of_confusion_matrix=k
        )

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    if not "figures" in os.listdir(os.getcwd()) and save_path:
        os.mkdir("figures")
    plt.savefig(os.path.join("figures", save_path), dpi=300, bbox_inches="tight")
    plt.close()
