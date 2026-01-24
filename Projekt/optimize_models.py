import torch
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, recall_score, auc, precision_recall_curve
import optuna
import numpy as np
from Conv1DAE import Conv1DAE, conv1dae_train, detect_anomalies_conv1dae
import pandas as pd
from numpy.typing import NDArray


def optimize_conv1dae(
        X: NDArray,
        y: NDArray,
        latent: int,
        dataset_name: str,
        show_optuna_output: bool = False,
        n_trials: int = 150,
        percentile: int = 95
) -> optuna.Study:

    def define_conv1dae(trial: optuna.Trial) -> tuple[dict, dict]:
        params = {
            "alpha": trial.suggest_float("alpha", 1.5, 2.0),
            "beta": trial.suggest_float("beta", 0.1, 0.4),
            "gamma": trial.suggest_float("gamma", 1.0, 1.5),
            "lr": trial.suggest_float("lr", 1e-3, 1e-1, log=True),
            "epochs": trial.suggest_int("epochs", 25, 50)
        }

        train_params = {k: params[k] for k in ["lr", "epochs", "alpha", "beta", "gamma"]}
        loss_params = {k: params[k] for k in ["alpha", "beta", "gamma"]}

        return train_params, loss_params

    def objective(trial: optuna.Trial):
        kf = KFold(n_splits=2, shuffle=False)
        reconstruction_errors = []

        for i, (train_index, val_index) in enumerate(kf.split(X=X, y=y)):
            x_train, x_test = X[train_index], X[val_index]

            x_train = x_train[:, :, None, :]
            x_test = x_test[:, :, None, :]

            x_train, x_test = torch.from_numpy(x_train).to(
                dtype=torch.float32,
                device="cuda"
            ), torch.from_numpy(x_test).to(
                dtype=torch.float32,
                device="cuda"
            )

            train_params, loss_params = define_conv1dae(trial)
            conv1dae = Conv1DAE(input_dim=1, latent_dim=latent)
            conv1dae_train(model=conv1dae, data=x_train, **train_params)

            errors, _, _ = detect_anomalies_conv1dae(conv1dae, x_test, **loss_params)
            reconstruction_errors.extend(errors)

        score = np.percentile(reconstruction_errors, percentile)
        return score

    study = optuna.create_study(
        study_name=f"Optuna for Conv1DAE on {dataset_name} dataset",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    if not show_optuna_output:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    study.optimize(objective, n_trials=n_trials)

    return study


def train_test_split_anomaly_sequence(
        X, y,
        test_size: float = 0.3,
        random_state: int = 42,
        epsilon: float = 0.1,
        pseudo_label: NDArray = None
):
    np.random.seed(random_state)
    num_seq = X.shape[0]
    seq_indices = np.arange(num_seq)
    np.random.shuffle(seq_indices)

    # Sprawdzenie czy nie jest zbyt mały test_size dla liczebności anomalii
    # Dodatkowo jeśli potrzeba to odseparowanie tych sekwencji,
    # które mają najmniejsze zanieczyszczenie
    key = "original" if pseudo_label is None else "pseudo"
    choose_dim = {
        "original": {
            2: (y == 1),
            3: (y == 1).all(axis=-1)
        },
        "pseudo": {
            2: (pseudo_label == 1),
            3: (pseudo_label == 1)
        }
    }

    count_seq_types = pd.Series(np.where(choose_dim.get(key).get(X.ndim), "normal", "anomaly")).value_counts(normalize=True)

    if count_seq_types.get("anomaly") >= test_size:
        test_size = count_seq_types.get("anomaly") + epsilon

    print(f"Zbyt mały test_size. Nowa wartość = {test_size}")

    test_size = int(num_seq * test_size)
    test_idx = seq_indices[:test_size]
    train_idx = seq_indices[test_size:]

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

    if X.ndim == 2 and pseudo_label is None:
        normal_mask_train = (y_train == 1)
        X_test = np.concatenate([X_test, X_train[~normal_mask_train]], axis=0)
        y_test = np.concatenate([y_test, y_train[~normal_mask_train]], axis=0)
        X_train = X_train[normal_mask_train]
        y_train = y_train[normal_mask_train]

    if X.ndim == 2 and pseudo_label is not None:
        normal_mask_train = (pseudo_label == 1)
        X_test = np.concatenate([X_test, X_train[~normal_mask_train]], axis=0)
        y_test = np.concatenate([y_test, y_train[~normal_mask_train]], axis=0)
        X_train = X_train[normal_mask_train]
        y_train = y_train[normal_mask_train]

    if X.ndim == 3:
        normal_mask_train = (y_train == 1).mean(axis=-1) >= 0.7
        X_test = np.concatenate([X_test, X_train[~normal_mask_train]], axis=0)
        y_test = np.concatenate([y_test, y_train[~normal_mask_train]], axis=0)
        X_train = X_train[normal_mask_train]
        y_train = y_train[normal_mask_train]

    return X_train, X_test, y_train, y_test


def find_anomaly_ranges(arr: NDArray):
    anomaly_indices = np.where(arr == -1)[0]
    if len(anomaly_indices) == 0:
        return []
    breaks = np.where(np.diff(anomaly_indices) > 1)[0] + 1
    regions = np.split(anomaly_indices, breaks)
    return [[region[0], region[-1]] for region in regions]


def ranges_overlap(start1, end1, start2, end2):
    return not (end1 < start2 or end2 < start1)


def latency_to_detection(y_true: NDArray, y_pred: NDArray) -> dict:
    all_latencies = []

    for labels_pred, labels_true in zip(y_pred, y_true):
        real_anomalies = find_anomaly_ranges(labels_true)
        predicted_anomalies = find_anomaly_ranges(labels_pred)

        for true_start, true_end in real_anomalies:
            first_detection = None

            for pred_start, pred_end in predicted_anomalies:
                if ranges_overlap(true_start, true_end, pred_start, pred_end):
                    if first_detection is None or pred_start < first_detection:
                        first_detection = pred_start

            if first_detection is not None:
                latency = max(0, first_detection - true_start)
                all_latencies.append(latency)
            else:
                all_latencies.append(np.nan)

    all_latencies = np.array(all_latencies)
    detected = all_latencies[~np.isnan(all_latencies)]

    return {
        "detection_rate": len(detected) / len(all_latencies),
        "detected_anomalies": len(detected),
        "missed_anomalies": len(all_latencies) - len(detected),
        "total_anomalies": len(all_latencies),
        "mean_latency": np.mean(detected),
        "median_latency": np.median(detected),
    }


def get_basic_metrics(
        y_true: NDArray,
        y_pred: NDArray,
        y_scores: NDArray
) -> dict:
    labels = [-1, 1]
    pos_label = -1

    precision, recall, _ = precision_recall_curve(
        y_true=y_true,
        y_score=y_scores,  # TRZEBA JUŻ Z MINUSEM PRZEKAZAĆ JEŚLI ISOLATION FOREST LUB LOF
        pos_label=pos_label
    )
    pr_auc = auc(recall, precision)

    return {
        "pr_auc": pr_auc,
        "f1": f1_score(y_true, y_pred, labels=labels, pos_label=pos_label),
        "recall": recall_score(y_true, y_pred, labels=labels, pos_label=pos_label)
    }


def convert_3d_to_2d(data: NDArray) -> NDArray:
    return data.reshape(-1, data.shape[-1])


def convert_2d_to_1d(data: NDArray) -> NDArray:
    return data.reshape(-1)
