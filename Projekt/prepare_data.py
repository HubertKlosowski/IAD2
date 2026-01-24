import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import os
import wfdb
import json
from joblib import Parallel, delayed
from collections import Counter
from numpy.typing import NDArray


def sliding_window(data: NDArray, window_size: int, overlap: float) -> NDArray[float]:
    step = int(window_size * (1 - overlap))
    without_nans = [data[i * step:i * step + window_size].tolist() for i in range(0, 2 + (len(data) - window_size) // step)]
    # żeby móc stworzyć np.array muszą być wymiary zgodne, dlatego np.nan
    # i również dlatego, że nie wpłynie to na tworzenie cech
    without_nans[-1] += [np.nan for _ in range(window_size - len(without_nans[-1]))]
    without_nans = np.array(without_nans)[:-1]
    return without_nans


# Ujednolicenie długości sekwencji
def trim_to_shortest(arr: list) -> NDArray:
    min_length = min([len(_) for _ in arr])
    return np.array([el[:min_length] for el in arr])


# ZBIÓR EEG BONN
def load_or_cache_bonn(cache_path = "bonn_cache.pkl") -> dict:
    if not Path("cache").exists():
        os.mkdir("cache")

    cache_file = Path(os.path.join("cache", cache_path))

    if cache_file.exists():
        load_pickle = open(cache_file, "rb")
        pickle.load(load_pickle)
        load_pickle.close()

    eeg_bonn_subpaths = [os.path.join(os.getcwd(), f"data{l}") for l in ["A", "B", "C", "D", "E"]]
    eeg_bonn_paths = np.array([
        sorted([
            os.path.join(subpath, p) for p in os.listdir(subpath)
        ]) for subpath in eeg_bonn_subpaths
    ]).flatten()

    eeg_bonn_dataset = {
        os.path.basename(p): pd.read_csv(p, header=None).rename(columns={0: "value"})
        for p in eeg_bonn_paths
    }

    save_pickle = open(cache_file, "wb")
    pickle.dump(eeg_bonn_dataset, save_pickle)
    save_pickle.close()
    return eeg_bonn_dataset


def process_bonn_sequence(
        eeg_bonn_sequence,
        path,
        sec: int = 2,
        bonn_overlap: float = 0.5
):
    bonn_label = {True: -1, False: 1}.get(
        os.path.basename(path).startswith("S")
    )
    frequency = 174  # oryginalnie było 173.61Hz
    bonn_segments = sliding_window(eeg_bonn_sequence["value"], sec * frequency, bonn_overlap)

    bonn_segments_label = np.ones(shape=(bonn_segments.shape[0],), dtype=int) * bonn_label

    return bonn_segments, bonn_segments_label

def get_bonn_segments(
        eeg_bonn_dataset: pd.DataFrame,
        sec: int = 2,
        bonn_overlap: float = 0.5,
        n_jobs: int = -1,
        is_classical_model: bool = True
) -> tuple[NDArray, NDArray, int]:
    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(process_bonn_sequence)(eeg_bonn_sequence, path, sec, bonn_overlap)
        for path, eeg_bonn_sequence in eeg_bonn_dataset.items()
    )

    x_segments = [r[0] for r in results]
    y_segments = [r[1] for r in results]

    x_segments, y_segments = np.array(x_segments), np.array(y_segments)

    return x_segments, y_segments, x_segments.shape[1]


# ZBIÓR TWITTER
# -1 outlier, 1 inlier
def set_labels_based_on_timestamp(df: pd.DataFrame, windows: list[list]) -> pd.DataFrame:
    df["label"] = 1
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    for window in windows:
        df.loc[
            (df["timestamp"] >= pd.Timestamp(window[0])) &
            (df["timestamp"] <= pd.Timestamp(window[1]))
        , "label"] = -1
    return df


def load_or_cache_twitter(cache_path: str = "twitter_cache.pkl"):
    combined_windows = {}
    with open(os.path.join("nab", "labels", "combined_windows.json")) as f:
        combined_windows = json.load(f)

    combined_windows = {
        os.path.basename(k): v for k, v in combined_windows.items()
        if k.startswith("artificialNoAnomaly") or k.startswith("artificialWithAnomaly")
    }

    if not Path("cache").exists():
        os.mkdir("cache")

    cache_file = Path(os.path.join("cache", cache_path))

    if cache_file.exists():
        load_pickle = open(cache_file, "rb")
        pickle.load(load_pickle)
        load_pickle.close()

    twitter = [
        set_labels_based_on_timestamp(
            pd.read_csv(os.path.join("nab", "artificialNoAnomaly", path)),
            combined_windows.get(path)
        ) for path in os.listdir(os.path.join("nab", "artificialNoAnomaly"))
    ]

    twitter.extend([
        set_labels_based_on_timestamp(
            pd.read_csv(os.path.join("nab", "artificialWithAnomaly", path)),
            combined_windows.get(path)
        ) for path in os.listdir(os.path.join("nab", "artificialWithAnomaly"))
    ])

    twitter_to_cache = open(cache_file, "wb")
    pickle.dump(twitter, twitter_to_cache)
    twitter_to_cache.close()

    return twitter


def process_twitter_sequence(
        twitter_dataset: pd.DataFrame,
        m: int = 60 * 6,
        twitter_overlap: float = 0.5
) -> tuple[NDArray, NDArray]:
    frequency_twitter = 1 / 5
    twitter_segments = sliding_window(
        twitter_dataset["value"],
        int(m * frequency_twitter),
        twitter_overlap
    )
    twitter_segments_label = sliding_window(
        twitter_dataset["label"],
        int(m * frequency_twitter),
        twitter_overlap
    )

    base = {-1: 0, 1: 0}
    twitter_segments_label = np.array([
        {**base, **Counter(sequence_label)}
        for sequence_label in twitter_segments_label
    ])
    twitter_segments_label = np.array([
        -1 if segment.get(-1) > segment.get(1) else 1
        for segment in twitter_segments_label
    ])

    return twitter_segments, twitter_segments_label


def get_twitter_segments(
        twitter,
        m: int = 60 * 6,
        twitter_overlap: float = 0.5,
        n_jobs: int = -1
) -> tuple[NDArray, NDArray, int]:
    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(process_twitter_sequence)(twitter_dataset, m, twitter_overlap)
        for twitter_dataset in twitter
    )

    x_segments = [r[0] for r in results]
    y_segments = [r[1] for r in results]

    x_segments = trim_to_shortest(x_segments)
    y_segments = trim_to_shortest(y_segments)

    return x_segments, y_segments, x_segments.shape[1]


# ZBIÓR MIT BIH
def set_labels_mit_bih(
        signal_length: int,
        ann_samples: np.ndarray,
        ann_symbols: list,
        normal_symbol: list[str] = "N"
):
    labels = np.full(signal_length, -1, dtype=np.int8)  # domyślnie anomalia

    if len(ann_samples) == 0:
        return labels

    samples = np.asarray(ann_samples)

    starts = samples[:-1]
    ends = samples[1:]

    starts = np.append(starts, samples[-1])
    ends = np.append(ends, signal_length)

    for start, end, symbol in zip(starts, ends, ann_symbols + [ann_symbols[-1]]):
        if symbol in normal_symbol:
            labels[start:end] = 1

    return labels


def load_or_cache_mit_bih(cache_path: str = "mit_bih_cache.pkl"):
    if not Path("cache").exists():
        os.mkdir("cache")

    cache_file = Path(os.path.join("cache", cache_path))

    if cache_file.exists():
        load_pickle = open(cache_file, "rb")
        pickle.load(load_pickle)
        load_pickle.close()

    mit_bih_paths = pd.read_csv(os.path.join("mit-bih", "RECORDS"), header=None, dtype=str)
    mit_bih_paths = mit_bih_paths.values.flatten()

    mit_bih = [{
        "record": wfdb.rdrecord(os.path.join("mit-bih", p)),
        "ann": wfdb.rdann(os.path.join("mit-bih", p), "atr")
    } for p in mit_bih_paths]

    with open(cache_file, "wb") as f:
        pickle.dump(mit_bih, f)

    return mit_bih


def process_mit_bih_sequence(
        sequence_info,
        sec: float,
        over: float
) -> tuple[NDArray, NDArray]:

    frequency_mit = sequence_info["ann"].fs

    sequence = sequence_info["record"].p_signal[:, 0]
    mit_segments = sliding_window(sequence, sec * frequency_mit, over)

    mit_segments_label = set_labels_mit_bih(
        signal_length=sequence.shape[0],
        ann_samples=sequence_info["ann"].sample,
        ann_symbols=sequence_info["ann"].symbol,
        normal_symbol=["N", "L", "R"]
    )
    mit_segments_label = sliding_window(mit_segments_label, sec * frequency_mit, over)

    base = {-1: 0, 1: 0}
    mit_segments_label = np.array([
        {**base, **Counter(sequence_label)}
        for sequence_label in mit_segments_label
    ])
    mit_segments_label = np.array([
        -1 if segment.get(-1) > segment.get(1) else 1
        for segment in mit_segments_label
    ])

    return mit_segments, mit_segments_label


def get_mit_bih_segments(
        mit_bih,
        sec: int = 2,
        mit_bih_overlap: float = 0.5,
        n_jobs: int = -1
) -> tuple[NDArray, NDArray, int]:
    results = Parallel(n_jobs=n_jobs, backend='loky', verbose=0)(
        delayed(process_mit_bih_sequence)(seq_info, sec, mit_bih_overlap)
        for seq_info in mit_bih
    )

    x_segments = [r[0] for r in results]
    y_segments = [r[1] for r in results]

    x_segments = np.array(x_segments)
    y_segments = np.array(y_segments)

    return x_segments, y_segments, x_segments.shape[1]