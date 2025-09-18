import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F

import struct
import os
import csv
import json

from scipy.signal import resample
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, mean_absolute_error, mean_squared_error


def read_binary_from(ragged_array, bin_file):

    while True:
        chunk = bin_file.read(4)
        if not chunk:
            break
        sub_array_size = struct.unpack('i', chunk)[0]
        sub_array = np.array(struct.unpack(f'{sub_array_size}h', bin_file.read(sub_array_size * 2)))
        ragged_array.append(sub_array)

def load_data(filename: str, mode: str):

    ragged_array = []
    with open(filename, mode) as file:
        if mode == 'rb':
            read_binary_from(ragged_array, file)
        else:
            data = file.readlines()
            ragged_array = list(map(lambda p: int(p[0]), data))

    return ragged_array

def write_binary_to(filename, arrays):

    dir_name = "compressed/"
    os.makedirs(dir_name, exist_ok=True)
    file_path = dir_name + filename
    
    with open(file_path, "wb") as f:
        for array in arrays:
            f.write(struct.pack('i', len(array)))
            f.write(struct.pack(f'{len(array)}h', *array))

def save_losses(train_losses, vald_losses, filename):
    dir_name = "losses/"
    os.makedirs(dir_name, exist_ok=True)
    file_path = dir_name + filename
    np.savez(file_path, train_losses=train_losses, vald_losses=vald_losses)

def load_losses(filename):
    file_path = "losses/" + filename
    data = np.load(file_path)
    return data

def save_cf_report(report, filename):
    dir_name = "cf_reports/"
    os.makedirs(dir_name, exist_ok=True)
    file_path = dir_name + filename
    with open(file_path, "w") as outfile:
        json.dump(report, outfile, indent=4)

def load_cf_report(filename):
    file_path = "cf_reports/" + filename
    with open(file_path, "r") as openfile:
        cf_report = json.load(openfile)
    return cf_report

# Algorithm: Piecewise Constant Approximation (PCA)
def compress_ecg_data(
    ecg_data: list[np.ndarray], 
    compress_to: float,
    dtype: np.dtype = None
) -> list[np.ndarray]:

    assert 0.9 >= compress_to >= 0.1
    compressed_data = []
    for i, signal in enumerate(ecg_data):
        
        progress = (100 * i) / len(ecg_data)
        print("Compressing data... ({:.0f}%)".format(progress), end="\r")
        final_points = int(len(signal) * compress_to)
        splits = np.array_split(signal, final_points)
    
        if len(signal) % final_points == 0:
            compressed_signal = np.array(splits).mean(axis=1)
        else:
            compressed_signal = np.array([split.mean().item() for split in splits])
        
        if dtype is not None:
            compressed_signal = compressed_signal.astype(dtype)

        compressed_data.append(compressed_signal)
    
    perc = int(100 * compress_to)
    print(f"\rData successfully compressed to {perc}% of original size")

    return compressed_data

def compute_compression_metrics(X_train_raw, X_train_lossy, compr_size):

    maes, mses = [], []
    for signal, lossy_signal in zip(X_train_raw, X_train_lossy):

        final_points = int(len(signal) * compr_size)
        splits = np.array_split(signal, final_points)

        result = []
        for split, val in zip(splits, lossy_signal):
            k = [val] * len(split)
            result += k

        reconstructed = np.array(result)
        mae = mean_absolute_error(signal, reconstructed)
        mse = mean_squared_error(signal, reconstructed)
        maes.append(mae)
        mses.append(mse)
    
    return round(np.mean(maes).item(), 5), round(np.mean(mses).item(), 5)

def normalize_data(sequences, scaler_kind="standard"):
    concat = np.concatenate(sequences).reshape(-1, 1)
    scaler = StandardScaler() if scaler_kind == "standard" else MinMaxScaler()
    scaler.fit(concat)
    normalized_data = [scaler.transform(seq.reshape(-1, 1)).flatten() for seq in sequences]
    return normalized_data

def plot_signals(
    data, 
    labels, 
    num_rows, 
    num_cols, 
    start_idx=0, 
    total_sequences=10, 
    xlim_right=1200
):
  
  assert 0 <= start_idx < len(data)
  assert 0 < total_sequences and total_sequences == num_rows * num_cols
  # assert start_idx + total_sequences < len(data)

  fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 5 + num_rows))
  sequences = data[start_idx : start_idx + total_sequences]
  classes = labels[start_idx : start_idx + total_sequences]
  sorted_idcs = np.argsort(classes)

  offset = 0
  for i in range(num_rows):
    for j in range(num_cols):
      seq = sequences[sorted_idcs[offset]]
      clss = classes[sorted_idcs[offset]]
      axs[i, j].plot(seq, label=clss)
      axs[i, j].set_xlim(0, xlim_right)
      axs[i, j].set_xlabel("Time")
      axs[i, j].set_ylabel("Amplitude")
      axs[i, j].legend()
      offset += 1
  
  fig.suptitle("ECG-Signals of different classes (Normalized)", fontsize=16, y=0.95)
  plt.show()

def get_class_statistics(df: pd.DataFrame, _class: int):

    group = df.groupby("Label")["Data"].get_group(_class)
    all_stats = np.zeros(8)

    for series in group:
        stats = pd.Series(series).describe()
        all_stats += stats

    all_stats /= group.shape[0]
    return all_stats

def show_descriptive_statistics(df: pd.DataFrame):

    boxplot_data = []
    for i in range(4):
        _, mean, std, min, quant_25, median, quant_75, max = get_class_statistics(df, i)
        boxplot_data.append({"mean": mean, "std": std, "min": min, "q1": quant_25, "median": median, "q3": quant_75, "max": max})
    
    headers = [f"-- Class {i} --" for i in range(len(boxplot_data))]
    rows = [
        [f"Mean: {data["mean"]:.2f}" for data in boxplot_data],
        [f"Std.: {data["std"]:.2f}" for data in boxplot_data],
        [f"Min: {data["min"]:.2f}" for data in boxplot_data],
        [f"Max: {data["max"]:.2f}" for data in boxplot_data],
        [f"25%: {data["q1"]:.2f}" for data in boxplot_data],
        [f"50%: {data["median"]:.2f}" for data in boxplot_data],
        [f"75%: {data["q3"]:.2f}" for data in boxplot_data],
    ]

    col_width = 22
    print("".join(h.ljust(col_width) for h in headers))
    for row in rows:
        print("".join(cell.ljust(col_width) for cell in row))

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, stats in enumerate(boxplot_data, start=1):
        # box from q1 to q3
        ax.broken_barh([(stats["q1"], stats["q3"] - stats["q1"])], (i - 0.25, 0.5), facecolors="lightblue")
        # median as red line
        ax.plot([stats["median"], stats["median"]], [i - 0.25, i + 0.25], color="red", linewidth=2)
        # whiskers: min to q1 and q3 to max
        ax.plot([stats["min"], stats["q1"]], [i, i], color="black")
        ax.plot([stats["q3"], stats["max"]], [i, i], color="black")
        # min and max as dots
        ax.plot(stats["min"], i, marker="o", color="black")
        ax.plot(stats["max"], i, marker="o", color="black")
    
    ax.set_yticks(range(1, len(boxplot_data) + 1))
    ax.set_yticklabels([f"class {i}" for i in range(len(boxplot_data))])
    ax.set_xlabel("Value")
    ax.set_title("Boxplot based on descriptive statistics")
    plt.grid(True, axis="x")
    plt.show()

def show_loss(train_losses, vald_losses, total_epochs):

    x = np.arange(1, total_epochs + 1)
    plt.figure(figsize=(5, 5))
    plt.plot(x, train_losses, label="Train Error")
    plt.plot(x, vald_losses, label="Validation Error")
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.title("Train and Validation Loss")
    plt.legend()
    plt.show()

def show_metrics(
    y_vald, 
    y_pred, 
    all_classes, 
    dataset="Train", 
    return_cf_report=False
):

    conf_mat = confusion_matrix(y_vald, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=all_classes)
    disp.plot(ax=ax)
    ax.tick_params(axis='x', labelsize=7)
    ax.tick_params(axis='y', labelsize=7)
    plt.title(f"Confusion Matrix for {dataset}-Set")
    plt.show()

    cf_report = classification_report(y_vald, y_pred, output_dict=True)
    df_cf_report = pd.DataFrame(cf_report).transpose()
    print("\n", df_cf_report.round(2))

    if return_cf_report:
        return cf_report

def evaluate_model(
    model,
    data_loader,
    device,
    all_classes,
    dataset="Train",
    return_cf_report=False
):
    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for data, target in data_loader:
            # forward pass
            data = data.to(device)
            output = model(data)
            # compute predictions
            pred = output.argmax(dim=1).tolist()
            y_pred += pred
            y_true += target.tolist()

    return show_metrics(y_true, y_pred, all_classes, dataset=dataset, return_cf_report=return_cf_report)

def save_inferred_labels(
    model,
    inference_loader,
    device,
    file_name
):
    # run inference
    model.eval()
    inferred_labels = []
    with torch.no_grad():
        for data in inference_loader:
            data = data.to(device)
            output = model(data)
            pred = output.argmax(dim=1).tolist()
            inferred_labels += pred

    # save inferred labels
    dir_name = "output/"
    os.makedirs(dir_name, exist_ok=True)
    file_path = dir_name + file_name
    with open(file_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for label in inferred_labels:
            writer.writerow([label])
    print(f"{file_name} successfully generated")

def resample_signal(x, interp_len=9000, dtype=np.float32):

    signal = x
    # crop signal at random positions if possible
    if len(signal) > interp_len:
        start = np.random.randint(0, len(signal) - interp_len) 
        signal = signal [start:start + interp_len]

    # resample signal
    elif len(signal) < interp_len:
        signal = resample(signal, interp_len) 

    if dtype == np.float32:
        signal = signal.astype(dtype)
    else:
        signal = torch.tensor(signal, dtype=dtype)  

    return signal
