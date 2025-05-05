import os
import csv 
import pandas as pd
import matplotlib.pyplot as plt

def save_model(model, tokenizer, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

def save_metrics(metrics, path):
    df = pd.DataFrame(metrics)
    df.to_csv(path, index=False)

def log_run(metrics, file_path):
    file_exists = os.path.isfile(file_path)
    with open(file_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)

def get_best_f1_from_csv(path="results/runs.csv"):
    if not os.path.exists(path):
        return 0.0
    df = pd.read_csv(path)
    return df["f1"].max()

def plot_losses(epoch_losses,save_path="results/training_loss.png"):
    epochs = list(range(1, len(epoch_losses) + 1))
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, epoch_losses, marker='o', linestyle='-', color='blue')
    plt.xlabel("Epoch")
    plt.ylabel("Average Training Loss")
    plt.title("Training Loss Over Epochs")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)  # Saves the figure to a file
    plt.show()              # Displays the figure

def get_row_count():
    df = pd.read_csv("results/runs.csv")
    return len(df)