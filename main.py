from src.config import config
from src.dataset import load_data
from src.model import build_model
from src.train import train_model
from src.evaluate import evaluate_model
from src.utils import  save_model, save_metrics,log_run, get_best_f1_from_csv, plot_losses, get_row_count
# Damn, only import to serve loading previous model
from transformers import AutoModelForSequenceClassification, AutoTokenizer

if __name__ == "__main__":
    tr_dataloader, eval_dataloader, tokenizer = load_data(config)
    if not config["SkipTrain"]:
        if config["loadBestModel"]:
            model = AutoModelForSequenceClassification.from_pretrained("models/best_model")
            tokenizer = AutoTokenizer.from_pretrained("models/best_model")
            
        else:
            model = build_model(config)

        #training_loss = train_model(config, tr_dataloader, model)
        epoch_losses = train_model(config, tr_dataloader, model)
        # Evaluate and compare
        accuracy, f1_loss, val_loss = evaluate_model(model, eval_dataloader)
        current_f1 = f1_loss["f1"]
        best_f1 = get_best_f1_from_csv()

        if current_f1 > best_f1:
            print(f"📈 New best F1: {current_f1:.4f} > {best_f1:.4f} — saving model.")
            save_model(model, tokenizer, "models/best_model")
            isbest = True
        else:
            print(f"⏩ F1 {current_f1:.4f} did not beat best {best_f1:.4f} — model not saved.")
            isbest = False

        
        if(config["save_log"]):


            log_run({
                "isbest": isbest,
                "epoch": config["epochs"],
                "train_loss": float(sum(epoch_losses) / len(epoch_losses)),
                "val_loss": float(val_loss),
                "accuracy": float(accuracy["accuracy"]),
                "f1": float(current_f1),
                "learning_rate": config["learning_rate"],
                "batch_size": config["batch_size"],
                "warmup_steps": config["warmup_steps"],
                "scheduler": config["schedulerType"],
                "checkpoint": config["checkpoint"],
                "pretrained": config["loadBestModel"]
            }, "results/runs.csv")
            row_count = get_row_count()
            plot_losses(epoch_losses, f"results/loss_graph/training_loss_{row_count}.png")
    else:
        # Just load and evaluate, no logging or saving
        if config["loadBestModel"]:
            model = AutoModelForSequenceClassification.from_pretrained("models/best_model")
            tokenizer = AutoTokenizer.from_pretrained("models/best_model")
        else:
            model = build_model(config)
        accuracy, f1_loss, val_loss = evaluate_model(model, eval_dataloader)
        print(f"🔍 Evaluation only — Accuracy: {accuracy['accuracy']:.4f}, F1: {f1_loss['f1']:.4f}, validation loss: {val_loss:.4f}")
        
        
        
        
        
        
        
        
