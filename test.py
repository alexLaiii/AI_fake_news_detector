from src.test_fun import load_data_for_train
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.config import config
import torch
from sklearn.metrics import classification_report

if __name__ == "__main__":
    
    test_dataLoader = load_data_for_train(config)
    model = AutoModelForSequenceClassification.from_pretrained("models/best_model")
    tokenizer = AutoTokenizer.from_pretrained("models/best_model")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_dataLoader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(classification_report(all_labels, all_preds))

