from transformers import AutoModelForSequenceClassification

def build_model(config):
    model = AutoModelForSequenceClassification.from_pretrained(config["checkpoint"], num_labels=config["num_labels"])
    return model

