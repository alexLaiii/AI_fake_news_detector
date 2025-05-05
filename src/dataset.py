
from transformers import AutoTokenizer, DataCollatorWithPadding
from datasets import load_dataset
from torch.utils.data import DataLoader

def map_labels(example):
    label = example["label"]
    if label in [0, 1]:  # true, mostly-true
        return {"label": 0}
    elif label in [2, 3]:  # half-true, barely-true
        return {"label": 1}
    elif label in [4, 5]:  # false, pants-fire
        return {"label": 2}
    else:
        return {"label": -1}  # handle edge case if needed
    


# def tk_function(batch):
#     texts = []
#     for i in range(len(batch["statement"])):
#         parts = [str(batch["statement"][i])]

#         if "speaker" in batch and batch["speaker"][i]:
#             parts.append(f"Speaker: {str(batch['speaker'][i])}")

#         if "subject" in batch and batch["subject"][i]:
#             parts.append(f"Subject: {str(batch['subject'][i])}")

#         if "context" in batch and batch["context"][i]:
#             parts.append(f"Context: {str(batch['context'][i])}")

#         if "justification" in batch and batch["justification"][i]:
#             parts.append(f"Justification: {str(batch['justification'][i])}")

#         texts.append(" [SEP] ".join(parts))

#     return tokenizer(texts, truncation=True, padding=True)
    
def load_data(config):
    tokenizer = AutoTokenizer.from_pretrained(config["checkpoint"])
    # tokenization + return dataloaders
    raw_datasets = load_dataset(
    "csv",
    data_files={"train": "data/train.csv",
                "validation": "data/valid.csv",
                "test": "data/test.csv"
            }
    )

    raw_datasets = raw_datasets.map(map_labels)

    def tk_function(example):
        return tokenizer(example["statement"], truncation = True, padding = True)
    
    tk_datasets = raw_datasets.map(tk_function, batched = True)
    data_collator = DataCollatorWithPadding(tokenizer = tokenizer)

    tk_datasets = tk_datasets.rename_column("label", "labels")
    tk_datasets = tk_datasets.remove_columns( ["id", "statement", "date", "subject", "speaker",
    "speaker_description", "state_info", "true_counts", "mostly_true_counts",
    "half_true_counts", "mostly_false_counts", "false_counts",
    "pants_on_fire_counts", "context", "justification"])
    tk_datasets.set_format("torch")

    tr_dataloader = DataLoader(
        tk_datasets["train"], 
        shuffle=True,
        batch_size =config["batch_size"], 
        collate_fn=data_collator,
        )

    eval_dataloader = DataLoader(
        tk_datasets["validation"],
        batch_size = config["batch_size"],
        collate_fn=data_collator,
        )   

    return tr_dataloader, eval_dataloader, tokenizer






