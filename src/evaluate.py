import evaluate, torch


def evaluate_model(model, eval_dataloader):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    f1 = evaluate.load("f1")
    metric = evaluate.load("accuracy")
    tol_loss = 0.0
    batches_count = 0
    model.eval()
    for batch in eval_dataloader:
        batch = {k:v.to(device) for k,v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        predictions = torch.argmax(logits,dim = -1)
        tol_loss += outputs.loss.item()
        batches_count += 1
        f1.add_batch(predictions=predictions, references = batch["labels"])
        metric.add_batch(predictions = predictions, references = batch["labels"])

    return metric.compute(), f1.compute(average="macro"), tol_loss/batches_count

            
        