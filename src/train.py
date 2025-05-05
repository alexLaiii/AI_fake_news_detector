# import torch
# from torch.optim import AdamW
# from transformers import get_scheduler
# from tqdm.auto import tqdm



# def train_model(config, training_dataloader, model):

#     device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#     model.to(device)
#     optimizer = AdamW(model.parameters(), lr = config["learning_rate"])
#     num_epochs = config["epochs"]
#     num_training_steps = num_epochs * len(training_dataloader)
#     lr_scheduler = get_scheduler(
#         "linear", 
#         optimizer = optimizer,
#         num_warmup_steps = config["warmup_steps"],
#         num_training_steps = num_training_steps
#     )

#     progress_bar  = tqdm(range(num_training_steps))

#     model.train()

#     tol_loss = 0.0
#     batches_count = 0
    
#     for epoch in range(num_epochs):
#         for batch in training_dataloader:
#             batch = {k:v.to(device) for k,v in batch.items()}
#             outputs = model(**batch)
#             loss = outputs.loss
#             tol_loss += loss
#             batches_count += 1
#             loss.backward()
#             optimizer.step()
#             lr_scheduler.step()
#             optimizer.zero_grad()
#             progress_bar.update(1)

#     return tol_loss /  batches_count
            
import torch
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm

def train_model(config, training_dataloader, model):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=config["learning_rate"])
    num_epochs = config["epochs"]
    num_training_steps = num_epochs * len(training_dataloader)
    lr_scheduler = get_scheduler(
        config["schedulerType"],
        optimizer=optimizer,
        num_warmup_steps=config["warmup_steps"],
        num_training_steps=num_training_steps
    )

    progress_bar = tqdm(range(num_training_steps))
    model.train()

    epoch_losses = []

    for epoch in range(num_epochs):
        tol_loss = 0.0
        batches_count = 0

        for batch in training_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            tol_loss += loss.item()
            batches_count += 1

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        avg_epoch_loss = tol_loss / batches_count
        epoch_losses.append(avg_epoch_loss)

    return epoch_losses


    

    
