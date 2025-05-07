
config = {
    "checkpoint" : "bert-base-uncased",
    "num_labels" : 3,
    "schedulerType" : "cosine",
    "learning_rate" : 1e-5,
    "warmup_steps" : 0, 
    "batch_size" : 16,
    "epochs" : 2,
    "loadBestModel" : True,
    "SkipTrain" : False,
    "save_log": True
}