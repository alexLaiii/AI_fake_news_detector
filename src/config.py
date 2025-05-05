
config = {
    "checkpoint" : "bert-base-uncased",
    "num_labels" : 3,
    "schedulerType" : "linear",
    "learning_rate" : 5e-5,
    "warmup_steps" : 0, 
    "batch_size" : 32,
    "epochs" : 3,
    "loadBestModel" : False,
    "SkipTrain" : False,
    "save_log": True
}