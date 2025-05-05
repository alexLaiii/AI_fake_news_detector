1. run this command to install all the lib
    pip install -r requirements.txt

2. You can tweak training behavior here:
   config = {
    "checkpoint"     : "bert-base-uncased",   # 🔒 Do not change
    "num_labels"     : 3,                     # 🔒 Do not change
    "schedulerType"  : "linear",              # ⚙️  Check Transformers docs for other options
    "learning_rate"  : 5e-5,                  # 🔢 Modify as needed
    "warmup_steps"   : 0,                     # 🔢 Set warmup steps
    "batch_size"     : 64,                    # 🧮 Recommended: 32–128 for 18k samples
    "epochs"         : 4,                     # 🔁 How many times to loop over the training set
    "loadBestModel"  : False,                 # ✅ True to start from last best model
    "SkipTrain"      : False,                 # ✅ True to skip training and only evaluate
    "save_log"       : True                   # 📝 Save training result and loss graph
}
3. Training results

    Results are saved in results/runs.csv

    Loss graphs are stored in results/loss_graph/
    