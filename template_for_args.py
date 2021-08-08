# Author: Leda Sari
import json

args_dict = {
    "train_set": "data/features_dev-clean.csv",
    "valid_set": "data/features_dev-clean.csv",
    "labels_file": "data/labels.json",
    "batch_size": 4,
    "num_workers": 0,
    "use_gpu": True,
    "lr": 0.0001,
    "clip": 10,
    "pad_token": 0,
    "blank_symbol": None,
    "num_epochs": 2,
    "bidirectional": False,
    "rnn_input_dim": 161,
    "nb_layers": 5,
    "rnn_hidden_size": 768,
    "num_classes": 29,
    "rnn_type": "lstm",
    "context": 20
}

with open("train_args.json", 'w') as f:
    json.dump(args_dict, f, indent=4)
