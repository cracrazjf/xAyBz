import numpy as np
from typing import Dict, Any
from psychai.config import TrainingConfig, update_config
from psychai.language.lm import LM_Trainer
from test_metric import test_metric


def main():
    cfg = TrainingConfig()
    updates = {
        "model": {
            "name": "lstm_16",
            "path": "./test/models/lstm-16",
            "tokenizer_path": "./models/je_2024/je_tokenizer",
            "customized_model": True,
            # "weight_init": {"layers.lstm.cell.gates.weight": ("uniform", 1)}
        },
        "data": {
            "train_path": "./data/je_exp2",
            "val_path": "./data/je_exp2",
            "shuffle_dataset": True,
            "stride": 1,
            "pad_left": True,
            "drop_last": False,
            "batch_size": 2,
            "sequence_length": 3
        },
        "optim": {
            "lr": 0.01,
            "optimizer": "adamw"
        },
        "logging": {
            "metric_for_best_model": "legal_b_accuracy",
            "save_total_limit": 3,
            "save_model": True,
            "eval_interval": 10
        },
        "experiment_name": "test_exp2",
        "experiment_directory": "./test/exp2/lstm_16",
        "training_method": "bptt",
        "num_runs": 1,
        "num_epochs": 500,
        "seed": 66
    }
    cfg = update_config(cfg, updates)
    trainer = LM_Trainer(cfg)
    trainer.train(test_metric=test_metric)

if __name__ == "__main__":
    main()
    