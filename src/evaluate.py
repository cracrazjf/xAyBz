import json
import torch
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import permutation_test_score
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
from psychai.language.lm import LM_Evaluator
from psychai.config import EvaluationConfig, update_config
from psychai.visualization import plot_lines, plot_heatmaps, plot_heatmap, plot_scatter, plot_bars
from test_metric import test_metric


def main():
    cfg = EvaluationConfig()
    updates = {
        "model": {
            "name": "elman_16",
            "customized_model": True,
        },
        "data": {
            "test_path": "./data/je_exp1",
            "stride": 1,
            "pad_left": True,
            "drop_last": True,
            "batch_size": 1,
            "sequence_length": 2
        },
        "experiment_name": "Exp2",
        "experiment_directory": "./jordan_elman_2024/exp2/elman_item_level",
        "training_method": "continuous"
    }


    def plot_accuracies_across_models(config, accuracy_key_we_care="Legal B acc"):
        line_props = {
            "mlp": {"color": "black", "linestyle": "-"},
            "elman_item_level": {"color": "blue", "linestyle": "-"},
            "elman_sequence_level": {"color": "blue", "linestyle": "--"},
            "jordan_item_level": {"color": "orange", "linestyle": "-"},
            "jordan_sequence_level": {"color": "orange", "linestyle": "--"},
        }
        exp_root = Path(config.experiment_directory)
        models_runs = [p for p in exp_root.iterdir() if p.is_dir()]
        models_names = [p.name for p in exp_root.iterdir() if p.is_dir()]
        accuracy_lines = defaultdict(list)
        accuracy_sems = defaultdict(list)
        models_best_acc = defaultdict(list)

        for i, runs_path in enumerate(models_runs):
            accuracies = defaultdict(list)
            activations = defaultdict(list)
            
            log_files = [str(f) for f in runs_path.rglob("train_metrics_info.jsonl")]
            for log_file in log_files:
                accuracies_we_care = []
                with open(log_file, "r") as f:
                    for line in f:
                        record = json.loads(line)
                        if "accuracy" in record:
                            accuracies[f"epoch {record[f'epoch']}"].append(record["accuracy"])
                            accuracies_we_care.append(record["accuracy"][accuracy_key_we_care])
                models_best_acc[models_names[i]].append(max(accuracies_we_care))

            for epoch, accuracy in accuracies.items():
                df = pd.DataFrame(accuracy)
                avg_dict = df.mean().to_dict()
                sem_dict = df.sem().to_dict()
                for key, value in avg_dict.items():
                    if key == "Legal B acc":
                        accuracy_lines[models_names[i]].append(value)
                        accuracy_sems[models_names[i]].append(sem_dict[key])
        
        plot_lines(
            accuracy_lines,
            x_labels="Epochs",
            y_labels="Accuracy",
            title=f"{config.experiment_name} B-item Accuracy Across Runs",
            output_path=config.experiment_directory + "/accuracy_across_runs.png",
            sems=accuracy_sems,
            y_lim=(0, 1.1),
            line_props=line_props,
            x=[epoch.split(' ')[1] for epoch in accuracies.keys()]
        )

        plot_bars(
            data=models_best_acc,
            output_path=config.experiment_directory + "/best_accuracy_comparison_across_models.png",
            ylabel="Best Legal B-item Accuracy",
            title=f"{config.experiment_name} Best Accuracy Comparison Across Models"
        )

    def plot_accuracy_and_activation_across_runs(config):  
        runs_path = Path(config.experiment_directory)  
        accuracies = defaultdict(list)
        activations = defaultdict(list)
        log_files = [str(f) for f in runs_path.rglob("train_metrics_info.jsonl")]
        for log_file in log_files:
            with open(log_file, "r") as f:
                for line in f:
                    record = json.loads(line)
                    if "accuracy" in record:
                        accuracies[f"epoch {record[f'epoch']}"].append(record["accuracy"])
                    if "activation" in record:
                        activations[f"epoch {record[f'epoch']}"].append(record["activation"])

        accuracy_lines = defaultdict(list)
        accuracy_sems = defaultdict(list)
        accuracy_we_care = ["Legal B acc", "A acc", "y acc", ". acc"]
        for epoch, accuracy in accuracies.items():
            df = pd.DataFrame(accuracy)
            avg_dict = df.mean().to_dict()
            sem_dict = df.sem().to_dict()
            for key, value in avg_dict.items():
                if key in accuracy_we_care:
                    accuracy_lines[key].append(value)
                    accuracy_sems[key].append(sem_dict[key])
        plot_lines(
            accuracy_lines,
            x_labels="Epochs",
            y_labels="Accuracy",
            title=f"{config.model.name} Accuracy Across Runs",
            output_path=config.experiment_directory + "/accuracy_across_runs.png",
            sems=accuracy_sems,
            y_lim=(0, 1.1)
        )

        activation_lines = defaultdict(list)
        activation_sems = defaultdict(list)
        for epoch, activation in activations.items():
            df = pd.DataFrame(activation)
            sem_dict = df.sem().to_dict()
            avg_dict = df.mean().to_dict()
            for key, value in avg_dict.items():
                activation_lines[key].append(value)
                activation_sems[key].append(sem_dict[key])

        plot_lines(
            activation_lines,
            x_labels="Epochs",
            y_labels="Activation",
            title=f"{config.model.name} Mean Activations Across Runs",
            output_path=config.experiment_directory + "/activations_across_runs.png",
            sems=activation_sems,
            y_lim=(0, 1.1)
        )

    def save_weights_and_embeddings(output_path: str):
        eval_results = evaluator.evaluate_language_model()
        raw_vocab = evaluator.model_manager.tokenizer.get_vocab()
        id2token = {v: k for k, v in raw_vocab.items()}
        vocab = [id2token[i] for i in range(len(id2token))]
        special_tokens = [vocab[id] for id in evaluator.model_manager.tokenizer.all_special_ids]
        repr_dict = defaultdict(list)
        for model_path, eval_result in eval_results.items():
            embeddings = eval_result["embeddings"]
            token_embed_dict = defaultdict(list)
            if "elman" in config.MODEL_NAME.lower():
                input_weights = eval_result["weights"]["elman.hx.weight"].numpy().T
                output_weights = eval_result["weights"]["lm_head.proj.weight"].numpy()
                for ids, emb_map in embeddings.items():
                    token_id = emb_map["token_id"]
                    token_str = emb_map["token_string"]
                    hidden_repr = emb_map["layers"]["elman"]["hidden"]
                    token_embed_dict[token_str].append(hidden_repr)
            elif "lstm" in config.MODEL_NAME.lower():
                input_weights = eval_result["weights"]["lstm.cell.gates.weight"].numpy().T[:len(vocab), :]
                output_weights = eval_result["weights"]["lm_head.proj.weight"].numpy()
                for ids, emb_map in embeddings.items():
                    token_str = emb_map["token_string"]
                    hidden_repr = emb_map["layers"]["lstm"]["hidden"]
                    token_embed_dict[token_str].append(hidden_repr)
            elif "transformer" in config.MODEL_NAME.lower():
                input_weights = eval_result["weights"]["word_embed.emb.weight"].numpy()
                output_weights = eval_result["weights"]["lm_head.proj.weight"].numpy()
                for ids, emb_map in embeddings.items():
                    token_str = emb_map["token_string"]
                    hidden_repr = emb_map["layers"]["causal_self_attention"]["hidden"]
                    token_embed_dict[token_str].append(hidden_repr)
            
            embed_dict = {k: np.mean(v, axis=0) for k, v in token_embed_dict.items()}
            embed_labels = [k for k in vocab if k in embed_dict]
            embed_arr = np.array([embed_dict[k] for k in embed_labels])
            embed_correlations = np.corrcoef(embed_arr)
            embed_cor_df = pd.DataFrame(embed_correlations,
                                        index=embed_labels,
                                        columns=embed_labels)
            embed_cor_df = embed_cor_df.drop(index=special_tokens, columns=special_tokens, errors='ignore')
            repr_dict["embedding_correlations"].append(embed_cor_df)
            input_correlations = np.corrcoef(input_weights)
            input_cor_df = pd.DataFrame(input_correlations, 
                                        index=vocab,
                                        columns=vocab)
            input_cor_df = input_cor_df.drop(index=special_tokens, columns=special_tokens, errors='ignore')
            repr_dict["input_correlations"].append(input_cor_df)
            output_correlations = np.corrcoef(output_weights)
            output_cor_df = pd.DataFrame(output_correlations, 
                                        index=vocab,
                                        columns=vocab)
            output_cor_df = output_cor_df.drop(index=special_tokens, columns=special_tokens, errors='ignore')
            repr_dict["output_correlations"].append(output_cor_df)
            
        
        with open(output_path, "wb") as f:
            pickle.dump(repr_dict, f)
        return repr_dict

    def plot_correlations(repr_dict):
        for cor_type, df_list in repr_dict.items():
            print(f"Plotting {cor_type} heatmaps...")
            plot_heatmaps(
                df_list,
                output_path=config.MODEL_ROOT + f"/{cor_type}_heatmaps.png",
                ncols=6
            )

    def compute_clusters(correlations):
        for cor_type, df_list in correlations.items():
            for i, df in enumerate(df_list):
                dist = 1 - df
                dist = (dist + dist.T) / 2
                np.fill_diagonal(dist.values, 0)
                condensed_distance = squareform(dist)
                linkage = sch.linkage(condensed_distance, method='average')
                cluster_ids = sch.fcluster(linkage, t=0.3, criterion='distance')

                clusters = {}
                for label, cid in zip(df.index, cluster_ids):
                    clusters.setdefault(cid, []).append(label)
                
                three_items_cluster = {cid: items for cid, items in clusters.items() if len(items) == 3}
                for cid, items in three_items_cluster.items():
                    print(f"Correlation type: {cor_type}, Run {i}, Cluster ID {cid}: {items}")

                six_items_cluster = {cid: items for cid, items in clusters.items() if len(items) == 6}
                for cid, items in six_items_cluster.items():
                    print(f"Correlation type: {cor_type}, Run {i}, Cluster ID {cid}: {items}")


    cfg = update_config(cfg, updates)
    evaluation_task = "test_metric"
    evaluator = LM_Evaluator(cfg)
    if evaluation_task == "accuracy_across_models":
        plot_accuracies_across_models(config=cfg)
    elif evaluation_task == "accuracy_and_activation_across_runs":
        plot_accuracy_and_activation_across_runs(config=cfg)
    elif evaluation_task == "test_metric":
        evaluator.evaluate_language_model(compute_metrics=test_metric)
    elif evaluation_task == "embeddings":
        if not (runs_path / "weights_and_embeddings.pkl").exists():
            repr_dict =  save_weights_and_embeddings(output_path=config.MODEL_ROOT + "/weights_and_embeddings.pkl")
        else:
            with open(config.MODEL_ROOT + "/weights_and_embeddings.pkl", "rb") as f:
                repr_dict = pickle.load(f)
        plot_correlations(repr_dict)
        compute_clusters(repr_dict)

if __name__ == "__main__":
    main()