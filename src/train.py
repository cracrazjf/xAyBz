from typing import Dict, Any
from psychai.config.lm_train import LM_TrainConfig
from psychai.trainer.nn_trainer import NN_Trainer
from psychai.artificial_dataset.xAyBz import XAYBZ
import numpy as np

def main():
    def build_training_config() -> LM_TrainConfig:
        cfg_overrides: Dict[str, Any] = {}

        # Model
        cfg_overrides["CUSTOMIZED_MODEL"] = True
        cfg_overrides["MODEL_NAME"] = "elman"
        cfg_overrides["MODEL_PATH"] = "./models/elman"
        cfg_overrides["TASK"] = "causal_lm"
        cfg_overrides["TOKENIZER_PATH"] = "./tokenizer/xaybz_tokenizer"

        # Random State
        cfg_overrides["RANDOM_SEED"] = 66

        # Data
        cfg_overrides["DATA_NAME"] = "xaybz"
        cfg_overrides["DATA_TYPE"] = "text"
        cfg_overrides["TRAIN_DATA_PATH"] = "./data/Dataset_2_3_1_0_0_0_0_0_3_1_1_0_0_0_0_all_pairs_1_massed_0_massed_66/Document:0 Group:all Len:36.jsonl"
        cfg_overrides["EVAL_DATA_PATH"] = "./data/Dataset_2_3_1_0_0_0_0_0_3_1_1_0_0_0_0_all_pairs_1_massed_0_massed_66/Document:0 Group:all Len:36.jsonl"
        cfg_overrides["SHUFFLE_DATASET"] = True
        cfg_overrides["SHUFFLE_DATALOADER"] = False
        cfg_overrides["OVERLAPPING_SEQUENCES"] = True
        cfg_overrides["SEQUENCE_LENGTH"] = 2
        cfg_overrides["OVERLAPPING_STRIDE"] = 1
        cfg_overrides["DATA_PROCESS_BATCH_SIZE"] = 4
        cfg_overrides["DATA_PROCESS_NUM_PROC"] = 2
        cfg_overrides["PIN_MEMORY"] = False
        cfg_overrides["DROP_LAST"] = False
        cfg_overrides["DATALOADER_WORKERS"] = 1

        # Training
        cfg_overrides["LEGACY_TRAIN"] = True  # This is specifically for RNN model
        cfg_overrides["NUM_EPOCHS"] = 200
        cfg_overrides["OPTIMIZER"] = "adamw"
        cfg_overrides["LEARNING_RATE"] = 0.001
        cfg_overrides["WEIGHT_DECAY"] = 0.01
        cfg_overrides["BATCH_SIZE"] = 1
        cfg_overrides["RANDOM_STATE"] = 66
        

        # Evaluation / Saving / Logging
        cfg_overrides["EVAL_STEPS"] = 10
        cfg_overrides["LOGGING_DIR"] = "./logs/elman"
        cfg_overrides["OUTPUT_DIR"] = "./outputs/elman"
        return LM_TrainConfig(**cfg_overrides)

    training_config = build_training_config()

    def reconstruct_from_stride1_windows(flat, k=training_config.SEQUENCE_LENGTH-1):
        if k < 1 or len(flat) < k:
            raise ValueError("k must be >= 1 and flat must have length >= k")
        # First k-1 tokens, then every k-th token starting at index k-1
        return flat[:k-1] + [flat[i] for i in range(k-1, len(flat), k)]

    def split_on_target(labels, preds, logits, target):
        labels_result, preds_result = [], []
        labels_current, preds_current = [], []
        logits_result, logits_current = [], []
        for i, token in enumerate(labels):
            labels_current.append(token)
            preds_current.append(preds[i])
            logits_current.append(logits[i])
            if token == target:
                labels_result.append(labels_current)
                preds_result.append(preds_current)
                labels_current, preds_current = [], []
                logits_result.append(logits_current)
                logits_current = []
        if labels_current:
            labels_result.append(labels_current)
            preds_result.append(preds_current)
            logits_result.append(logits_current)
        return labels_result, preds_result, logits_result

    def softmax(logits):
        exp_logits = np.exp(logits - np.max(logits, axis=0, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=0, keepdims=True)

    def test_metric(eval_dataset, all_labels, all_preds, all_logits, tokenizer):
        accuracy = {'B acc: ': 0, 
                    'Cat B acc: ': 0, 
                    'Legal B acc: ': 0, 
                    'A acc: ': 1, 
                    'Cat A acc: ': 1, 
                    'Legal A acc: ': 1,
                    'y acc: ': 0,
                    '. acc: ': 0}

        activation = {'Legal_B': 0, 'Omitted_B': 0, 'Illegal_B': 0, 'y': 0, 'A': 0}
        flattened_logits = all_logits.reshape(-1, all_logits.shape[-1])
        reconstructed_labels = reconstruct_from_stride1_windows(tokenizer.batch_decode(all_labels.flatten()))
        reconstructed_preds = reconstruct_from_stride1_windows(tokenizer.batch_decode(all_preds.flatten()))
        labels_by_sentence, preds_by_sentence, logits_by_sentence = split_on_target(reconstructed_labels, reconstructed_preds, flattened_logits, '.')
        vocab_to_index = tokenizer.get_vocab()
        for i, (labels, preds, logits) in enumerate(zip(labels_by_sentence, preds_by_sentence, logits_by_sentence)):
            legality_labels = eval_dataset[i]['label']
            label_to_indices = {label: [i for i, v in enumerate(legality_labels) if v == label]
                     for label in set(legality_labels)}
            for label, pred, logit in zip(labels, preds, logits):
                logit = softmax(logit)
                if label[0] == pred[0] == '.':
                    accuracy['. acc: '] += 1
                elif label[0] == pred[0] == 'y':
                    accuracy['y acc: '] += 1
                elif label[0] == pred[0] == 'A':
                    accuracy['A acc: '] += 1
                    pred_cat, pred_idx = XAYBZ._parse_cat_idx(pred)
                    label_cat, label_idx = XAYBZ._parse_cat_idx(label)
                    if pred_cat == label_cat:
                        accuracy[f'Cat A acc: '] += 1
                        if vocab_to_index[pred] in label_to_indices['A_Legal']:
                            accuracy[f'Legal A acc: '] += 1
                elif label[0] == pred[0] == 'B':
                    accuracy['B acc: '] += 1
                    pred_cat, pred_idx = XAYBZ._parse_cat_idx(pred)
                    label_cat, label_idx = XAYBZ._parse_cat_idx(label)
                    if pred_cat == label_cat:
                        accuracy[f'Cat B acc: '] += 1
                        if vocab_to_index[pred] in label_to_indices['B_Legal']:
                            accuracy[f'Legal B acc: '] += 1
                if label[0] == 'B':
                    activation['Legal_B'] += np.mean(logit[label_to_indices['B_Legal']])
                    activation['Omitted_B'] += np.mean(logit[label_to_indices['B_Omitted']])
                    activation['Illegal_B'] += np.mean(logit[label_to_indices['B_Illegal']])
                    activation['y'] += np.mean(logit[label_to_indices['y']])
                    activation['A'] += np.mean(np.concatenate([logit[label_to_indices['A_Legal']], logit[label_to_indices['A_Omitted']], logit[label_to_indices['A_Illegal']]]))
        for key, value in accuracy.items():
            accuracy[key] = f"{value / len(labels_by_sentence):.2%}"
        for key, value in activation.items():
            activation[key] = f"{value / len(labels_by_sentence):.4f}"
        metric_info = {}
        metric_info['accuracy'] = accuracy
        metric_info['activation'] = activation
        return metric_info
    
    
    trainer = NN_Trainer(training_config)
    trainer.train_language_model(test_metric)

if __name__ == "__main__":
    main()
    