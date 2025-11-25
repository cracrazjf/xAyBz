import numpy as np
from psychai.artificial_dataset import XAYBZ

def reconstruct_from_stride1_windows(flat, k):
    if k < 1 or len(flat) < k:
        raise ValueError("k must be >= 1 and flat must have length >= k")
    if type(flat) == list:  
        return flat[:k-1] + [flat[i] for i in range(k-1, len(flat), k)]
    elif flat.ndim == 2:    
        prefix = flat[:k-1, :]
        new_tokens = flat[np.arange(k-1, len(flat), k), :]
        return np.concatenate([prefix, new_tokens], axis=0)

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
            logits_result.append(logits_current)
            labels_current, preds_current, logits_current = [], [], []
    if labels_current:
        labels_result.append(labels_current)
        preds_result.append(preds_current)
        logits_result.append(logits_current)
    return labels_result, preds_result, logits_result

def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits, axis=0, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=0, keepdims=True)

def test_metric(eval_dataset, all_labels, all_preds, all_logits, tokenizer, config):
    accuracy = {'B acc': 0, 
                'Cat B acc': 0, 
                'Legal B acc': 0, 
                'A acc': 0, 
                'Cat A acc': 0, 
                'Legal A acc': 0,
                'y acc': 0,
                '. acc': 0}

    activation = {'Legal_B': 0, 'Omitted_B': 0, 'Illegal_B': 0, 'y': 0, 'A': 0}
    reconstructed_logits = reconstruct_from_stride1_windows(all_logits.reshape(-1, all_logits.shape[-1]), k= config.data.sequence_length-1)
    reconstructed_labels = reconstruct_from_stride1_windows(tokenizer.batch_decode(all_labels.flatten()), k= config.data.sequence_length-1)
    reconstructed_preds = reconstruct_from_stride1_windows(tokenizer.batch_decode(all_preds.flatten()), k= config.data.sequence_length-1)
    labels_by_sentence, preds_by_sentence, logits_by_sentence = split_on_target(reconstructed_labels, reconstructed_preds, reconstructed_logits, '.')
    vocab_to_index = tokenizer.get_vocab()
    for i, (labels, preds, logits) in enumerate(zip(labels_by_sentence, preds_by_sentence, logits_by_sentence)):
        legality_labels = eval_dataset[i]['label']
        label_to_indices = {label: [i for i, v in enumerate(legality_labels) if v == label]
                    for label in set(legality_labels)}
        for label, pred, logit in zip(labels, preds, logits):
            logit = softmax(logit)
            if label[0] == pred[0] == '.':
                accuracy['. acc'] += 1
            elif label[0] == pred[0] == 'y':
                accuracy['y acc'] += 1
            elif label[0] == pred[0] == 'A':
                accuracy['A acc'] += 1
                pred_cat, pred_idx = XAYBZ._parse_cat_idx(pred)
                label_cat, label_idx = XAYBZ._parse_cat_idx(label)
                if pred_cat == label_cat:
                    accuracy[f'Cat A acc'] += 1
                    if vocab_to_index[pred] in label_to_indices['A_Legal']:
                        accuracy[f'Legal A acc'] += 1
            elif label[0] == pred[0] == 'B':
                accuracy['B acc'] += 1
                pred_cat, pred_idx = XAYBZ._parse_cat_idx(pred)
                label_cat, label_idx = XAYBZ._parse_cat_idx(label)
                if pred_cat == label_cat:
                    accuracy[f'Cat B acc'] += 1
                    if vocab_to_index[pred] in label_to_indices['B_Legal']:
                        accuracy[f'Legal B acc'] += 1
            if label[0] == 'B':
                activation['Legal_B'] += np.mean(logit[label_to_indices['B_Legal']])
                if label_to_indices.get('B_Omitted', None) is not None:
                    activation['Omitted_B'] += np.mean(logit[label_to_indices['B_Omitted']])
                if label_to_indices.get('B_Illegal', None) is not None:
                    activation['Illegal_B'] += np.mean(logit[label_to_indices['B_Illegal']])
                activation['y'] += np.mean(logit[label_to_indices['y']])
                A_category_logits = [logit[label_to_indices['A_Legal']]]
                if label_to_indices.get('A_Omitted', None) is not None:
                    A_category_logits.append(logit[label_to_indices['A_Omitted']])
                if label_to_indices.get('A_Illegal', None) is not None:
                    A_category_logits.append(logit[label_to_indices['A_Illegal']])
                activation['A'] += np.mean(np.concatenate(A_category_logits))
    for key, value in accuracy.items():
        accuracy[key] = round(value / len(labels_by_sentence), 2)
    for key, value in activation.items():
        activation[key] = round(value / len(labels_by_sentence), 4)
    metric_info = {}
    metric_info['accuracy'] = accuracy
    metric_info['activation'] = activation
    return metric_info