from psychai.evaluator.custom import Custom_Evaluator
from psychai.config.lm_eval import LM_EvalConfig
from collections import defaultdict
import numpy as np
from typing import Dict, Any

def main():
    def build_evaluation_config() -> LM_EvalConfig:
        cfg_overrides: Dict[str, Any] = {}
        cfg_overrides["MODEL_ROOT"] = "./xaybz-2022/lstm-16"
        cfg_overrides["MODEL_NAME"] = "lstm-16"
        cfg_overrides["TASK"] = "causal_lm"
        cfg_overrides["BATCH_SIZE"] = 1
        cfg_overrides["CUSTOMIZED_MODEL"] = True
        cfg_overrides["TRUST_REMOTE_CODE"] = True
        cfg_overrides["SEQUENCE_LENGTH"] = 3
        cfg_overrides["OVERLAPPING_SEQUENCES"] = True
        cfg_overrides["PIN_MEMORY"] = False
        cfg_overrides["DROP_LAST"] = False
        cfg_overrides["EVAL_DATA_PATH"] = "./data/xAyBz_2_3_1_0_0_0_0_0_3_1_1_0_0_0_0_all_pairs_1_massed_0_massed_66/Document:0 Group:all Len:36.jsonl"
        return LM_EvalConfig(**cfg_overrides)

    def mean_activations(token_map, layer: str, key: str, by: str = "id"):
        buckets = defaultdict(list)
        for rec in token_map.values():
            if by == "id":
                group_key = rec["token_id"]
            elif by == "str":
                group_key = rec.get("token_str", str(rec["token_id"]))
            else:
                raise ValueError("by must be 'id' or 'str'")

            vec = rec["layers"][layer][key]
            buckets[group_key].append(vec)
        means = {k: np.mean(np.stack(v, axis=0), axis=0) for k, v in buckets.items()}
        return means
        
    config = build_evaluation_config()
    evaluator = Custom_Evaluator(config)
    token_representations_map, weights = evaluator.evaluate_language_model()
    mean_activations = mean_activations(token_representations_map, "lstm", "hidden", by="str")

if __name__ == "__main__":
    main()