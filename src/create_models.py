from psychai.nn_builder.nn_builder import ModelSpec
from psychai.nn_builder.nn_builder import Model, CausalLMWrapper
from transformers import PreTrainedTokenizerFast
from psychai.nn_builder.save_api import save_pretrained
import torch

def main():
    tokenizer_path = "./tokenizer/xaybz_tokenizer"
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

    def build_elman(tokenizer):
        vocab_size = len(tokenizer.get_vocab())
        hidden_dim = 16
        spec = ModelSpec(vocab_size=vocab_size)

        spec.add_layer({
            "type": "embedding",
            "num_embeddings": vocab_size,
            "embedding_dim": vocab_size,
            "kind": "one_hot",
        }, name="embed")

        spec.add_layer({
            "type": "elman",
            "input_size": vocab_size,
            "hidden_size": hidden_dim,
            "nonlinearity": "tanh",
            "batch_first": True,
        }, name="elman")

        spec.add_layer({
            "type": "lm_head",
            "hidden_size": hidden_dim,
            "vocab_size": vocab_size,
            "bias": True,
        }, name="lm_head")

        model = Model(spec)
        return model

    model = build_elman(tokenizer)
    save_pretrained(
    model,
    "./models/elman",
    model_type="custom-elman",
    extra_metadata={"notes": "Elman one-hot experiment"},
)


if __name__ == "__main__":
    main()

