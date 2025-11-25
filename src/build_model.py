from psychai.nn_builder import ModelSpec, Model, save_config, build_config_dict, save_pretrained, from_pretrained
from transformers import AutoTokenizer

def build_mlp(vocab_size, embed_size):
        spec = ModelSpec(vocab_size=vocab_size)

        spec.add_layer({
            "type": "embedding",
            "vocab_size": vocab_size,
            "embed_size": vocab_size,
            "kind": "one_hot",
        }, name="embed")

        spec.add_layer({
            "type": "linear",
            "in_features": vocab_size,
            "out_features": embed_size
        }, name="mlp")

        spec.add_layer({
            "type": "lm_head",
            "embed_size": embed_size,
            "vocab_size": vocab_size
        }, name="lm_head")

        model = Model(spec)
        return model

def build_elman(vocab_size, embed_size):
        spec = ModelSpec(vocab_size=vocab_size)

        spec.add_layer({
            "type": "embedding",
            "vocab_size": vocab_size,
            "embed_size": vocab_size,
            "kind": "one_hot",
        }, name="embed")

        spec.add_layer({
            "type": "elman",
            "input_size": vocab_size,
            "embed_size": embed_size,
        }, name="elman")

        spec.add_layer({
            "type": "lm_head",
            "embed_size": embed_size,
            "vocab_size": vocab_size
        }, name="lm_head")

        model = Model(spec)
        return model

def build_lstm(vocab_size, embed_size):
    spec = ModelSpec(vocab_size=vocab_size)

    spec.add_layer({
        "type": "embedding",
        "vocab_size": vocab_size,
        "embed_size": vocab_size,
        "kind": "one_hot",
    }, name="embed")
    
    spec.add_layer({
        "type": "lstm",
        "input_size": vocab_size,
        "embed_size": embed_size,
    }, name="lstm")
    
    spec.add_layer({
        "type": "lm_head",
        "embed_size": embed_size,
        "vocab_size": vocab_size,
        "bias": True,
    }, name="lm_head")
    
    model = Model(spec)
    return model
    
def build_jordan(vocab_size, embed_size):
    spec = ModelSpec(vocab_size=vocab_size)
    
    spec.add_layer({
        "type": "embedding",
        "vocab_size": vocab_size,
        "embed_size": vocab_size,
        "kind": "one_hot"
    }, name="embed")
    
    spec.add_layer({
        "type": "jordan",
        "input_size": vocab_size,
        "embed_size": embed_size,
        "output_size": vocab_size
    }, name="jordan")
    
    model = Model(spec)
    return model

def build_transformer(vocab_size, embed_size, block_size, num_heads=2, num_layers=2): 
    spec = ModelSpec(vocab_size=vocab_size)

    spec.add_layer({
        "type": "embedding",
        "vocab_size": vocab_size,
        "embed_size": embed_size,
        "kind": "learned",
    }, name="embed")

    spec.add_layer({
        "type": "position_embedding",
        "embed_size": embed_size,
        "block_size": block_size,
        "kind": "learned",
    }, name="pos_embed")

    for i in range(num_layers):
        spec.add_layer({
            "type": "decoder_block",
            "embed_size": embed_size,
            "block_size": block_size,
            "num_heads": num_heads,
            "dropout": 0.1,
            "activation": "gelu",
            "bias": True,
        }, name=f"transformer_block_{i}")

    spec.add_layer({
        "type": "lm_head",
        "embed_size": embed_size,
        "vocab_size": vocab_size,
        "bias": True,
    }, name="lm_head")

    model = Model(spec)
    return model

def main():
    tokenizer = AutoTokenizer.from_pretrained("./models/je_2024/je_tokenizer")
    vocab_size = tokenizer.vocab_size
    embed_size = 16
    block_size = 2
    model_type = "transformer"
    model = None

    if model_type == "mlp":
        model = build_mlp(vocab_size, embed_size)
    elif model_type == "elman":
        model = build_elman(vocab_size, embed_size)
    elif model_type == "lstm":
        model = build_lstm(vocab_size, embed_size)
    elif model_type == "jordan":
        model = build_jordan(vocab_size, embed_size)
    elif model_type == "transformer":
        model = build_transformer(vocab_size, embed_size, block_size)

    config_dict = build_config_dict(model, model_type=f"custom-{model_type}-{embed_size}")
    save_config(f"./test/models/{model_type}-{embed_size}", config_dict)

if __name__ == "__main__":
    main()

