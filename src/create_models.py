from psychai.nn_builder.nn_builder import ModelSpec
from psychai.nn_builder.nn_builder import Model
from psychai.nn_builder.io import save_config, build_config_dict, save_pretrained, from_pretrained

def main():
    vocab_size = 19
    embed_size = 4
    def build_elman():
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

    # model = build_elman()
    # config_dict = build_config_dict(model, model_type=f"custom-elman-{embed_size}")
    # save_config(f"./models/elman-{embed_size}", config_dict)

    def build_lstm():
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

    # model = build_lstm()
    # config_dict = build_config_dict(model, model_type=f"custom-lstm-{embed_size}")
    # save_config(f"./models/lstm-{embed_size}", config_dict)

    def build_jordan():
        vocab_size = 18
        hidden_dim = 16
        spec = ModelSpec(vocab_size=vocab_size)
        
        spec.add_layer({
            "type": "embedding",
            "num_embeddings": vocab_size,
            "embedding_dim": vocab_size,
            "kind": "one_hot",
        }, name="embed")
        
        spec.add_layer({
            "type": "jordan",
            "input_size": vocab_size,
            "hidden_size": hidden_dim,
            "output_size": vocab_size,
            "nonlinearity": "tanh",
            "batch_first": True,
        }, name="jordan")
        
        model = Model(spec)
        return model

    # model = build_jordan()
    # config_dict = build_config_dict(model, model_type="custom-jordan-16", extra_metadata={"notes": "Jordan with 18 vocab size and 16 hidden size and one-hot embedding"})
    # save_config("./models/jordan-16", config_dict)

    def build_transformer():
        spec = ModelSpec(vocab_size=vocab_size)
        
        spec.add_layer({
            "type": "embedding",
            "vocab_size": vocab_size,
            "embed_size": embed_size,
            "kind": "learned",
        }, name="word_embed")
        
        spec.add_layer({
            "type": "position_embedding",
            "embed_size": embed_size,
            "block_size": 3,
            "kind": "fixed",
        }, name="pos_embed")

        spec.add_layer({
            "type": "causal_self_attention",
            "block_size": 3,
            "embed_size": embed_size,
            "num_heads": 4,
        }, name="causal_self_attention")

        spec.add_layer({
            "type": "mlp",
            "embed_size": embed_size,
        }, name="mlp")

        spec.add_layer({
            "type": "lm_head",
            "embed_size": embed_size,
            "vocab_size": vocab_size,
            "bias": True,
        }, name="lm_head")

        model = Model(spec)
        return model

    model = build_transformer()
    config_dict = build_config_dict(model, model_type=f"transformer-{embed_size}")
    save_config(f"./models/transformer-{embed_size}", config_dict)
        

if __name__ == "__main__":
    main()

