from .mlp import MLP

def build_model(config, num_inputs):
    "Model builder."

    model_type = config.MODEL.NAME

    if model_type == 'mlp':
        model = MLP(
            num_inputs=num_inputs,
            num_hidden_layers=config.MODEL.NUM_HIDDEN_LAYERS, 
            hidden_dim=config.MODEL.HIDDEN_DIM,
            lr=config.TRAIN.LR,
            weight_decay=config.TRAIN.WEIGHT_DECAY,
            max_epochs=config.TRAIN.EPOCHS
        )
    else:
        raise NotImplementedError(f"Unknown model: {model_type}")
    
    return model
 