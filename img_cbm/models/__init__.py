from .mirage_img import *

# Optional modules are not always present in deployment images.
try:
    from .mirage_txt import *
except Exception:
    pass

try:
    from .tbm_api import TBMEncoderAPI
except Exception:
    TBMEncoderAPI = None

def get_model(config):
    """
    Retrieves the model class specified in the config and initializes it with provided parameters.
    """
    model_name = config['model']['name']
    model_params = config['model'].get('params', {})
    
    # Mapping model names to their classes
    model_classes = {
        # Original linear models
        "img-linear": ImageLinearModel,
        "cbm-encoder": ObjectClassCBMEncoder,  # All 300 classifiers in one model
        "cbm-predictor": ObjectClassCBMPredictor,
        "mirage-img": MiRAGeImg,
        "txt-linear": globals().get("TextLinearModel"),
        "tbm-predictor": globals().get("TBMPredictor"),
        "tbm-encoder": globals().get("TBMEncoder"),
        "mirage-txt": globals().get("MiRAGeTxt"),
        
        # MLP versions (防止 overfitting)
        "img-linear-mlp": ImageMLPModel,
        "cbm-encoder-mlp": ObjectClassCBMMLPEncoder,
        "cbm-predictor-mlp": ObjectClassCBMMLPPredictor,
        "mirage-img-mlp": globals().get("MiRAGeImgMLP"),
        # Add other models here as needed
    }

    model_classes = {k: v for k, v in model_classes.items() if v is not None}
    
    if model_name in model_classes:
        model_class = model_classes[model_name]
        return model_class(**model_params)  # Instantiate model with parameters
    elif model_name == 'tbm-encoder-api' and TBMEncoderAPI is not None:
        return TBMEncoderAPI(**model_params)
    else:
        raise ValueError(f"Model {model_name} not recognized. Please check config.")
