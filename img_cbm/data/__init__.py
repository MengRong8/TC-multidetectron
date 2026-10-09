from .dataset import *
import yaml
from PIL import Image
import numpy as np
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

# 讀取 YAML 設定檔
def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

# 根據 config 初始化並回傳對應的 Dataset 物件
def get_dataset(config, test_set=None, is_eval=False):
    """
    Retrieves the dataset class specified in the config and initializes it with provided parameters.
    
    Args:
        config (dict): Configuration dictionary loaded from a YAML file.
        is_eval (bool): Flag indicating whether to load an evaluation dataset.
        test_set (str, optional): Specifies which test dataset to load.
        
    Returns:
        Tuple[Dataset, str] or List[Tuple[Dataset, str]]
    """
    dataset_classes = {
        "img-or-text": MiRAGeImageOrTextDataset,
        "multimodal": MiRAGeNewsDataset,
        "chinese-text": ChineseNewsDataset,
    }

    # Handle all test datasets case
    if test_set == "all":
        results = []
        for key, val in config.items():
            # consider keys like "..._dataset" or keys that start with "test"
            if not (key.endswith("_dataset") or key.startswith("test")):
                continue
            name = val.get("name")
            if name in dataset_classes:
                params = val.get("params", {}) or {}
                test_name = val.get("test_name", key)
                results.append((dataset_classes[name](**params), test_name))
        return results

    # Try different dataset keys in order of preference
    if test_set:
        dataset_key = test_set
    else:
        if is_eval:
            # 驗證集：優先使用 test_dataset, 其次 eval_dataset, val_dataset
            if 'test_dataset' in config:
                dataset_key = 'test_dataset'
            elif 'eval_dataset' in config:
                dataset_key = 'eval_dataset'
            elif 'val_dataset' in config:
                dataset_key = 'val_dataset'
            elif 'chinese_eval_dataset' in config:
                dataset_key = 'chinese_eval_dataset'
            else:
                raise KeyError("No eval dataset found in config")
        else:
            # 訓練集：優先使用 train_dataset
            if 'train_dataset' in config:
                dataset_key = 'train_dataset'
            elif 'chinese_train_dataset' in config:
                dataset_key = 'chinese_train_dataset'
            else:
                raise KeyError("No train dataset found in config")
    
    print(f"get_dataset selected dataset_key='{dataset_key}' (is_eval={is_eval})")
    
    dataset_info = config[dataset_key]
    dataset_name = dataset_info['name']
    dataset_params = dataset_info.get('params', {})
    test_name = dataset_info.get('test_name', dataset_key)
    
    # 檢查是否使用 API 模型
    model_name = config.get('model', {}).get('name', '')
    is_api_model = 'api' in model_name.lower()
    
    if dataset_name == 'chinese-text' and is_api_model:
        dataset_params['use_api'] = True
        print(f"Using API mode for dataset")
    
    dataset_class = dataset_classes.get(dataset_name)
    if not dataset_class:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    
    return dataset_class(**dataset_params), test_name
    
# data/class_names.txt 載入分類標籤 (為 crops 用)
def get_object_class(class_file_path=None, limit=300):
    """Load class names from specified file or default"""
    if not class_file_path:
        class_file_path = 'data/class_names.txt'
    
    print(f"Reading classes from: {class_file_path}")
    with open(class_file_path, 'r', encoding='utf-8') as f:
        classes = [line.strip() for line in f if line.strip()]
    return classes[:limit] if limit else classes


def get_object_class_caption(class_file_path='data/class_names.txt', limit=300):
    """
    Loads object class captions from a text file.

    Args:
        class_file_path (str): Path to the text file containing class names, with each class name on a separate line.
        limit (int, optional): The maximum number of classes to load. If None, loads all classes.

    Returns:
        List[str]: A list of class captions.
    """
    classes = []
    with open(class_file_path, 'r', encoding='utf-8') as file:
        classes = [f'a photo of {line.strip().lower()}' for line in file][:limit]
    return classes

# crops 用的動態設定
def get_object_class_config(class_name, encoder_type='cbm-encoder', crops_dir='encodings/crops', train_split='train', eval_split='test'):
    """
    Return a config dict for a given class_name.
    - For cbm-encoder: use encodings/crops/<class_name>/... paths (image crops)
    - For tbm-encoder: use original JSONL text data, NOT crops
    
    Args:
        class_name: Object class name (e.g., 'aircraft')
        encoder_type: 'cbm-encoder' or 'tbm-encoder'
        crops_dir: Base directory for crops encodings (default: 'encodings/crops')
        train_split: Training split name (default: 'train', can be 'train_noise')
        eval_split: Evaluation split name (default: 'test', can be 'test_janus')
    """
    # CBM: per-class crops-based config using custom splits
    config = {
        'train_dataset': {
            'name': 'img-or-text',
            'params': {
                'real_pt': f"{crops_dir}/{class_name}/{train_split}/real.pt",
                'fake_pt': f"{crops_dir}/{class_name}/{train_split}/fake.pt",
            }
        },
        'eval_dataset': {
            'name': 'img-or-text',
            'params': {
                'real_pt': f"{crops_dir}/{class_name}/{eval_split}/real.pt",
                'fake_pt': f"{crops_dir}/{class_name}/{eval_split}/fake.pt",
            }
        }
    }
    return config

def get_preprocessed_image(pixel_values):
    pixel_values = pixel_values.squeeze().numpy()
    unnormalized_image = (pixel_values * np.array(OPENAI_CLIP_STD)[:, None, None]) + np.array(OPENAI_CLIP_MEAN)[:, None, None]
    unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
    unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
    unnormalized_image = Image.fromarray(unnormalized_image)
    return unnormalized_image