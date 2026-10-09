import os
import argparse
import json
from models import get_model
from data import load_config, get_dataset
from torch.utils.data import DataLoader
from utils import load_model_checkpoint, test_model, test_multimodal_model, save_metrics



def main(mode, model_class, checkpoint=None, config_path=None):

    if mode == "multimodal":
        # Load configuration
        config_img = load_config(f"configs/image/mirage.yaml")
        config_txt = load_config(f"configs/text/mirage.yaml")
        config = load_config(f"configs/multimodal/mirage.yaml")
        # Initialize models 載入訓練好模型的 checkpoint
        mirage_img = get_model(config_img).to("cuda")
        mirage_txt = get_model(config_txt).to("cuda")
        checkpoint_path = config['training']['image_model_path']
        mirage_img, best_threshold = load_model_checkpoint(mirage_img, checkpoint_path, "cuda")
        print(f"Loaded image model checkpoint from {checkpoint_path} with threshold: {best_threshold}")
        checkpoint_path = config['training']['text_model_path']
        mirage_txt, best_threshold = load_model_checkpoint(mirage_txt, checkpoint_path, "cuda")
        print(f"Loaded text model checkpoint from {checkpoint_path} with threshold: {best_threshold}")

        # Load all test datasets
        test_datasets = get_dataset(config, test_set='all')

        # Set path for saving metrics 
        metrics_save_path = f"results/multimodal/mirage.jsonl"
        # metrics_save_path = f"results/text/linear_test_chinese.jsonl"
        # Evaluate model on each test set and save metrics with test_name
        for dataset, test_name in test_datasets:
            test_loader = DataLoader(dataset, batch_size=config['testing']['batch_size'])
            
            #=========================================================
            # Run evaluation and save metrics using test_model
            #test_metrics : accuracy、precision、recall、F1
            #predictions : label, score
            test_metrics, predictions = test_multimodal_model(mirage_img, mirage_txt, test_loader, device="cuda")
            #=========================================================
            
            metrics = {"test_name": test_name}
            metrics.update(test_metrics)
            save_metrics(metrics, metrics_save_path)
            
            # === 新增：存逐筆結果 ===
            #mode : text,image,multimodal
            preds_save_path = f"results/{mode}/{test_name}_predictions.jsonl"
            with open(preds_save_path, "w") as f:
                for p in predictions:
                    f.write(json.dumps(p) + "\n")
                    
    # mode : text or image
    else:
        # Load configuration
        #model_class : linear, cbm-predictor, mirage
        if config_path:
            config = load_config(config_path)
        else:
            config = load_config(f"configs/{mode}/{model_class}.yaml")
        # Initialize model
        model = get_model(config).to("cuda")
        # print(f"Loaded model: {model}")
        # Load the model checkpoint and best threshold
        checkpoint_path = f"checkpoints/{mode}/{checkpoint}.pt" if checkpoint else config['training']['save_path']
    
        checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]

        model, best_threshold = load_model_checkpoint(model, checkpoint_path, "cuda")
        print(f"Loaded model checkpoint from {checkpoint_path} with threshold: {best_threshold}")
            
        # Load all test datasets
        test_datasets = get_dataset(config, test_set='all')

        # Set path for saving metrics 
        metrics_save_path = f"results/{mode}/{checkpoint_name}.jsonl"

        # Evaluate model on each test set and save metrics with test_name
        for dataset, test_name in test_datasets:
            test_loader = DataLoader(dataset, batch_size=config['testing']['batch_size'])
            
            # Run evaluation and save metrics using test_model
            test_metrics, predictions = test_model(model, test_loader, checkpoint_path, device="cuda")

            metrics = {"test_name": test_name}
            metrics.update(test_metrics)
            save_metrics(metrics, metrics_save_path)
            
            # === 新增：存逐筆結果 ===
            preds_save_path = f"results/{mode}/{checkpoint_name}_{test_name}_predictions.jsonl"
            with open(preds_save_path, "w") as f:
                for p in predictions:
                    f.write(json.dumps(p) + "\n")

    print(f"Testing results saved.")

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Evaluate a trained model on a test dataset")
    parser.add_argument(
        '--mode',
        type=str,
        choices=['image', 'text', 'multimodal'],
        required=True,
        help="Specify model mode: 'image', 'text', or 'multimodal'"
    )
    parser.add_argument(
        '--model_class',
        type=str,
        required=True,
        help="Choose from 'linear', 'cbm-encoder', 'cbm-predictor', 'mirage', or provide a custom model class"
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=False,
        help="Path to a specific model checkpoint file. Overrides config file path if provided."
    )
    parser.add_argument(
        '--config',
        type=str,
        required=False,
        help="Path to a specific config file. Overrides default config path if provided."
    )

    args = parser.parse_args()
    main(args.mode, args.model_class, args.checkpoint, args.config)
