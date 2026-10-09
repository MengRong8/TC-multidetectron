import argparse
import torch
import os
from PIL import Image
from tqdm import tqdm
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from datasets import load_dataset

def preprocess_image(image_path, processor):
    image = processor(Image.open(image_path).convert("RGB")).data['pixel_values'][0]
    return torch.Tensor(image)

def preprocess_hf_image(hf_image, processor):
    image = processor(hf_image.convert("RGB")).data['pixel_values'][0]
    return torch.Tensor(image)

def process_batch(images, model, device):
    images_tensor = torch.stack(images).to(device)
    with torch.no_grad():
        return model(images_tensor).pooler_output

def process_directory(directory, model, processor, device, batch_size=32):
    encodings = []
    batch = []
    image_files = sorted(os.listdir(directory))
    print(f"Found {len(image_files)} images in {directory}")

    for idx, filename in enumerate(tqdm(image_files, desc=f"Processing {os.path.basename(directory)}")):
        image_path = os.path.join(directory, filename)
        try:
            batch.append(preprocess_image(image_path, processor))
        except Exception as e:
            print(f"Skipping {image_path}: {e}")
            continue
        if len(batch) == batch_size:
            encodings.append(process_batch(batch, model, device))
            print(f"Processed batch {idx // batch_size + 1} ({len(batch)} images)")
            batch = []

    if batch:
        encodings.append(process_batch(batch, model, device))
        print(f"Processed final batch ({len(batch)} images)")
    if encodings:
        return torch.cat(encodings)
    else:
        # 若沒有成功讀入任何圖片，返回空 tensor
        print(f"No valid images found in {directory}")
        return torch.empty(0, model.config.hidden_size, device=device)
    
    return torch.cat(encodings)

def process_hf_dataset_by_label(dataset, model, processor, device, batch_size=32):
    encodings_real, encodings_fake = [], []
    batch_real, batch_fake = [], []
    print(f"Processing Hugging Face dataset with {len(dataset)} items")

    for idx, item in enumerate(tqdm(dataset, desc="Processing HF dataset")):
        image_tensor = preprocess_hf_image(item["image"], processor)
        label = item["label"]

        if label == 0:
            batch_real.append(image_tensor)
            if len(batch_real) == batch_size:
                encodings_real.append(process_batch(batch_real, model, device))
                batch_real = []
        elif label == 1:
            batch_fake.append(image_tensor)
            if len(batch_fake) == batch_size:
                encodings_fake.append(process_batch(batch_fake, model, device))
                batch_fake = []

    if batch_real:
        encodings_real.append(process_batch(batch_real, model, device))
    if batch_fake:
        encodings_fake.append(process_batch(batch_fake, model, device))

    return torch.cat(encodings_real), torch.cat(encodings_fake)

def save_encodings(encodings, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(encodings, filename)
    print(f"Saved {encodings.shape[0]} feature vectors to {filename}")

def main(custom=False, read_dirs=None, batch_size=64, device="cuda"):
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
    try:
        model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl").vision_model.to(device)
    except RuntimeError as e:
        if "out of memory" in str(e).lower() and device == "cuda":
            print("CUDA OOM, falling back to CPU")
            torch.cuda.empty_cache()
            device = "cpu"
            model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl").vision_model.to(device)
        else:
            raise

    if custom:
        # python encode_image.py --custom --read_dirs test
        # 現在若未指定 --read_dirs，預設使用 test_
        if not read_dirs:
            read_dirs = ["test"]

        for read_dir in read_dirs:
            real_dir = os.path.join("data/split_datasets", read_dir, "real")
            fake_dir = os.path.join("data/split_datasets", read_dir, "fake")

            if os.path.exists(real_dir):
                print(f"\nEncoding real images in {real_dir} ...")
                output_file_real = f"encodings/image/{read_dir}/real.pt"
                image_features_real = process_directory(real_dir, model, processor, device, batch_size)
                save_encodings(image_features_real, output_file_real)

            if os.path.exists(fake_dir):
                print(f"\nEncoding fake images in {fake_dir} ...")
                output_file_fake = f"encodings/image/{read_dir}/fake.pt"
                image_features_fake = process_directory(fake_dir, model, processor, device, batch_size)
                save_encodings(image_features_fake, output_file_fake)

    else:
        dataset_name = "anson-huang/mirage-news"
        available_splits = load_dataset(dataset_name).keys()
        for split in available_splits:
            print(f"\nLoading HF dataset split: {split}")
            dataset = load_dataset(dataset_name, split=split)
            output_file_real = f"encodings/image/{split}/real.pt"
            output_file_fake = f"encodings/image/{split}/fake.pt"

            image_features_real, image_features_fake = process_hf_dataset_by_label(dataset, model, processor, device, batch_size)
            save_encodings(image_features_real, output_file_real)
            save_encodings(image_features_fake, output_file_fake)

    print("\nAll feature vectors saved successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode images from local directory or HF dataset")
    parser.add_argument("--custom", action="store_true", help="Use local directories instead of Hugging Face dataset")
    parser.add_argument("--read_dirs", nargs="+", help="List of directories to read images from (if --custom is set)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for processing images (default: 64)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to run encoding on")

    args = parser.parse_args()
    print(f"\nArguments: {args}")
    main(custom=args.custom, read_dirs=args.read_dirs, batch_size=args.batch_size, device=args.device)
