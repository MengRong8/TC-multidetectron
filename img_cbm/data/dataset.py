import torch
import json
import os

class MiRAGeImageOrTextDataset(torch.utils.data.Dataset):
    def __init__(self, real_pt, fake_pt, mode='image'):
        """
        Args:
            real_pt: 真實資料的 .pt 檔案路徑
            fake_pt: 假資料的 .pt 檔案路徑
            mode: 'image' 或 'text'
        """
        self.mode = mode
        
        print(f" Loading dataset:")
        print(f"   Real: {real_pt}")
        print(f"   Fake: {fake_pt}")
        
        if not os.path.exists(real_pt):
            raise FileNotFoundError(f"Real data file not found: {real_pt}")
        if not os.path.exists(fake_pt):
            raise FileNotFoundError(f"Fake data file not found: {fake_pt}")
        
        # 載入資料
        real_data = torch.load(real_pt, weights_only=False)
        fake_data = torch.load(fake_pt, weights_only=False)

        if isinstance(real_data, dict):
            real_features = real_data['features']
            real_labels = real_data['labels']
            print(f"   Real format: dict with keys {real_data.keys()}")
        else:
            real_features = real_data
            real_labels = torch.zeros(len(real_data))  # Real = 0 (negative class)
            print(f"   Real format: tensor")
        
        if isinstance(fake_data, dict):
            fake_features = fake_data['features']
            fake_labels = fake_data['labels']
            print(f"   Fake format: dict with keys {fake_data.keys()}")
        else:
            fake_features = fake_data
            fake_labels = torch.ones(len(fake_data))
            print(f"   Fake format: tensor")
        
        # 合併資料
        self.all_data = torch.cat([real_features, fake_features], dim=0)
        self.all_labels = torch.cat([real_labels, fake_labels], dim=0)
        self.labels = self.all_labels  # Alias for balanced sampler compatibility
        
        print(f" Loaded {len(self.all_data)} samples")
        print(f"   Real: {len(real_labels)}, Fake: {len(fake_labels)}")
        print(f"   Feature shape: {self.all_data.shape}")
        print(f"   Label shape: {self.all_labels.shape}")
    
    def __len__(self):
        return len(self.all_data)
    
    def __getitem__(self, idx):
        features = self.all_data[idx]
        
        if features.dtype != torch.float32:
            features = features.to(torch.float32)
        
        return {
            'features': features,
            'label': self.all_labels[idx].item()
        }
    def _load_meta(self, meta_path):
        if meta_path and meta_path.endswith(".jsonl"):
            with open(meta_path, "r", encoding="utf-8") as f:
                return [json.loads(line) for line in f]
        return None    


class MiRAGeNewsDataset(torch.utils.data.Dataset):
    """
    Multimodal dataset that loads both image and text features
    Used for final MiRAGe multimodal model
    """
    def __init__(self, real_img_pt, fake_img_pt, real_text_pt, fake_text_pt):
        print(f" Loading multimodal dataset:")
        print(f"   Image Real: {real_img_pt}")
        print(f"   Image Fake: {fake_img_pt}")
        print(f"   Text Real: {real_text_pt}")
        print(f"   Text Fake: {fake_text_pt}")
        
        # Load image features
        real_img = torch.load(real_img_pt, weights_only=False)
        fake_img = torch.load(fake_img_pt, weights_only=False)
        
        # Load text features
        real_text = torch.load(real_text_pt, weights_only=False)
        fake_text = torch.load(fake_text_pt, weights_only=False)
        
        #  處理不同的資料格式（list of dicts → tensor）
        if isinstance(real_img, list) and isinstance(real_img[0], dict):
            real_img = torch.stack([item['features'] for item in real_img])
            fake_img = torch.stack([item['features'] for item in fake_img])
        
        if isinstance(real_text, list) and isinstance(real_text[0], dict):
            real_text = torch.stack([item['features'] for item in real_text])
            fake_text = torch.stack([item['features'] for item in fake_text])
        
        print(f"    Real samples: {len(real_img)}")
        print(f"    Fake samples: {len(fake_img)}")
        
        # Concatenate real and fake
        self.all_imgs = torch.cat((real_img, fake_img), dim=0)
        self.all_texts = torch.cat((real_text, fake_text), dim=0)
        self.labels = torch.cat((
            torch.zeros(len(real_img)),   # Real = 0
            torch.ones(len(fake_img))      # Fake = 1
        ))
        
        print(f" Total samples: {len(self.all_imgs)}")
        print(f"   Image feature shape: {self.all_imgs[0].shape}")
        print(f"   Text feature shape: {self.all_texts[0].shape}")
        print(f"Class distribution: Real={len(real_img)}, Fake={len(fake_img)}")
    
    def __len__(self):
        return len(self.all_imgs)
    
    def __getitem__(self, index):
        image_features = self.all_imgs[index]
        text_features = self.all_texts[index]
        label = self.labels[index].item()
        return {
            'image_features': image_features,
            'text_features': text_features,
            'label': label
        }


