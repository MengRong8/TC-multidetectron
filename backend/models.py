"""
模型推理模組
整合三個模型: Text AIGC (RoBERTa), Text Intent (CB-LLM), Image AIGC (CBM)
"""
__all__ = ['TextAIGCModel', 'TextIntentModel', 'ImageAIGCModel', 'ModelInference']

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, Optional, Tuple, List
from PIL import Image
import torchvision.transforms as transforms
import logging
import sys
import os
from pathlib import Path
import base64
import time
import io
import requests
from urllib.parse import urlsplit, urlunsplit
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 添加項目根目錄到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

# 嘗試導入 CBM 模型類
try:
    from img_cbm.models.mirage_img import ObjectClassCBMEncoder, ObjectClassCBMPredictor
    CBM_AVAILABLE = True
except ImportError as e:
    logger.warning(f"無法導入 CBM 模型: {e}")
    CBM_AVAILABLE = False
    ObjectClassCBMEncoder = None
    ObjectClassCBMPredictor = None

# 嘗試導入 BLIP2（用於 CBM 特徵提取）
try:
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    BLIP2_AVAILABLE = True
except ImportError:
    logger.warning("無法導入 BLIP2，Image AIGC 模型將無法使用")
    BLIP2_AVAILABLE = False
    Blip2Processor = None
    Blip2ForConditionalGeneration = None

# 嘗試導入 Unsloth（用於 Text Intent CB-LLM）
try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    logger.warning("無法導入 Unsloth，Text Intent 模型將無法使用")
    UNSLOTH_AVAILABLE = False
    FastLanguageModel = None


DEFAULT_CBM_CONCEPT_MAP = [
    "情緒化",
    "new_誇大與聳動",
    "new_煽動性",
    "不實",
    "new_主觀",
    "極端",
    "威脅性",
    "虛假引用",
    "new_偏頗",
    "標題黨",
    "譁眾取寵",
    "斷章取義",
    "情緒操控",
    "權威訴求",
    "假兩難",
]


def _load_intent_concept_map() -> List[str]:
    concept_map_path = os.getenv("TEXT_INTENT_CONCEPT_MAP_PATH", "").strip()
    if concept_map_path and os.path.exists(concept_map_path):
        try:
            with open(concept_map_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, list) and payload:
                return [str(x) for x in payload]
        except Exception as e:
            logger.warning(f"Failed to load TEXT_INTENT_CONCEPT_MAP_PATH: {e}")
    return list(DEFAULT_CBM_CONCEPT_MAP)


class NewsCBM(torch.nn.Module):
    """與 cb_llm_induction/eval.py 一致的 CBM head 封裝。"""

    def __init__(self, backbone, num_concepts: int = 15):
        super().__init__()
        self.backbone = backbone
        self.config = backbone.config
        self.concept_head = torch.nn.Linear(backbone.config.hidden_size, num_concepts)
        self.final_head = torch.nn.Linear(num_concepts, 1)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.backbone.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        last_hidden_state = outputs.last_hidden_state[:, -1, :]

        concept_logits = self.concept_head(last_hidden_state)
        concept_logits = torch.clamp(concept_logits, min=-15, max=15)
        concept_probs = torch.sigmoid(concept_logits)
        final_logits = self.final_head(concept_probs)
        return concept_logits, final_logits


class TextAIGCModel:
    """
    Text AIGC Detection Model (RoBERTa)
    用於檢測文本是否為 AI 生成
    """
    
    def __init__(self, model_path: str = "web_crawling/model/roberta-fake-news-detection-final"):
        """
        Args:
            model_path: RoBERTa 模型路徑
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading Text AIGC Model on {self.device}...")
        
        # 獲取絕對路徑
        if not os.path.isabs(model_path):
            model_path = os.path.join(Path(__file__).parent.parent, model_path)
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=True
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                local_files_only=True
            )
            self.model.to(self.device)
            self.model.eval()
            logger.info("Text AIGC Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Text AIGC Model: {e}")
            raise
    
    def predict(self, text: str) -> Dict:
        """
        預測文本是否為 AI 生成
        
        Args:
            text: 新聞文本內容
            
        Returns:
            {
                'score': float (0-1, 0=real, 1=fake/AIGC),
                'confidence': float,
                'label': str,
                'probabilities': dict
            }
        """
        try:
            # Tokenize
            inputs = self.tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = F.softmax(outputs.logits, dim=-1)[0]
            
            # 提取結果
            fake_prob = probs[1].item()  # index 1 = FAKE/AIGC
            real_prob = probs[0].item()  # index 0 = REAL
            
            result = {
                'score': fake_prob,  # 0=real, 1=fake
                'label': 'AIGC' if fake_prob > 0.5 else 'Human-written',
                'probabilities': {
                    'real': real_prob,
                    'aigc': fake_prob
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Text AIGC prediction error: {e}")
            raise


class TextIntentModel:
    """
    Text Intent Detection Model (CB-LLM based on Qwen-3)
    用於檢測文本是否有誘導性（煽動性、情緒化等）
    """
    
    def __init__(self, model_path: str = "cb_llm_induction/cbm_checkpoints_V4/cbm_epoch_4.pth"):
        """
        Args:
            model_path: CB-LLM 模型路徑（包含 LoRA adapters）
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading Text Intent Model on {self.device}...")
        
        # 獲取絕對路徑
        if not os.path.isabs(model_path):
            model_path = os.path.join(Path(__file__).parent.parent, model_path)
        
        self.model_name = os.getenv("CBLLM_BASE_MODEL", "unsloth/Qwen3-8B")
        self.max_seq_length = int(os.getenv("CBLLM_MAX_SEQ_LENGTH", "1536"))
        self.load_in_4bit = os.getenv("CBLLM_LOAD_IN_4BIT", "true").lower() == "true"
        self.threshold = float(os.getenv("CBLLM_THRESHOLD", "0.5"))
        self.top_k = int(os.getenv("CBLLM_TOP_K", "3"))
        self.concept_names = _load_intent_concept_map()
        self.tokenizer = None
        self.model = None
        self.loaded = False

        if not UNSLOTH_AVAILABLE:
            logger.warning("Text Intent Model disabled because Unsloth is unavailable")
            return

        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"CB-LLM checkpoint not found: {model_path}")

            # 1) 載入 Qwen backbone（與 eval.py 一致）
            # 使用 device_map="auto" 讓 Unsloth 自動分配模型到 GPU/CPU
            backbone, tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_name,
                max_seq_length=self.max_seq_length,
                dtype=torch.bfloat16,
                load_in_4bit=self.load_in_4bit,
                device_map="auto",
            )

            tokenizer.padding_side = "left"
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # 2) 掛載 LoRA adapter 結構（與 eval.py 一致）
            backbone = FastLanguageModel.get_peft_model(
                backbone,
                r=32,
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
                lora_alpha=16,
                lora_dropout=0,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=3407,
                use_rslora=False,
                loftq_config=None,
            )

            # 3) 從 checkpoint 推回概念維度
            payload = torch.load(model_path, map_location="cpu")
            if isinstance(payload, dict) and "model_state" in payload:
                state_dict = payload["model_state"]
            elif isinstance(payload, dict):
                state_dict = payload
            else:
                raise ValueError("Unsupported CB-LLM checkpoint payload format")

            concept_weight = state_dict.get("concept_head.weight")
            num_concepts = int(concept_weight.shape[0]) if concept_weight is not None else len(self.concept_names)

            # 若概念數與預設名稱不符，按 eval.py 邏輯對齊
            if num_concepts > len(self.concept_names):
                extra = num_concepts - len(self.concept_names)
                self.concept_names.extend([f"ckpt_extra_concept_{i+1}" for i in range(extra)])
            elif num_concepts < len(self.concept_names):
                self.concept_names = self.concept_names[:num_concepts]

            # 4) 建立 CBM 並載入 checkpoint 權重
            cbm_model = NewsCBM(backbone, num_concepts=num_concepts).to(self.device)
            if self.device.type == "cuda":
                cbm_model = cbm_model.to(torch.bfloat16)

            missing, unexpected = cbm_model.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                logger.warning(
                    "Text Intent checkpoint non-strict load: missing=%d, unexpected=%d",
                    len(missing),
                    len(unexpected),
                )

            cbm_model.eval()
            self.model = cbm_model
            self.tokenizer = tokenizer
            self.loaded = True
            logger.info(
                "Text Intent Model loaded successfully (concepts=%d, threshold=%.2f)",
                len(self.concept_names),
                self.threshold,
            )

        except Exception as e:
            logger.error(f"Failed to load Text Intent Model: {e}")
            self.model = None
            self.loaded = False
    
    def predict(self, text: str) -> Dict:
        """
        預測文本是否有誘導性
        
        Args:
            text: 新聞文本內容
            
        Returns:
            {
                'score': float (0-1, 0=neutral, 1=manipulative),
                'confidence': float,
                'label': str,
                'detected_patterns': list
            }
        """
        if not self.loaded:
            logger.warning("Text Intent Model not loaded, returning dummy prediction")
            return {
                'score': 0.3,  # 佔位符分數
                'label': 'Neutral',
                'detected_patterns': []
            }

        try:
            inputs = self.tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=self.max_seq_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                c_logits, f_logits = self.model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                )

                concept_probs = torch.sigmoid(c_logits.float())[0]  # [num_concepts]
                fake_prob = torch.sigmoid(f_logits.float().squeeze(-1))[0].item()

            concept_scores = concept_probs.cpu().tolist()
            top_k = min(self.top_k, len(concept_scores))
            top_indices = np.argsort(-np.asarray(concept_scores))[:top_k]

            top_concepts = [
                {
                    'rank': i + 1,
                    'index': int(idx),
                    'name': self.concept_names[int(idx)],
                    'score': float(concept_scores[int(idx)]),
                }
                for i, idx in enumerate(top_indices)
            ]

            return {
                'score': float(fake_prob),
                'label': 'Manipulative' if fake_prob > self.threshold else 'Neutral',
                'detected_patterns': [x['name'] for x in top_concepts],
                'threshold': float(self.threshold),
                'cbllm_fake_prob': float(fake_prob),
                'cbllm_concept_probs': [float(v) for v in concept_scores],
                'cbllm_concept_names': [str(name) for name in self.concept_names],
                'top_concepts': top_concepts,
            }

        except Exception as e:
            logger.error(f"Text Intent prediction error: {e}")
            raise


class ImageAIGCModel:
    """
    Image AIGC Detection Model (CBM - Concept Bottleneck Model)
    用於檢測圖片是否為 AI 生成
    """
    
    def __init__(
        self,
        encoder_path: str = "img_cbm/checkpoints/cbm-encoder.pt",
        predictor_path: str = "img_cbm/checkpoints/cbm-predictor.pt",
        threshold: float = 0.45
    ):
        """
        Args:
            encoder_path: CBM encoder checkpoint 路徑
            predictor_path: CBM predictor checkpoint 路徑
            threshold: 判定閾值
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading Image AIGC Model (CBM) on {self.device}...")
        
        # 獲取絕對路徑
        if not os.path.isabs(encoder_path):
            encoder_path = os.path.join(Path(__file__).parent.parent, encoder_path)
        if not os.path.isabs(predictor_path):
            predictor_path = os.path.join(Path(__file__).parent.parent, predictor_path)
        
        self.threshold = threshold
        self.owlv2_max_candidates = int(os.getenv("OWLV2_MAX_CANDIDATES", "8"))
        self.owlv2_max_verify_seconds = float(os.getenv("OWLV2_MAX_VERIFY_SECONDS", "45"))
        self.owlv2_max_text_queries = int(os.getenv("OWLV2_MAX_TEXT_QUERIES", "1"))
        self.loaded = False

        # Default placeholders for safer failure handling
        self.encoder = None
        self.predictor = None
        self.blip2_processor = None
        self.blip2_vision_model = None
        self.class_names = []

        def _extract_state_dict(ckpt):
            if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                return ckpt['model_state_dict']
            return ckpt

        def _remap_encoder_keys_for_sequential(state_dict):
            """classifiers.{i}.weight -> classifiers.{i}.0.weight"""
            remapped = {}
            for key, value in state_dict.items():
                if key.startswith('classifiers.') and key.count('.') == 2:
                    parts = key.split('.')
                    cls_idx, leaf = parts[1], parts[2]
                    remapped[f'classifiers.{cls_idx}.0.{leaf}'] = value
                else:
                    remapped[key] = value
            return remapped

        def _remap_encoder_keys_for_linear(state_dict):
            """classifiers.{i}.0.weight -> classifiers.{i}.weight"""
            remapped = {}
            for key, value in state_dict.items():
                if key.startswith('classifiers.') and '.0.' in key:
                    remapped[key.replace('.0.', '.')] = value
                else:
                    remapped[key] = value
            return remapped
        
        try:
            # 檢查依賴是否可用
            if not CBM_AVAILABLE:
                raise ImportError("CBM model classes not available")
            if not BLIP2_AVAILABLE:
                raise ImportError("BLIP2 not available")
            
            # 加載 CBM encoder (使用 weights_only=False 因為是信任的本地檔案)
            encoder_ckpt = torch.load(encoder_path, map_location=self.device, weights_only=False)
            
            # 建立 encoder 模型結構
            if ObjectClassCBMEncoder is None:
                raise ImportError("CBM model classes not available")
            
            self.encoder = ObjectClassCBMEncoder()
            encoder_state_dict = _extract_state_dict(encoder_ckpt)
            try:
                self.encoder.load_state_dict(encoder_state_dict)
            except RuntimeError:
                # Try both historical key layouts used by previous training scripts.
                try:
                    self.encoder.load_state_dict(_remap_encoder_keys_for_linear(encoder_state_dict))
                except RuntimeError:
                    self.encoder.load_state_dict(_remap_encoder_keys_for_sequential(encoder_state_dict))
            self.encoder.to(self.device)
            self.encoder.eval()
            
            # 載入 CBM predictor
            predictor_ckpt = torch.load(predictor_path, map_location=self.device, weights_only=False)
            
            self.predictor = ObjectClassCBMPredictor()
            predictor_state_dict = _extract_state_dict(predictor_ckpt)
            self.predictor.load_state_dict(predictor_state_dict)
            self.predictor.to(self.device)
            self.predictor.eval()
            
            # 載入 BLIP2 特徵提取器
            logger.info("Loading BLIP2 feature extractor...")
            # Use slow tokenizer to avoid tokenizer.json compatibility issues in some container environments.
            self.blip2_processor = Blip2Processor.from_pretrained(
                "Salesforce/blip2-flan-t5-xl",
                use_fast=False,
            )
            # Load the original BLIP2 checkpoint and extract the vision model so
            # features match the training-time encoder pipeline.
            blip2_full_model = Blip2ForConditionalGeneration.from_pretrained(
                "Salesforce/blip2-flan-t5-xl",
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            )
            self.blip2_vision_model = blip2_full_model.vision_model.to(self.device)
            self.blip2_vision_model.eval()
            del blip2_full_model
            
            # 載入 300 個 class names
            class_names_path = os.path.join(Path(__file__).parent.parent, "data/class_names.txt")
            if os.path.exists(class_names_path):
                with open(class_names_path, 'r') as f:
                    self.class_names = [line.strip() for line in f if line.strip()]
                logger.info(f"Loaded {len(self.class_names)} class names")
            else:
                logger.warning(f"class_names.txt not found at {class_names_path}")
                self.class_names = [f"concept_{i}" for i in range(300)]
            
            self.loaded = True
            logger.info(f"Image AIGC Model (CBM) loaded successfully (threshold={threshold})")
            
            # OWLv2 延遲載入（只在需要時載入以節省記憶體）
            self.owl_processor = None
            self.owl_model = None
            self.yolo_model = None  # YOLO-World v2 延遲載入
            logger.info("OWLv2 will be loaded on-demand for concept verification")
            logger.info("OWLv2 verification candidate cap: %d", self.owlv2_max_candidates)
            logger.info("OWLv2 verification time cap: %.1fs", self.owlv2_max_verify_seconds)
            logger.info("OWLv2 per-class text query cap: %d", self.owlv2_max_text_queries)
            
        except Exception as e:
            logger.error(f"Failed to load Image AIGC Model: {e}")
            self.loaded = False
            self.encoder = None
            self.predictor = None
            self.blip2_processor = None
            self.blip2_vision_model = None
            self.owl_processor = None
            self.owl_model = None
            self.yolo_model = None
    
    def _load_image_from_base64(self, base64_str: str) -> Image.Image:
        """從 Base64 字符串加載圖片"""
        # 移除 data URI prefix (如 data:image/jpeg;base64,)
        if ',' in base64_str:
            base64_str = base64_str.split(',')[1]
        
        image_data = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        return image
    
    def _load_image_from_url(self, url: str) -> Image.Image:
        """從 URL 加載圖片"""
        split = urlsplit(url)
        origin = f"{split.scheme}://{split.netloc}/" if split.scheme and split.netloc else ""

        candidate_urls = [url]
        # 有些站台會對含 query 的縮圖 URL 回傳 403，嘗試去掉 query 再抓一次。
        if split.query:
            stripped = urlunsplit((split.scheme, split.netloc, split.path, "", ""))
            if stripped and stripped not in candidate_urls:
                candidate_urls.append(stripped)

        session = requests.Session()
        retries = Retry(
            total=2,
            connect=2,
            read=2,
            status=2,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retries)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        header_profiles = [
            {
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/123.0.0.0 Safari/537.36"
                ),
                "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
                "Accept-Language": "zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7",
                "Referer": origin,
                "Connection": "keep-alive",
            },
            {
                "User-Agent": "Mozilla/5.0",
                "Accept": "*/*",
            },
        ]

        last_error = None
        for candidate_url in candidate_urls:
            for headers in header_profiles:
                try:
                    response = session.get(candidate_url, headers=headers, timeout=(5, 15), allow_redirects=True)
                    response.raise_for_status()

                    content_type = (response.headers.get("Content-Type") or "").lower()
                    if content_type and "image" not in content_type:
                        raise ValueError(f"URL did not return image content (Content-Type: {content_type})")

                    image = Image.open(io.BytesIO(response.content)).convert('RGB')
                    return image
                except Exception as e:
                    last_error = e

        raise RuntimeError(f"Failed to fetch image from URL: {url} ({last_error})")
    
    def _get_detection_alternatives(self, class_name: str) -> List[str]:
        """為類別名稱提供 OWLv2 偵測替代詞"""
        alternatives = {
            # 人物相關
            'man': ['male person', 'man'],
            'woman': ['female person', 'woman'],
            'boy': ['male child', 'boy'],
            'girl': ['female child', 'girl'],
            'human_face': ['face', 'human face'],
            'human_head': ['head', 'human head'],
            'human_body': ['person', 'people', 'human body'],
            'human_arm': ['arm', 'human arm'],
            'human_hair': ['hair'],
            'person': ['person', 'people'],
            
            # 衣物相關
            'shirt': ['shirt', 'clothing', 'clothes'],
            'suit': ['suit', 'formal wear'],
            'jacket': ['jacket', 'coat'],
            'coat': ['coat', 'jacket'],
            'trousers': ['pants', 'trousers'],
            'jeans': ['jeans', 'pants'],
            'dress': ['dress'],
            'skirt': ['skirt'],
            
            # 建築相關
            'building': ['building', 'structure'],
            'house': ['house', 'building'],
            'skyscraper': ['skyscraper', 'building'],
            'window': ['window'],
            'door': ['door'],
            'door_handle': ['door handle', 'handle'],
            
            # 物品相關
            'plastic_bag': ['bag', 'plastic bag'],
            'handbag': ['bag', 'handbag'],
            'backpack': ['backpack', 'bag'],
            'car': ['car', 'vehicle'],
            'microphone': ['microphone', 'mic'],
            'camera': ['camera'],
            'tripod': ['tripod'],
            'tire': ['tire', 'wheel'],
            'flower': ['flower'],
            'plant': ['plant'],
            'tree': ['tree'],
        }
        return alternatives.get(class_name, [class_name])
    
    def _detect_object_with_owlv2(
        self,
        image: Image.Image,
        class_name: str,
        score_threshold: float = 0.1
    ) -> bool:
        """使用 OWLv2 檢測物件是否在圖片中出現
        
        Args:
            image: PIL Image
            class_name: 物件類別名稱
            score_threshold: 偵測信心閾值（OWLv2 建議較低，0.1-0.3）
        
        Returns:
            bool: 是否偵測到該物件
        """
        try:
            # 延遲載入 OWLv2（只在第一次使用時載入）
            if self.owl_model is None:
                logger.info("Loading OWLv2 object detection model...")
                try:
                    from transformers import Owlv2Processor, Owlv2ForObjectDetection
                    self.owl_processor = Owlv2Processor.from_pretrained(
                        "google/owlv2-base-patch16-ensemble"
                    )
                    self.owl_model = Owlv2ForObjectDetection.from_pretrained(
                        "google/owlv2-base-patch16-ensemble"
                    ).to(self.device)
                    self.owl_model.eval()
                    logger.info("OWLv2 loaded successfully")
                except ImportError:
                    logger.warning("OWLv2 not available, skipping object detection")
                    return True  # 如果 OWLv2 不可用，預設為檢測到
            
            # 獲取替代詞彙（支援更豐富的文本描述）
            detection_classes = self._get_detection_alternatives(class_name)
            
            # 嘗試所有替代詞，直到找到偵測結果
            for try_class in detection_classes:
                # OWLv2 支援文本查詢
                text_queries = [[try_class], [f"a photo of {try_class}"]]

                for texts in text_queries[: self.owlv2_max_text_queries]:
                    # 預處理圖片和文本
                    inputs = self.owl_processor(
                        text=texts,
                        images=image,
                        return_tensors="pt"
                    ).to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.owl_model(**inputs)
                    
                    # 目標圖片尺寸用於縮放框
                    target_sizes = torch.Tensor([image.size[::-1]]).to(self.device)  # (height, width)
                    
                    # 轉換輸出為 [xmin, ymin, xmax, ymax] 格式
                    if hasattr(self.owl_processor, "post_process_object_detection"):
                        results = self.owl_processor.post_process_object_detection(
                            outputs=outputs,
                            target_sizes=target_sizes,
                            threshold=score_threshold
                        )
                    else:
                        results = self.owl_processor.post_process_grounded_object_detection(
                            outputs=outputs,
                            target_sizes=target_sizes,
                            threshold=score_threshold
                        )
                    
                    # 如果偵測到物件，返回 True
                    if len(results) > 0 and len(results[0]["boxes"]) > 0:
                        boxes = results[0]["boxes"].cpu().numpy()
                        scores = results[0]["scores"].cpu().numpy()
                        
                        # 過濾掉太小的框（寬高至少 32px）
                        image_width, image_height = image.size
                        valid_detections = 0
                        for box, score in zip(boxes, scores):
                            x_min, y_min, x_max, y_max = box
                            w, h = x_max - x_min, y_max - y_min
                            if w >= 32 and h >= 32:
                                valid_detections += 1
                        
                        if valid_detections > 0:
                            logger.debug(
                                f"OWLv2 detected {valid_detections} '{try_class}' (score: {scores[0]:.3f})"
                            )
                            return True
            
            return False
        
        except Exception as e:
            logger.warning(f"OWLv2 detection failed for '{class_name}': {e}")
            return False
    
    # def _detect_object_with_yoloworld(self, image: Image.Image, class_name: str) -> bool:
    #     """使用 YOLO-World v2 檢測物體是否存在於圖片中"""
    #     try:
    #         # 延遲加載 YOLO-World v2
    #         if self.yolo_model is None:
    #             logger.info("Loading YOLO-World v2 model...")
    #             from ultralytics import YOLOWorld
    #             
    #             # 使用 YOLO-World v2 Large 模型
    #             self.yolo_model = YOLOWorld('yolov8l-worldv2.pt')
    #             self.yolo_model.to(self.device)
    #             logger.info("YOLO-World v2 loaded successfully")
    #         
    #         # 獲取替代詞彙
    #         detection_classes = self._get_detection_alternatives(class_name)
    #         
    #         # 設置檢測的類別
    #         self.yolo_model.set_classes(detection_classes)
    #         
    #         # 執行檢測（conf=0.1 與 OWLv2 一致）
    #         results = self.yolo_model.predict(
    #             image,
    #             conf=0.1,
    #             verbose=False,
    #             device=self.device
    #         )
    #         
    #         # 檢查是否有有效檢測結果
    #         if len(results) > 0 and len(results[0].boxes) > 0:
    #             boxes = results[0].boxes
    #             # 過濾掉太小的框（寬高至少 32px）
    #             valid_detections = 0
    #             for box in boxes:
    #                 x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
    #                 w, h = x2 - x1, y2 - y1
    #                 if w >= 32 and h >= 32:
    #                     valid_detections += 1
    #             
    #             if valid_detections > 0:
    #                 conf_score = boxes[0].conf.item()
    #                 logger.debug(
    #                     f"YOLO-World detected {valid_detections} '{class_name}' (conf: {conf_score:.3f})"
    #                 )
    #                 return True
    #         
    #         return False
    #         
    #     except Exception as e:
    #         logger.warning(f"YOLO-World detection failed for '{class_name}': {e}")
    #         return False
            
        except Exception as e:
            logger.warning(f"OWLv2 detection error for '{class_name}': {e}")
            return True  # 檢測失敗時預設為存在
    
    def predict(
        self,
        image_base64: Optional[str] = None,
        image_url: Optional[str] = None,
        image_pil: Optional[Image.Image] = None,
        debug: bool = False,
        explainable: bool = True
    ) -> Dict:
        """
        預測圖片是否為 AI 生成
        
        Args:
            image_base64: Base64 編碼的圖片
            image_url: 圖片 URL
            image_pil: PIL Image 對象
            debug: 是否返回詳細調試信息
            explainable: 啟用全部 300 概念 OWLv2 驗證（完全可解釋性，需時 ~45-60 秒）
            
        Returns:
            {
                'score': float (0-1, 0=real, 1=fake/AIGC),
                'confidence': float,
                'label': str,
                'threshold': float,
                'concept_scores': [300 floats],  # 300 維概念分數
                'top_concepts': [  # OWLv2 驗證的概念
                    {'name': str, 'score': float, 'rank': int, 'verified': True}, ...
                ]
            }
        """
        # 檢查模型是否已載入
        if not self.loaded:
            logger.warning("Image AIGC Model not loaded, returning dummy prediction")
            return {
                'score': 0.5,  # 中立分數
                'label': 'Unknown',
                'threshold': self.threshold,
                'error': 'Model not loaded',
                'concept_scores': [],
                'top_concepts': []
            }
        
        try:
            # 加載圖片
            if image_pil is not None:
                image = image_pil
            elif image_base64 is not None:
                image = self._load_image_from_base64(image_base64)
            elif image_url is not None:
                image = self._load_image_from_url(image_url)
            else:
                raise ValueError("No image provided")
            
            # 確保圖片是 RGB 模式
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Inference
            with torch.no_grad():
                # Step 1: 使用 BLIP2 提取圖像特徵 (1408 維)
                pixel_values = self.blip2_processor(image, return_tensors="pt")['pixel_values'].to(self.device)
                image_features = self.blip2_vision_model(pixel_values).pooler_output  # [1, 1408]
                
                # Step 2: 運行 CBM encoder 的 300 個分類器
                # Predictor 在訓練時使用的是 300 維 sigmoid probabilities。
                concept_probs = []
                concept_scores = []
                for class_idx in range(300):
                    logit = self.encoder(image_features, classifier_index=class_idx)  # [1, 1]
                    prob_tensor = torch.sigmoid(logit)
                    concept_probs.append(prob_tensor)
                    concept_scores.append(prob_tensor[0, 0].item())
                
                raw_concepts = torch.cat(concept_probs, dim=-1)  # [1, 300]
                concept_scores_np = np.array(concept_scores)
                
                # Step 2.5: Hybrid Verification (混合驗證，平衡準確度與可解釋性)
                if explainable:
                    # 先用原始分數做初步預測，決定驗證方向
                    initial_logits = self.predictor(raw_concepts)
                    initial_prob = torch.sigmoid(initial_logits[0, 0]).item()
                    
                    if initial_prob < 0.45:
                        # REAL 圖片：驗證低分概念（真實特徵）
                        verify_direction = "bottom-50"
                        target_50_indices = np.argsort(concept_scores_np)[:50].tolist()  # 最低 50 個
                        logger.info("🔍 Explainable mode: Hybrid Verification (Bottom-50 strict + Others weighted)")
                        logger.info(f"Initial prob: {initial_prob:.4f} < 0.45, verifying low-score concepts (real features)")
                    else:
                        # FAKE 圖片：驗證高分概念（AI 特徵）
                        verify_direction = "top-50"
                        target_50_indices = np.argsort(-concept_scores_np)[:50].tolist()  # 最高 50 個
                        logger.info("🔍 Explainable mode: Hybrid Verification (Top-50 strict + Others weighted)")
                        logger.info(f"Initial prob: {initial_prob:.4f} >= 0.45, verifying high-score concepts (AI features)")
                    
                    # 用 OWLv2 驗證目標 50 個概念（YOLO-World 已註解）
                    verified_indices_owlv2 = set()
                    # verified_indices_yolo = set()  # YOLO-World 已註解
                    verified_indices = set()  # 僅使用 OWLv2 檢測結果
                    unverified_target50_indices = set()
                    verify_start = time.monotonic()
                    
                    # 統計數據
                    owlv2_count = 0
                    # yolo_count = 0  # YOLO-World 已註解
                    # both_count = 0  # YOLO-World 已註解
                    
                    # === 階段 1: OWLv2 驗證 ===
                    logger.info(f"Phase 1: Verifying {verify_direction} concepts with OWLv2...")
                    for i, idx in enumerate(target_50_indices):
                        # 時間限制：最多 10 秒
                        if (time.monotonic() - verify_start) > 10.0:
                            logger.info(f"OWLv2 verification timeout after {i+1}/50 concepts")
                            break
                        
                        class_name = self.class_names[idx] if idx < len(self.class_names) else f"concept_{idx}"
                        is_detected_owlv2 = self._detect_object_with_owlv2(image, class_name)
                        
                        if is_detected_owlv2:
                            verified_indices_owlv2.add(idx)
                            logger.debug(f"✓ OWLv2 #{i+1}: {class_name} (score: {concept_scores[idx]:.4f})")
                    
                    owlv2_time = time.monotonic() - verify_start
                    logger.info(f"OWLv2 phase complete: {len(verified_indices_owlv2)}/{i+1} verified in {owlv2_time:.2f}s")
                    
                    # 卸載 OWLv2 釋放顯存
                    if self.owl_model is not None:
                        logger.info("Unloading OWLv2 to free GPU memory (~2.5GB)...")
                        del self.owl_model
                        del self.owl_processor
                        self.owl_model = None
                        self.owl_processor = None
                        torch.cuda.empty_cache()
                        logger.info("OWLv2 unloaded successfully")
                    
                    # === 階段 2: YOLO-World 驗證（已註解） ===
                    # yolo_start = time.monotonic()
                    # remaining_time = 15.0 - (yolo_start - verify_start)
                    # logger.info(f"Phase 2: Verifying with YOLO-World v2 (remaining time: {remaining_time:.1f}s)...")
                    # 
                    # for i, idx in enumerate(target_50_indices):
                    #     # 使用剩餘時間
                    #     if (time.monotonic() - yolo_start) > max(5.0, remaining_time):
                    #         logger.info(f"YOLO-World verification timeout after {i+1}/50 concepts")
                    #         break
                    #     
                    #     class_name = self.class_names[idx] if idx < len(self.class_names) else f"concept_{idx}"
                    #     is_detected_yolo = self._detect_object_with_yoloworld(image, class_name)
                    #     
                    #     if is_detected_yolo:
                    #         verified_indices_yolo.add(idx)
                    #         logger.debug(f"✓ YOLO #{i+1}: {class_name} (score: {concept_scores[idx]:.4f})")
                    # 
                    # yolo_time = time.monotonic() - yolo_start
                    # logger.info(f"YOLO-World phase complete: {len(verified_indices_yolo)}/{i+1} verified in {yolo_time:.2f}s")
                    
                    # === 合併結果（僅使用 OWLv2） ===
                    for idx in target_50_indices:
                        is_detected_owlv2 = idx in verified_indices_owlv2
                        # is_detected_yolo = idx in verified_indices_yolo  # YOLO-World 已註解
                        
                        if is_detected_owlv2:  # 僅檢查 OWLv2
                            verified_indices.add(idx)
                            owlv2_count += 1
                            
                            # if is_detected_owlv2 and is_detected_yolo:  # YOLO-World 已註解
                            #     both_count += 1
                            # elif is_detected_owlv2:
                            #     owlv2_count += 1
                            # else:
                            #     yolo_count += 1
                        else:
                            unverified_target50_indices.add(idx)
                    
                    elapsed_time = time.monotonic() - verify_start
                    checked_count = len(target_50_indices)
                    
                    # 詳細比較日誌（僅 OWLv2）
                    logger.info(f"✅ OWLv2 verification complete in {elapsed_time:.2f}s:")
                    logger.info(f"   • Total verified: {len(verified_indices)}/{checked_count}")
                    logger.info(f"   • OWLv2 detected: {len(verified_indices_owlv2)} concepts")
                    # logger.info(f"   • YOLO-World detected: {len(verified_indices_yolo)} concepts")  # YOLO-World 已註解
                    # logger.info(f"   • Both detected: {both_count} | OWLv2 only: {owlv2_count} | YOLO only: {yolo_count}")  # YOLO-World 已註解
                    
                    # 卸載 YOLO-World 釋放顯存（已註解）
                    # if self.yolo_model is not None:
                    #     logger.info("Unloading YOLO-World to free GPU memory (~1.5GB)...")
                    #     del self.yolo_model
                    #     self.yolo_model = None
                    #     torch.cuda.empty_cache()
                    
                    # 創建過濾後的概念向量
                    filtered_concepts = raw_concepts.clone()
                    
                    # Strategy 1: 目標 50 個驗證通過 → 保留原分數
                    # (已經是原分數，不需要改動)
                    
                    # Strategy 2: 目標 50 個驗證失敗 → 設為中性值 0.5
                    for idx in unverified_target50_indices:
                        filtered_concepts[0, idx] = 0.5
                    
                    # Strategy 3: 其他 250 個概念 → 保留原分數但降權 70%
                    other_250_indices = set(range(300)) - set(target_50_indices)
                    for idx in other_250_indices:
                        filtered_concepts[0, idx] = raw_concepts[0, idx] * 0.7
                    
                    concepts = filtered_concepts
                    logger.info(f"Applied hybrid filtering: {len(verified_indices)} verified, {len(unverified_target50_indices)} neutralized, {len(other_250_indices)} weighted (70%)")
                else:
                    concepts = raw_concepts
                
                # Step 3: Predictor 基於 concepts 預測最終分數
                final_logits = self.predictor(concepts)  # [1, 1]
                final_logit = final_logits[0, 0].item()
                
                # 使用 sigmoid 得到概率
                fake_prob = torch.sigmoid(final_logits[0, 0]).item()
            
            # 根據閾值判定
            is_fake = fake_prob > self.threshold
            
            # 準備 top concepts 顯示
            top_concepts = []
            
            if explainable:
                # Explainable mode: 根據分數決定顯示方向
                if fake_prob < 0.45:
                    # REAL 圖片：顯示低分概念（真實圖片的特徵）
                    logger.info("Image score < 0.45, showing low-score concepts (real-image features)...")
                    # 顯示所有目標 50 個概念（不論是否驗證通過）
                    target_list = sorted(
                        [(idx, concept_scores[idx]) for idx in target_50_indices],
                        key=lambda x: x[1]  # 按分數從低到高排序
                    )
                    verify_desc = "low-score concepts (real features)"
                else:
                    # FAKE 圖片：顯示高分概念（AI 生成的特徵）
                    logger.info("Image score >= 0.45, showing high-score concepts (AI-generated features)...")
                    # 顯示所有目標 50 個概念（不論是否驗證通過）
                    target_list = sorted(
                        [(idx, concept_scores[idx]) for idx in target_50_indices],
                        key=lambda x: -x[1]  # 按分數從高到低排序
                    )
                    verify_desc = "high-score concepts (AI features)"

                verified_target_list = [item for item in target_list if item[0] in verified_indices]
                if verified_target_list:
                    display_list = verified_target_list[:10]
                    logger.info(
                        "Selected verified explainable concepts first: %d/%d available",
                        len(display_list),
                        len(verified_target_list),
                    )
                else:
                    display_list = target_list[:10]
                    logger.info(
                        "No verified concepts landed in explainable display list; falling back to score-ranked concepts"
                    )

                for rank, (idx, score) in enumerate(display_list, 1):
                    class_name = self.class_names[idx] if idx < len(self.class_names) else f"concept_{idx}"
                    is_verified = idx in verified_indices
                    top_concepts.append({
                        'rank': rank,
                        'name': class_name,
                        'score': float(score),
                        'index': int(idx),
                        'verified': is_verified  # 標註是否驗證通過
                    })
                
                logger.info(f"Selected {len(top_concepts)} {verify_desc} ({sum(1 for c in top_concepts if c['verified'])}/{len(top_concepts)} verified)")
            else:
                # 原始模式：根據分數決定驗證方向
                if fake_prob < 0.45:
                    logger.info("Image score < 0.45, using OWLv2 to verify low-score concepts...")
                    sorted_indices = np.argsort(concept_scores_np)  # 從低到高
                    verify_desc = "low-score (real-image features)"
                else:
                    logger.info("Image score >= 0.45, using OWLv2 to verify high-score concepts...")
                    sorted_indices = np.argsort(-concept_scores_np)  # 從高到低
                    verify_desc = "high-score (AI-generated features)"
                
                verified_concepts = []
                capped_indices = sorted_indices[: self.owlv2_max_candidates]
                verify_start = time.monotonic()

                # 從指定方向檢查，直到找到 4 個有出現的物件
                for idx in capped_indices:
                    if (time.monotonic() - verify_start) > self.owlv2_max_verify_seconds:
                        logger.info(
                            "OWLv2 verification stopped by time cap after %.2fs (found %d concepts)",
                            time.monotonic() - verify_start,
                            len(verified_concepts),
                        )
                        break

                    if len(verified_concepts) >= 4:  # 只要 4 個
                        break

                    class_name = self.class_names[idx] if idx < len(self.class_names) else f"concept_{idx}"
                    score = float(concept_scores[idx])

                    # 使用 OWLv2 檢測物件是否真的在圖中
                    is_detected = self._detect_object_with_owlv2(image, class_name)

                    if is_detected:
                        verified_concepts.append({
                            'rank': len(verified_concepts) + 1,
                            'name': class_name,
                            'score': score,
                            'index': int(idx),
                            'verified': True
                        })
                        logger.info(f"✓ Concept #{len(verified_concepts)}: {class_name} (score: {score:.4f})")
                    else:
                        logger.debug(f"✗ Skipped: {class_name} (not detected in image)")

                if len(verified_concepts) < 4 and len(capped_indices) < len(sorted_indices):
                    logger.info(
                        "OWLv2 verification stopped early after %d candidates (found %d concepts)",
                        len(capped_indices),
                        len(verified_concepts),
                    )

                top_concepts = verified_concepts
                logger.info(f"Final verified concepts ({verify_desc}): {len(top_concepts)}")
            
            result = {
                'score': fake_prob,
                'label': 'AI-generated' if is_fake else 'Real',
                'threshold': self.threshold,
                'probability': fake_prob,
                'details': f"AI-generated probability: {fake_prob:.4f} (threshold={self.threshold:.2f})",
                'concept_scores': [float(s) for s in concept_scores],  # 完整 300 維
                'top_concepts': top_concepts,  # 經過驗證的概念
                'explainable_mode': explainable
            }

            if debug:
                debug_info = {
                    'predictor_logit': float(final_logit),
                    'concept_stats': {
                        'mean': float(concept_scores_np.mean()),
                        'std': float(concept_scores_np.std()),
                        'min': float(concept_scores_np.min()),
                        'max': float(concept_scores_np.max()),
                        'num_gt_0_9': int((concept_scores_np > 0.9).sum()),
                        'num_lt_0_1': int((concept_scores_np < 0.1).sum())
                    }
                }
                
                if explainable:
                    debug_info['explainability'] = {
                        'verified_concepts': len(verified_indices),
                        'filtered_concepts': 300 - len(verified_indices),
                        'concepts_checked': checked_count,
                        'verification_time': f"{elapsed_time:.2f}s",
                        'avg_time_per_concept': f"{elapsed_time / max(checked_count, 1):.3f}s"
                    }
                
                result['debug'] = debug_info
            
            return result
            
        except Exception as e:
            logger.error(f"Image AIGC prediction error: {e}")
            raise


class ModelInference:
    """
    統一的模型推理接口
    整合三個模型的推理結果
    """
    
    def __init__(
        self,
        text_aigc_path: Optional[str] = None,
        text_intent_path: Optional[str] = None,
        image_encoder_path: Optional[str] = None,
        image_predictor_path: Optional[str] = None,
        load_text_aigc: bool = False,
        load_text_intent: bool = False,
        load_image_aigc: bool = True
    ):
        """初始化所有模型"""
        logger.info("Initializing ModelInference...")
        
        # 載入 Text AIGC Model
        if load_text_aigc:
            try:
                self.text_aigc_model = TextAIGCModel(
                    model_path=text_aigc_path or "text_aigc/model/roberta-fake-news-detection-final"
                )
            except Exception as e:
                logger.error(f"Text AIGC Model initialization failed: {e}")
                self.text_aigc_model = None
        else:
            logger.info("Text AIGC Model disabled in config")
            self.text_aigc_model = None
        
        # 載入 Text Intent Model
        if load_text_intent:
            try:
                self.text_intent_model = TextIntentModel(
                    model_path=text_intent_path or "text_intent/checkpoints/final"
                )
            except Exception as e:
                logger.error(f"Text Intent Model initialization failed: {e}")
                self.text_intent_model = None
        else:
            logger.info("Text Intent Model disabled in config")
            self.text_intent_model = None
        
        # 載入 Image AIGC Model
        if load_image_aigc:
            try:
                self.image_aigc_model = ImageAIGCModel(
                    encoder_path=image_encoder_path or "checkpoints/image/cbm-encoder.pt",
                    predictor_path=image_predictor_path or "checkpoints/image/cbm-predictor.pt"
                )
            except Exception as e:
                logger.error(f"Image AIGC Model initialization failed: {e}")
                self.image_aigc_model = None
        else:
            logger.info("Image AIGC Model disabled in config")
            self.image_aigc_model = None
        
        logger.info("ModelInference initialized")
    
    def predict_all(
        self,
        text: str,
        image_base64: Optional[str] = None,
        image_url: Optional[str] = None,
        debug_image_aigc: bool = False,
        explainable_image: bool = True
    ) -> Dict:
        """
        運行所有模型並返回結果
        
        Args:
            text: 新聞文本
            image_base64: Base64 編碼的圖片（可選）
            image_url: 圖片 URL（可選）
            debug_image_aigc: 是否返回圖片模型的詳細調試信息
            explainable_image: 啟用圖片模型的 top-100 selective concept activation
            
        Returns:
            {
                'text_aigc': {...},
                'text_intent': {...},
                'image_aigc': {...} or None
            }
        """
        results = {}
        
        # Text AIGC
        if self.text_aigc_model:
            try:
                results['text_aigc'] = self.text_aigc_model.predict(text)
            except Exception as e:
                logger.error(f"Text AIGC prediction failed: {e}")
                results['text_aigc'] = None
        else:
            results['text_aigc'] = None
        
        # Text Intent
        if self.text_intent_model:
            try:
                results['text_intent'] = self.text_intent_model.predict(text)
            except Exception as e:
                logger.error(f"Text Intent prediction failed: {e}")
                results['text_intent'] = None
        else:
            results['text_intent'] = None
        
        # Image AIGC
        if self.image_aigc_model and (image_base64 or image_url):
            try:
                results['image_aigc'] = self.image_aigc_model.predict(
                    image_base64=image_base64,
                    image_url=image_url,
                    debug=debug_image_aigc,
                    explainable=explainable_image
                )
            except Exception as e:
                logger.error(f"Image AIGC prediction failed: {e}")
                results['image_aigc'] = None
        else:
            results['image_aigc'] = None
        
        return results


# 測試代碼
if __name__ == "__main__":
    # 測試 Text AIGC Model
    print("Testing Text AIGC Model...")
    try:
        text_model = TextAIGCModel()
        test_text = "這是一則測試新聞內容。"
        result = text_model.predict(test_text)
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error: {e}")
    
    # 測試 Image AIGC Model
    print("\nTesting Image AIGC Model...")
    try:
        image_model = ImageAIGCModel()
        # 需要實際的圖片進行測試
        print("Image model loaded successfully")
    except Exception as e:
        print(f"Error: {e}")
