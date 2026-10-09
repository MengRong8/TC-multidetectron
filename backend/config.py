"""
配置文件
存儲模型路徑、API 設置等配置
"""
import os
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}

# 項目根目錄
PROJECT_ROOT = Path(__file__).parent.parent

# 模型路徑配置
MODEL_PATHS = {
    # Text AIGC Model (RoBERTa)
    'text_aigc': {
        'path': os.getenv(
            'TEXT_AIGC_MODEL_PATH',
            os.path.join(PROJECT_ROOT, 'web_crawling/model/roberta-fake-news-detection-final')
        ),
        'enabled': _env_bool('TEXT_AIGC_ENABLED', True),
    },
    
    # Text Intent Model (CB-LLM)
    'text_intent': {
        'path': os.getenv(
            'TEXT_INTENT_MODEL_PATH',
            os.path.join(PROJECT_ROOT, 'cb_llm_induction/cbm_checkpoints_V4/cbm_epoch_4.pth')
        ),
        'enabled': _env_bool('TEXT_INTENT_ENABLED', True),
    },
    
    # Image AIGC Model (CBM)
    'image_aigc': {
        'encoder_path': os.getenv(
            'IMAGE_AIGC_ENCODER_PATH',
            os.path.join(PROJECT_ROOT, 'checkpoints/image/cbm-encoder.pt')
        ),
        'predictor_path': os.getenv(
            'IMAGE_AIGC_PREDICTOR_PATH',
            os.path.join(PROJECT_ROOT, 'checkpoints/image/cbm-predictor.pt')
        ),
        'threshold': float(os.getenv('IMAGE_AIGC_THRESHOLD', '0.45')),
        'enabled': _env_bool('IMAGE_AIGC_ENABLED', True),
    }
}

# 決策層配置
FUSION_CONFIG = {
    'method': 'linear',  # 'linear', 'max', 'min', 'avg', 'voting'
    'weights': {
        'text_aigc': 0.4,
        'text_intent': 0.3,
        'image_aigc': 0.3
    },
    'threshold': 0.5
}

# API 服務器配置
API_CONFIG = {
    'host': '0.0.0.0',
    'port': 5000,
    'debug': False,
    'cors_origins': '*',  # 允許所有跨域請求
    'max_content_length': 16 * 1024 * 1024  # 16MB 最大上傳大小
}

# 日誌配置
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': os.path.join(PROJECT_ROOT, 'backend/logs/api.log')
}

# 爬蟲配置
CRAWLER_CONFIG = {
    'timeout': 30,
    'max_retries': 3,
    'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'max_images_per_article': 5
}

# Google Custom Search API (用於爬蟲)
GOOGLE_API_CONFIG = {
    'api_key': os.getenv('GOOGLE_API_KEY', ''),
    'cx': os.getenv('GOOGLE_CSE_ID', '')
}
