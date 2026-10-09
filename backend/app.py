"""
Flask API 服務器
提供假新聞檢測的 REST API
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import uuid
import logging
from typing import Dict, Optional
import traceback
import sys
import os

# 添加項目根目錄到 Python 路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# 嘗試兩種導入方式
try:
    from backend.crawler import NewsCrawler
    from backend.models import ModelInference
    from backend.fusion import DecisionMaker
except ModuleNotFoundError:
    # 如果從 backend 目錄內運行，使用直接導入
    from crawler import NewsCrawler
    from models import ModelInference
    from fusion import DecisionMaker

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 創建 Flask app
app = Flask(__name__)
CORS(app)  # 允許跨域請求

# 全局變量：模型實例
model_inference = None
decision_maker = None
crawler = None


def initialize_models():
    """初始化所有模型"""
    global model_inference, decision_maker, crawler
    
    logger.info("Initializing models...")
    
    try:
        # 讀取配置
        try:
            from backend.config import MODEL_PATHS
        except ModuleNotFoundError:
            from config import MODEL_PATHS
        
        # 初始化模型推理器
        image_cfg = MODEL_PATHS.get('image_aigc', {})
        model_inference = ModelInference(
            text_aigc_path=MODEL_PATHS.get('text_aigc', {}).get('path'),
            text_intent_path=MODEL_PATHS.get('text_intent', {}).get('path'),
            image_encoder_path=image_cfg.get('encoder_path'),
            image_predictor_path=image_cfg.get('predictor_path'),
            load_text_aigc=MODEL_PATHS['text_aigc'].get('enabled', False),
            load_text_intent=MODEL_PATHS['text_intent'].get('enabled', False),
            load_image_aigc=MODEL_PATHS['image_aigc'].get('enabled', True)
        )
        logger.info("✓ Model inference initialized")
        
        # 初始化決策器
        decision_maker = DecisionMaker(
            fusion_method='linear',
            weights={
                'text_aigc': 0.4,
                'text_intent': 0.3,
                'image_aigc': 0.3
            },
            threshold=0.5
        )
        logger.info("✓ Decision maker initialized")
        
        # 初始化爬蟲
        crawler = NewsCrawler()
        logger.info("✓ Crawler initialized")
        
        logger.info("All services ready!")
        
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")
        logger.error(traceback.format_exc())
        raise


def format_model_score(
    model_name: str,
    model_type: str,
    result: Optional[Dict]
) -> Dict:
    """
    格式化模型分數為前端期望的格式
    
    Returns:
        {
            'modelName': str,
            'modelType': str,
            'score': float (0-1),
            'confidence': float,
            'label': str,
            'details': str,
            'conceptScores': [float] (optional, for Image AIGC),
            'topConcepts': [dict] (optional, for Image AIGC)
        }
    """
    if result is None:
        return {
            'modelName': model_name,
            'modelType': model_type,
            'score': 0.5,
            'confidence': 0.0,
            'label': 'N/A',
            'details': 'Model prediction failed or unavailable'
        }
    
    formatted = {
        'modelName': model_name,
        'modelType': model_type,
        'score': result.get('score', 0.5),
        'confidence': result.get('confidence', 0.0),
        'label': result.get('label', 'Unknown'),
        'details': result.get('details', '')
    }
    
    # 如果有 concept_scores 和 top_concepts，添加到響應中
    if 'concept_scores' in result:
        formatted['conceptScores'] = result['concept_scores']
    if 'top_concepts' in result:
        formatted['topConcepts'] = result['top_concepts']
    if 'cbllm_fake_prob' in result:
        formatted['cbllm_fake_prob'] = result['cbllm_fake_prob']
    if 'cbllm_concept_probs' in result:
        formatted['cbllm_concept_probs'] = result['cbllm_concept_probs']
    if 'cbllm_concept_names' in result:
        formatted['cbllm_concept_names'] = result['cbllm_concept_names']
    if 'debug' in result:
        formatted['debug'] = result['debug']
    
    return formatted


@app.route('/', methods=['GET'])
def index():
    """API 根路徑 - 顯示 API 文檔"""
    return jsonify({
        'service': '假新聞檢測 API',
        'version': '1.0.0',
        'status': 'running',
        'endpoints': {
            'health': {
                'method': 'GET',
                'path': '/health',
                'description': '健康檢查'
            },
            'crawl': {
                'method': 'POST',
                'path': '/api/crawl',
                'description': '爬取新聞 URL',
                'body': {
                    'url': 'string (required)'
                }
            },
            'detect': {
                'method': 'POST',
                'path': '/api/detect',
                'description': '檢測假新聞（手動輸入）',
                'body': {
                    'title': 'string (required)',
                    'content': 'string (required)',
                    'imageUrl': 'string (optional)',
                    'imageBase64': 'string (optional)',
                    'debugImageAigc': 'boolean (optional)'
                }
            },
            'detect_from_url': {
                'method': 'POST',
                'path': '/api/detect-from-url',
                'description': '從 URL 爬取並檢測',
                'body': {
                    'url': 'string (required)'
                }
            }
        },
        'frontend_url': 'http://localhost:5173',
        'documentation': '/health'
    })


@app.route('/health', methods=['GET'])
def health_check():
    """健康檢查端點"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': model_inference is not None,
        'decision_maker_loaded': decision_maker is not None,
        'crawler_loaded': crawler is not None
    })


@app.route('/api/crawl', methods=['POST'])
def crawl_url():
    """
    爬取新聞 URL
    
    Request Body:
        {
            "url": "https://www.ettoday.net/news/..."
        }
    
    Response:
        {
            "url": str,
            "title": str,
            "content": str,
            "imageUrls": [str],
            "publishedAt": str or null,
            "author": str or null,
            "siteName": str
        }
    """
    try:
        data = request.get_json()
        
        if not data or 'url' not in data:
            return jsonify({
                'error': 'Bad Request',
                'message': 'Missing required field: url'
            }), 400
        
        url = data['url']
        logger.info(f"Crawling URL: {url}")
        
        # 爬取新聞
        result = crawler.crawl(url)
        
        if 'error' in result:
            return jsonify({
                'error': 'Crawl Failed',
                'message': result['error']
            }), 500
        
        # 格式化響應
        response = {
            'url': result.get('url', url),
            'title': result.get('title', ''),
            'content': result.get('content', ''),
            'imageUrls': result.get('images', []),
            'publishedAt': result.get('published_at'),
            'author': result.get('author'),
            'siteName': result.get('site_name', 'Unknown')
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Crawl error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Internal Server Error',
            'message': str(e)
        }), 500


@app.route('/api/detect', methods=['POST'])
def detect_fake_news():
    """
    檢測假新聞
    
    Request Body:
        {
            "title": str,
            "content": str,
            "imageBase64": str (optional),
            "imageUrl": str (optional),
            "sourceUrl": str (optional)
        }
    
    Response:
        {
            "id": str,
            "title": str,
            "sourceUrl": str or null,
            "textAigcScore": ModelScore,
            "textIntentScore": ModelScore,
            "imageAigcScore": ModelScore,
            "finalScore": float (0-1),
            "finalLabel": str,
            "finalConfidence": float,
            "report": str,
            "analyzedAt": str (ISO format)
        }
    """
    try:
        data = request.get_json()
        
        # 驗證必需字段
        if not data or 'title' not in data or 'content' not in data:
            return jsonify({
                'error': 'Bad Request',
                'message': 'Missing required fields: title, content'
            }), 400
        
        title = data['title']
        content = data['content']
        image_base64 = data.get('imageBase64')
        image_url = data.get('imageUrl')
        source_url = data.get('sourceUrl')
        debug_image_aigc = bool(data.get('debugImageAigc', False))
        
        logger.info(f"Detecting fake news: {title[:50]}...")
        
        # 合併文本（標題 + 內容）
        full_text = f"{title} {content}"
        
        # 運行所有模型
        model_results = model_inference.predict_all(
            text=full_text,
            image_base64=image_base64,
            image_url=image_url,
            debug_image_aigc=debug_image_aigc
        )

        image_result = model_results.get('image_aigc')
        if image_result:
            logger.info(
                "Image AIGC probability=%.6f label=%s threshold=%.2f",
                float(image_result.get('score', 0.5)),
                image_result.get('label', 'Unknown'),
                float(image_result.get('threshold', 0.45))
            )
        else:
            logger.info("Image AIGC probability unavailable (image_aigc result is None)")
        
        # 做出最終決策
        decision = decision_maker.make_decision(model_results)
        
        # 格式化響應
        response = {
            'id': str(uuid.uuid4()),
            'title': title,
            'sourceUrl': source_url,
            'textAigcScore': format_model_score(
                'RoBERTa',
                'Text AIGC Detection',
                model_results.get('text_aigc')
            ),
            'textIntentScore': format_model_score(
                'CB-LLM (Qwen-3)',
                'Text Intent Detection',
                model_results.get('text_intent')
            ),
            'imageAigcScore': format_model_score(
                'CBM (Concept Bottleneck)',
                'Image AIGC Detection',
                model_results.get('image_aigc')
            ),
            'finalScore': decision['final_score'],
            'finalLabel': decision['final_label'],
            'finalConfidence': decision['final_confidence'],
            'report': decision['report'],
            'analyzedAt': datetime.now().isoformat()
        }
        
        logger.info(f"Detection complete: {decision['final_label']} (score={decision['final_score']:.3f})")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Detection error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Internal Server Error',
            'message': str(e)
        }), 500


@app.route('/api/detect-from-url', methods=['POST'])
def detect_from_url():
    """
    從 URL 爬取並檢測假新聞（組合端點）
    
    Request Body:
        {
            "url": str
        }
    
    Response:
        Same as /api/detect
    """
    try:
        data = request.get_json()
        
        if not data or 'url' not in data:
            return jsonify({
                'error': 'Bad Request',
                'message': 'Missing required field: url'
            }), 400
        
        url = data['url']
        debug_image_aigc = bool(data.get('debugImageAigc', False))
        logger.info(f"Detecting from URL: {url}")
        
        # Step 1: 爬取新聞
        crawl_result = crawler.crawl(url)
        
        if 'error' in crawl_result:
            return jsonify({
                'error': 'Crawl Failed',
                'message': crawl_result['error']
            }), 500
        
        # Step 2: 檢測假新聞
        title = crawl_result.get('title', '')
        content = crawl_result.get('content', '')
        images = crawl_result.get('images', [])
        
        if not title or not content:
            return jsonify({
                'error': 'Insufficient Content',
                'message': 'Failed to extract title or content from URL'
            }), 400
        
        # 使用第一張圖片（如果有）
        image_url = images[0] if images else None
        
        # 合併文本
        full_text = f"{title} {content}"
        
        # 運行所有模型
        model_results = model_inference.predict_all(
            text=full_text,
            image_url=image_url,
            debug_image_aigc=debug_image_aigc
        )

        image_result = model_results.get('image_aigc')
        if image_result:
            logger.info(
                "Image AIGC probability=%.6f label=%s threshold=%.2f",
                float(image_result.get('score', 0.5)),
                image_result.get('label', 'Unknown'),
                float(image_result.get('threshold', 0.45))
            )
        else:
            logger.info("Image AIGC probability unavailable (image_aigc result is None)")
        
        # 做出最終決策
        decision = decision_maker.make_decision(model_results)
        
        # 格式化響應
        response = {
            'id': str(uuid.uuid4()),
            'title': title,
            'sourceUrl': url,
            'textAigcScore': format_model_score(
                'RoBERTa',
                'Text AIGC Detection',
                model_results.get('text_aigc')
            ),
            'textIntentScore': format_model_score(
                'CB-LLM (Qwen-3)',
                'Text Intent Detection',
                model_results.get('text_intent')
            ),
            'imageAigcScore': format_model_score(
                'CBM (Concept Bottleneck)',
                'Image AIGC Detection',
                model_results.get('image_aigc')
            ),
            'finalScore': decision['final_score'],
            'finalLabel': decision['final_label'],
            'finalConfidence': decision['final_confidence'],
            'report': decision['report'],
            'analyzedAt': datetime.now().isoformat()
        }
        
        logger.info(f"Detection from URL complete: {decision['final_label']} (score={decision['final_score']:.3f})")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Detect from URL error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Internal Server Error',
            'message': str(e)
        }), 500


@app.errorhandler(404)
def not_found(error):
    """404 錯誤處理"""
    return jsonify({
        'error': 'Not Found',
        'message': 'The requested endpoint does not exist'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """500 錯誤處理"""
    return jsonify({
        'error': 'Internal Server Error',
        'message': 'An unexpected error occurred'
    }), 500


def main():
    """啟動服務器"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fake News Detection API Server')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # 初始化模型
    initialize_models()
    
    # 啟動服務器
    logger.info(f"Starting server on {args.host}:{args.port}...")
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        threaded=True
    )


if __name__ == '__main__':
    main()
