from datetime import datetime
import argparse
import json
import logging
import os
import re
import traceback
from typing import Dict, List

import numpy as np
import requests
import torch
import torch.nn as nn
from flask import Flask, jsonify, request
from flask_cors import CORS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

AIGC_URL = os.getenv("AIGC_SERVICE_URL", "http://backend-aigc:5000")
INTENT_URL = os.getenv("INTENT_SERVICE_URL", "http://backend-intent:5000")
IMG_URL = os.getenv("IMG_SERVICE_URL", "http://backend-img:5000")
CALL_TIMEOUT = float(os.getenv("SERVICE_CALL_TIMEOUT", "120"))

# LLM Report Generation Settings
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:1.5b")
HF_MODEL = os.getenv("HF_MODEL", LLM_MODEL)
HF_API_BASE = os.getenv("HF_API_BASE", "https://api-inference.huggingface.co/models").rstrip("/")
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "").strip()
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "120"))
LLM_ENABLED = os.getenv("LLM_REPORT_ENABLED", "true").lower() == "true"

FUSION_CHECKPOINT_PATH = os.getenv(
    "FUSION_CHECKPOINT_PATH",
    "/app/fusion_layer/fusion_dataset/train/model/fusion_model_v2/fusion_mlp_v1_best.pt",
)
FUSION_THRESHOLD = float(os.getenv("FUSION_THRESHOLD", "0.5"))

session = requests.Session()


class FusionMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float):
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class FusionRuntime:
    def __init__(self, checkpoint_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.feature_names: List[str] = []
        self.feature_mean = None
        self.feature_std = None
        self.loaded = False

    def load(self):
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"fusion checkpoint not found: {self.checkpoint_path}")

        ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        self.feature_names = ckpt.get("feature_names", [])
        if not self.feature_names:
            raise ValueError("feature_names missing in fusion checkpoint")

        input_dim = int(ckpt.get("input_dim", len(self.feature_names)))
        hidden_dims = ckpt.get("hidden_dims", [])
        dropout = float(ckpt.get("dropout", 0.0))

        model = FusionMLP(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout).to(self.device)
        model.load_state_dict(ckpt["model_state"], strict=True)
        model.eval()

        self.feature_mean = np.asarray(ckpt.get("feature_mean"), dtype=np.float32)
        self.feature_std = np.asarray(ckpt.get("feature_std"), dtype=np.float32)
        if self.feature_mean.shape[0] != len(self.feature_names) or self.feature_std.shape[0] != len(self.feature_names):
            raise ValueError("feature_mean/feature_std shape mismatch")
        self.feature_std[self.feature_std < 1e-8] = 1.0

        self.model = model
        self.loaded = True
        logger.info("Fusion checkpoint loaded: %s", self.checkpoint_path)

    @staticmethod
    def _safe_float(x, default=0.0):
        try:
            if x is None:
                return default
            return float(x)
        except (TypeError, ValueError):
            return default

    def _feature_value(self, feature_name: str, agg: Dict):
        image_score = (agg.get("imageAigcScore") or {}).get("score")
        roberta_score = (agg.get("textAigcScore") or {}).get("score")
        intent = agg.get("textIntentScore") or {}

        if feature_name == "img_probability":
            return self._safe_float(image_score, 0.0)
        if feature_name == "roberta_p_fake":
            return self._safe_float(roberta_score, 0.0)
        if feature_name == "cbllm_fake_prob":
            return self._safe_float(intent.get("cbllm_fake_prob", intent.get("score", 0.0)), 0.0)
        if feature_name.startswith("cbllm_concept_"):
            idx = int(feature_name.split("_")[-1]) - 1
            concepts = intent.get("cbllm_concept_probs")
            if isinstance(concepts, list) and 0 <= idx < len(concepts):
                return self._safe_float(concepts[idx], 0.0)
            return 0.0
        return 0.0

    def predict(self, agg: Dict) -> float:
        if not self.loaded:
            raise RuntimeError("fusion model not loaded")

        row = [self._feature_value(name, agg) for name in self.feature_names]
        x = np.asarray(row, dtype=np.float32)
        x = (x - self.feature_mean) / self.feature_std

        xb = torch.from_numpy(x).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logit = self.model(xb)
            prob_fake = torch.sigmoid(logit).item()
        return float(prob_fake)


fusion_runtime = FusionRuntime(FUSION_CHECKPOINT_PATH)


def _post_json(base_url: str, path: str, payload: Dict) -> Dict:
    url = f"{base_url}{path}"
    resp = session.post(url, json=payload, timeout=CALL_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def _normalize_report_text(text: str, max_chars: int = 200, keep_markdown: bool = False) -> str:
    """Normalize report format, force traditional Chinese and keep concise length."""
    if not text:
        return text

    if keep_markdown:
        lines = [line.rstrip() for line in text.strip().splitlines()]
        cleaned = "\n".join(lines)
    else:
        cleaned = " ".join(text.strip().split())

        # Remove markdown markers in plain-text mode.
        cleaned = cleaned.replace("**", "")
        cleaned = cleaned.replace("###", "")
        cleaned = cleaned.replace("##", "")
        cleaned = cleaned.replace("#", "")

    cleaned = cleaned.replace("置信度", "confidence")
    # Remove training-time prefix from class names for cleaner report text.
    cleaned = re.sub(r"\bnew_", "", cleaned)

    if keep_markdown:
        return cleaned

    if len(cleaned) <= max_chars:
        return cleaned

    truncated = cleaned[:max_chars]
    # Prefer cutting at sentence boundary for readability.
    cut_points = [truncated.rfind("。"), truncated.rfind("！"), truncated.rfind("？")]
    cut_at = max(cut_points)
    if cut_at >= int(max_chars * 0.6):
        return truncated[: cut_at + 1]
    return truncated.rstrip() + "…"


def _enforce_intent_top1_in_report(text: str, top1_name: str, top1_score: float) -> str:
    """Force Top-1 intent class wording to use topConcepts[0].name."""
    if not text or not top1_name or top1_name == "無":
        return text

    replacement = f"Top 1 意圖類別為「{top1_name}」，分數為 {top1_score:.2%}"
    cleaned = text

    # Guard against common mis-generation where binary label is written as intent class.
    cleaned = re.sub(
        r"Top\s*1\s*意圖類別為?[「\"']?(Neutral|Manipulative)[」\"']?[^。\n]*",
        replacement,
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"Top\s*1\s*類別為?[「\"']?(Neutral|Manipulative)[」\"']?[^。\n]*",
        replacement,
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"Top\s*1\s*(?:意圖類別|類別)\s*[:：]\s*(Neutral|Manipulative)\b[^。\n]*",
        replacement,
        cleaned,
        flags=re.IGNORECASE,
    )

    return cleaned


def _normalize_intent_risk_wording(text: str) -> str:
    """Rewrite awkward Neutral/Manipulative phrasing into risk-oriented wording."""
    if not text:
        return text

    cleaned = text

    # Example to avoid:
    # 「整體意圖被判定為中立，中性意圖的出現概率高達 45.42%，反映內容客觀性強。」
    cleaned = re.sub(
        r"整體意圖被判定為中立[，,]\s*中性意圖的出現概率高達\s*([0-9]+(?:\.[0-9]+)?%)",
        r"整體意圖判定為中立，誘導風險分數為 \1，表示目前未觀察到明顯操控語氣",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"中性意圖的出現概率",
        "誘導風險分數",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"中立意圖的出現概率",
        "誘導風險分數",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"操控意圖的出現概率",
        "誘導風險分數",
        cleaned,
        flags=re.IGNORECASE,
    )

    return cleaned


def _clarify_low_intent_score(text: str, top1_score: float) -> str:
    """Add clarification when Top1 intent score is very low."""
    if not text or top1_score >= 0.05:  # 5% threshold
        return text

    cleaned = text
    percentage = f"{top1_score * 100:.2f}%"

    # Add clarification after Top1 intent sentence
    pattern = r"(Top\s*1\s*意圖類別為[「\"']?[^「\"'。]+[「\"']?[，,]\s*(?:分數|其分數為)[為\s]*[0-9.]+%[。，])"
    replacement = rf"\1最高也只有 {percentage}，表示沒有特別意圖。"
    cleaned = re.sub(pattern, replacement, cleaned)

    return cleaned


def _normalize_visual_keyword_wording(text: str) -> str:
    """Rewrite misleading visual-keyword sentences into AI-trace wording."""
    if not text:
        return text

    cleaned = text

    # Example to avoid:
    # 「視覺特徵分析鎖定關鍵詞為「武器」，暗示畫面中可能包含危險或敏感物件。」
    cleaned = re.sub(
        r"視覺特徵分析鎖定關鍵詞為[「\"]?([^」\"，,。]+)[」\"]?[，,]\s*暗示畫面中可能包含危險或敏感物件[。.]?",
        r"影像模型偵測到「\1」相關物件，且該物件呈現 AI 生成痕跡。",
        cleaned,
        flags=re.IGNORECASE,
    )

    return cleaned


def _remove_intent_top1_mentions(text: str) -> str:
    """Remove Top-1 intent sentences when there is no intent signal."""
    if not text:
        return text

    cleaned = text
    cleaned = re.sub(r"^.*Top\s*1\s*意圖類別.*$", "", cleaned, flags=re.IGNORECASE | re.MULTILINE)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _report_looks_truncated(text: str) -> bool:
    if not text:
        return True

    stripped = text.strip()
    if stripped.endswith(("。", "！", "？")):
        return False
    if "### 最終彙整" not in stripped:
        return True
    return True


def _request_report(request_prompt: str, num_predict: int, temperature: float) -> str:
    """Call the configured LLM provider and return the raw generated text."""
    if LLM_PROVIDER == "hf":
        if not HF_API_TOKEN:
            return "LLM report generation unavailable: missing HF_API_TOKEN."

        response = requests.post(
            f"{HF_API_BASE}/{HF_MODEL}",
            headers={
                "Authorization": f"Bearer {HF_API_TOKEN}",
                "Content-Type": "application/json",
            },
            json={
                "inputs": request_prompt,
                "parameters": {
                    "temperature": temperature,
                    "top_p": 0.9,
                    "max_new_tokens": num_predict,
                    "return_full_text": False,
                },
                "options": {"wait_for_model": True},
            },
            timeout=LLM_TIMEOUT,
        )
        if response.status_code != 200:
            detail = response.text[:300]
            try:
                detail = json.dumps(response.json(), ensure_ascii=False)[:300]
            except Exception:
                pass
            logger.warning("Hugging Face API returned status %d", response.status_code)
            return f"LLM report generation failed with status {response.status_code}: {detail}"

        result = response.json()
        if isinstance(result, list) and result and isinstance(result[0], dict):
            return (result[0].get("generated_text") or "").strip()
        if isinstance(result, dict):
            return (result.get("generated_text") or "").strip()
        return ""

    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": LLM_MODEL,
            "prompt": request_prompt,
            "stream": False,
            "think": False,
            "options": {
                "temperature": temperature,
                "top_p": 0.9,
                "num_predict": num_predict,
            },
        },
        timeout=LLM_TIMEOUT,
    )
    if response.status_code != 200:
        logger.warning("Ollama API returned status %d", response.status_code)
        return f"LLM report generation failed with status {response.status_code}."
    result = response.json()
    return result.get("response", "").strip()


def _generate_llm_report(scores: Dict) -> str:
    """使用 LLM 產生繁體中文 Markdown 報告。"""
    if not LLM_ENABLED:
        return "LLM report disabled."

    text_aigc = scores.get("textAigcScore", {})
    text_intent = scores.get("textIntentScore", {})
    image_aigc = scores.get("imageAigcScore", {})
    final_score = scores.get("finalScore", 0.0)
    final_label = scores.get("finalLabel", "Unknown")

    intent_score_list = text_intent.get("cbllm_concept_probs")
    if not isinstance(intent_score_list, list):
        intent_score_list = [c.get("score", 0.0) for c in (text_intent.get("topConcepts") or [])]
    max_intent_score = max([float(v) for v in intent_score_list], default=0.0)
    no_intent_signal = max_intent_score <= 1e-12

    intent_concepts = []
    if text_intent.get("topConcepts"):
        intent_concepts = [
            f"{str(c['name']).replace('new_', '')}({float(c['score']):.2%})"
            for c in text_intent["topConcepts"][:3]
        ]

    top1_intent = (text_intent.get("topConcepts") or [{}])[0]
    top1_intent_name = str(top1_intent.get("name", "無"))
    top1_intent_score = float(top1_intent.get("score", 0.0) or 0.0)

    image_concepts = []
    if image_aigc.get("topConcepts"):
        # Only include concept names when OWLv2 was used to verify them (debug mode)
        if any(c.get("verified") for c in image_aigc["topConcepts"]):
            image_concepts = [c["name"] for c in image_aigc["topConcepts"][:4] if c.get("verified")]

    top1_summary_line = (
            "  意圖概念分數均為 0.00%，不輸出 Top 1 意圖類別。"
            if no_intent_signal
            else f"  Top 1 意圖類別（固定取自 topConcepts[0].name）：{top1_intent_name}({top1_intent_score:.2%})"
    )

    detection_summary = f"""檢測結果：
- text aigc 檢測：{text_aigc.get('label', 'N/A')} (概率: {text_aigc.get('score', 0):.2%})
- text intent 檢測：{text_intent.get('label', 'N/A')} (概率: {text_intent.get('score', 0):.2%})
{top1_summary_line}
  檢測到的意圖特徵：{', '.join(intent_concepts) if intent_concepts else '無'}
- image aigc 檢測：{image_aigc.get('label', 'N/A')} (概率: {image_aigc.get('score', 0):.2%})
  檢測到的視覺特徵：{', '.join(image_concepts) if image_concepts else '無'}
- 融合模型最終判定：{final_label} (假新聞概率: {final_score:.1%})"""

    requirements = """要求：
1. 必須使用繁體中文輸出，不要使用簡體中文。
2. 請使用台灣常用繁體字詞。
3. 第一句必須提到最終判定與假新聞概率，不要使用「置信度」這個詞。
4. 請使用 Markdown 輸出，必須包含以下段落標題：
   ### 最終判定
   ### text aigc 檢測
   ### text intent 檢測
   ### image aigc 檢測
   ### 融合模型分析
   ### 含義解讀
   ### 最終彙整
5. 每個模型段落至少包含 2 點條列；若意圖概念分數全為 0.00%，請明確寫「無明顯意圖訊號」且不要輸出 Top 1。
6. text intent 段落中的「Top 1 意圖類別」只能使用 topConcepts[0].name，不得使用 Neutral 或 Manipulative 當作意圖類別名稱。
7. 圖片模型段落不要列出圖片類別名稱及類別分數。
8. 報告必須完整結束，不可中途截斷，最後一段一定要寫完「### 最終彙整」並以完整句號結尾。
9. 建議長度約 280-450 字，以完整性優先。"""

    prompt = f"""你是一個專業的假新聞檢測分析師。請根據以下多模態檢測結果，用繁體中文撰寫一份完整的 Markdown 分析報告。

{detection_summary}

{requirements}

分析報告："""

    try:
        report = _request_report(prompt, num_predict=640, temperature=0.5)

        if report and _report_looks_truncated(report):
            retry_prompt = f"""請根據以下相同檢測結果，重新輸出一份完整的繁體中文 Markdown 報告。

{detection_summary}

{requirements}

請直接輸出完整報告："""
            retried_report = _request_report(retry_prompt, num_predict=800, temperature=0.3)
            if retried_report:
                report = retried_report

        if report:
            normalized = _normalize_report_text(report, max_chars=900, keep_markdown=True)
            if no_intent_signal:
                normalized = _remove_intent_top1_mentions(normalized)
            else:
                normalized = _enforce_intent_top1_in_report(
                    normalized,
                    top1_name=top1_intent_name,
                    top1_score=top1_intent_score,
                )
            normalized = _normalize_intent_risk_wording(normalized)
            normalized = _clarify_low_intent_score(normalized, top1_intent_score)
            normalized = _normalize_visual_keyword_wording(normalized)
            used_model = HF_MODEL if LLM_PROVIDER == "hf" else LLM_MODEL
            logger.info("LLM report generated successfully using %s (%s)", used_model, LLM_PROVIDER)
            return normalized

        return "LLM report generation returned empty response."
    except requests.RequestException as e:
        provider_name = "Hugging Face" if LLM_PROVIDER == "hf" else "Ollama"
        logger.warning("%s API unavailable (%s)", provider_name, e)
        return f"LLM report generation unavailable: {e}"
    except Exception as e:
        logger.error("LLM report generation error: %s", e)
        return f"LLM report generation error: {e}"


def _aggregate_predictions(payload: Dict) -> Dict:
    aigc = _post_json(AIGC_URL, "/api/detect", payload)
    intent = _post_json(INTENT_URL, "/api/detect", payload)
    img = _post_json(IMG_URL, "/api/detect", payload)

    merged = {
        "title": payload.get("title", ""),
        "sourceUrl": payload.get("sourceUrl"),
        "textAigcScore": aigc.get("textAigcScore"),
        "textIntentScore": intent.get("textIntentScore"),
        "imageAigcScore": img.get("imageAigcScore"),
        "upstream": {
            "aigc": {"service": AIGC_URL},
            "intent": {"service": INTENT_URL},
            "img": {"service": IMG_URL},
        },
    }

    final_score = fusion_runtime.predict(merged)
    merged["finalScore"] = final_score
    merged["finalLabel"] = "Highly Likely Fake" if final_score >= FUSION_THRESHOLD else "Likely Real"
    merged["fusion"] = {
        "model": "FusionMLP",
        "checkpoint_path": fusion_runtime.checkpoint_path,
        "decision_threshold": FUSION_THRESHOLD,
        "feature_names": fusion_runtime.feature_names,
    }
    merged["analyzedAt"] = datetime.now().isoformat()

    # 生成口語化分析報告
    merged["report"] = _generate_llm_report(merged)

    return merged


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "healthy",
            "gateway": True,
            "aigc_url": AIGC_URL,
            "intent_url": INTENT_URL,
            "img_url": IMG_URL,
            "fusion_loaded": fusion_runtime.loaded,
            "fusion_checkpoint": fusion_runtime.checkpoint_path,
            "timestamp": datetime.now().isoformat(),
        }
    )


@app.route("/api/detect", methods=["POST"])
def detect():
    try:
        payload = request.get_json() or {}
        if not payload.get("title") or not payload.get("content"):
            return jsonify({"error": "Bad Request", "message": "Missing required fields: title, content"}), 400

        result = _aggregate_predictions(payload)
        return jsonify(result)
    except requests.RequestException as e:
        logger.error("Upstream service error: %s", e)
        return jsonify({"error": "Upstream Error", "message": str(e)}), 502
    except Exception as e:
        logger.error("Gateway detect error: %s", e)
        logger.error(traceback.format_exc())
        return jsonify({"error": "Internal Server Error", "message": str(e)}), 500


@app.route("/api/crawl", methods=["POST"])
def crawl_passthrough():
    try:
        payload = request.get_json() or {}
        if not payload.get("url"):
            return jsonify({"error": "Bad Request", "message": "Missing required field: url"}), 400
        data = _post_json(AIGC_URL, "/api/crawl", payload)
        return jsonify(data)
    except requests.RequestException as e:
        return jsonify({"error": "Upstream Error", "message": str(e)}), 502


@app.route("/api/detect-from-url", methods=["POST"])
def detect_from_url():
    try:
        payload = request.get_json() or {}
        if not payload.get("url"):
            return jsonify({"error": "Bad Request", "message": "Missing required field: url"}), 400

        crawl = _post_json(AIGC_URL, "/api/crawl", {"url": payload["url"]})
        title = crawl.get("title", "")
        content = crawl.get("content", "")
        images = crawl.get("imageUrls") or []

        detect_payload = {
            "title": title,
            "content": content,
            "imageUrl": images[0] if images else None,
            "sourceUrl": payload["url"],
            "debugImageAigc": True,
        }
        result = _aggregate_predictions(detect_payload)
        return jsonify(result)
    except requests.RequestException as e:
        return jsonify({"error": "Upstream Error", "message": str(e)}), 502
    except Exception as e:
        logger.error("Gateway detect-from-url error: %s", e)
        logger.error(traceback.format_exc())
        return jsonify({"error": "Internal Server Error", "message": str(e)}), 500


def main():
    parser = argparse.ArgumentParser(description="Gateway API server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    fusion_runtime.load()
    logger.info("Gateway ready")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
