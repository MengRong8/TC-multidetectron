"""
推理層(nli+llm hybrid) 架構圖：
├─ 基礎函數
│  ├─ split_sentences(): 句子分割
│  ├─ nli_predict(): NLI 矛盾檢測（單條）
│  ├─ nli_predict_batch(): ⭐ NLI批量預測（效能優化）
│  └─ token_overlap_ratio(): （可選）相似度過濾，矛盾檢測時通常不需使用
│
├─ Claim 层     
│  ├─ extract_factual_claims(): 提取可驗證聲明
│  └─ query_llm_confidence(): LLM 置信度查詢
│
├─ 分類層
│  └─ classify_contradiction_type(): 矛盾類型分類
│
├─ Google 层 (可选)
│  └─ google_search_claim(): Google API 集成
│
├─ 混合驗證核心
│  ├─ hybrid_verify_contradiction(): 主驗證函數
│  └─ synthesize_verdict(): 綜合判斷
│
└─ 主程序
   └─ detect_internal_contradictions(): ⭐ 全文矛盾檢測（批量NLI優化）
"""

import json
import re
from openai import OpenAI
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Tuple, Optional

# 嘗試載入 spaCy，若無則使用 Regex
try:
    import spacy
    try:
        nlp = spacy.load("zh_core_web_trf")
    except:
        try:
            nlp = spacy.load("zh_core_web_sm")
        except:
            nlp = None
except ImportError:
    nlp = None

# ========================
# 初始化
# ========================
client = OpenAI(api_key="")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

DEVICE = None
tokenizer = None
model = None

# ========================
# 基礎工具函數
# ========================
def cosine_sim(a, b):
    """計算餘弦相似度"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def split_sentences(text):
    """將文本分割成句子"""
    if nlp:
        doc = nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    else:
        return [s.strip() for s in re.split(r'(?<=[。！？.!?])\s*', text) if s.strip()]


def token_overlap_ratio(a, b):
    """計算兩段文本的token重疊率
    該函數主要是為去重設計，在純粹檢測矛盾時可以不使用，可視為備用。
    """
    a_tokens = set(a.split())
    b_tokens = set(b.split())
    overlap = len(a_tokens & b_tokens)
    return overlap / max(len(a_tokens), 1)


# ========================
# 核心NLI函數
# ========================
def nli_predict(premise: str, hypothesis: str) -> Dict:
    """
    使用NLI模型預測兩句話的關係
    返回: {label, confidence, prob_distribution}
    """
    MAX_LENGTH = 512
    inputs = tokenizer(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=-1)[0]
    id2label = model.config.id2label
    pred = torch.argmax(probs).item()

    return {
        "label": id2label[pred],
        "confidence": float(probs[pred]),
        "prob_distribution": {
            id2label[i]: float(probs[i])
            for i in range(len(probs))
        }
    }


def nli_predict_batch(premises: List[str], hypotheses: List[str]) -> List[Dict]:
    """
    批量預測NLI關係（效能優化）
    一次性將多個句子對送給模型，大幅降低推理時間
    
    Args:
        premises: 前提句子列表
        hypotheses: 假設句子列表
    
    返回: [{label, confidence, prob_distribution}, ...]
    """
    if not premises or not hypotheses:
        return []
    
    if len(premises) != len(hypotheses):
        raise ValueError(f"Premises ({len(premises)}) and hypotheses ({len(hypotheses)}) must have same length")
    
    MAX_LENGTH = 512
    
    # 批量編碼
    inputs = tokenizer(
        premises,
        hypotheses,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

    # 逐條提取結果
    probs_batch = torch.softmax(outputs.logits, dim=-1)
    id2label = model.config.id2label
    
    results = []
    for i, probs in enumerate(probs_batch):
        pred = torch.argmax(probs).item()
        results.append({
            "label": id2label[pred],
            "confidence": float(probs[pred]),
            "prob_distribution": {
                id2label[j]: float(probs[j])
                for j in range(len(probs))
            }
        })
    
    return results


# ========================
# Claim提取與驗證
# ========================
def extract_factual_claims(sentence: str, article_context: str = "") -> List[Dict]:
    """
    使用LLM從句子提取可驗證的claim
    返回: [{"claim": str, "type": str, "priority": str}, ...]
    """
    context_text = article_context[:500] if article_context else "無"
    
    extraction_prompt = f"""
    請從目標句子中提取可驗證的事實聲明（Claim）。
    
    【文章背景 Context】：
    {context_text}
    
    【目標句子】：
    "{sentence}"
    
    任務要求：
    1. 提取可由外部資訊驗證的事實。
    2. **上下文補全（極重要）**：如果句子中有代名詞（他、該公司、此政策）或省略的背景（時間、地點），請務必根據【文章背景】將其還原為具體的專有名詞。
    3. **生成最佳搜尋關鍵字**：為了進行 Fact-checking，請針對該聲明生成一組最精煉、最容易在 Google 搜出結果的關鍵字（Search Query）。
    
    請務必返回合法的 JSON 物件，格式必須包含一個 "claims" 陣列，如下所示：
    {{
        "claims": [
            {{
                "claim": "補齊主語與背景後的完整聲明（例：台灣教育部2026年推出的新型教育政策初期效果不明）",
                "search_query": "教育部 新型教育政策 效果 2026 台灣",
                "type": "date|number|person|organization|event|other",
                "priority": "high|medium|low"
            }}
        ]
    }}
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": extraction_prompt}],
            temperature=0.3,
            max_tokens=800, 
            response_format={"type": "json_object"} 
        )
        
        content = response.choices[0].message.content.strip()
        
        result_dict = json.loads(content)
            
        return result_dict.get("claims", [])
    except Exception as e:
        print(f"[Warning] Claim extraction failed: {e}")
        return []


def query_llm_confidence(claim: str) -> Dict:
    """
    查詢LLM對某個聲明的置信度
    返回: {confidence: 0-100, source_type: str, reasoning: str}
    """
    prompt = f"""
    對於這個聲明："{claim}"
    
    請根據你的訓練數據評估（JSON格式）：
    {{
      "confidence": <0-100之間的數字>,
      "source_type": "training_data | reasoning | uncertain",
      "reasoning": "簡短說明為什麼"
    }}
    
    注意：
    - confidence: 你對這個資訊準確性的確信度（0=完全不確定，100=非常確定）
    - 如果涉及最近發生的事件（2023+），confidence應該較低
    - 如果是明確的歷史事實，confidence應該較高
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=300, 
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content.strip()

        result = json.loads(content)
        
        # 確保confidence是0-100之間的數字
        result['confidence'] = max(0, min(100, int(result.get('confidence', 50))))
        return result
    except Exception as e:
        print(f"[Warning] Confidence query failed: {e}")
        return {
            'confidence': 50,
            'source_type': 'uncertain',
            'reasoning': f'Error: {str(e)}'
        }


def classify_contradiction_type(sent1: str, sent2: str, article_context: str = "") -> Dict:
    context_text = article_context[:300] if article_context else "[無上文]"
    
    # 1. 提示詞中【必須】包含 "JSON" 這個詞，你原本已經有了，這點很好
    prompt = f"""
    判斷這兩個句子的矛盾性質：
    
    句子1: {sent1}
    句子2: {sent2}
    
    前文背景（供參考）: {context_text}
    
    請分析並返回 JSON 格式：
    {{
      "contradiction_type": "觀點差異" | "事實對立" | "時間矛盾" | "上下文差異" | "非矛盾",
      "is_real_contradiction": true | false,
      "confidence": <0-1之間的浮點數>,
      "explanation": "簡短說明"
    }}
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=400,
            # 2. 加入這行：強制模型輸出合法的 JSON 物件
            response_format={"type": "json_object"} 
        )
        
        # 3. 直接拿取字串
        content = response.choices[0].message.content.strip()
        
        # 4. 直接解析，不需要再用 Regex (re.search) 找括號了
        result = json.loads(content)
        
        return result
        
    except Exception as e:
        print(f"[Warning] Contradiction classification failed: {e}")
        return {
            'contradiction_type': '未知',
            'is_real_contradiction': False,
            'confidence': 0.0,
            'explanation': f'Error: {str(e)}'
        }


# ========================
# Google API（可选）
# ========================
def google_search_claim(claim: str, top_k: int = 3) -> List[Dict]:
    """
    使用Google Custom Search API搜索claim
    注意：需要配置 GOOGLE_API_KEY 和 GOOGLE_CSE_ID 环境变量
    
    返回: [{"title": str, "link": str, "snippet": str}, ...]
    """
    try:
        from googleapiclient.discovery import build
        import os
        
        api_key = os.getenv('GOOGLE_API_KEY')
        cse_id = os.getenv('GOOGLE_CSE_ID')
        
        if not api_key or not cse_id:
            print("[Info] Google API credentials not configured, skipping search")
            return []
        
        service = build("customsearch", "v1", 
                       developerKey=api_key,
                       customsearchengineId=cse_id)
        
        results = service.cse().list(q=claim, num=top_k).execute()
        
        return [
            {
                'title': item.get('title', ''),
                'link': item.get('link', ''),
                'snippet': item.get('snippet', '')
            }
            for item in results.get('items', [])
        ]
    except Exception as e:
        print(f"[Info] Google search unavailable: {e}")
        return []


def synthesize_verdict(
    sent1: str,
    sent2: str,
    verification: Dict, # LLM對每個claim的評估
    google_results: Dict, # Google API的驗證結果
    classification: Dict, # 矛盾類型分類結果
    contradiction_score: float # NLI模型給出的矛盾概率分布中的矛盾得分
) -> Dict:
    """
    綜合所有資訊做最終判斷
    """
    
    # 計算每個句子的平均可信度
    sent1_claims = [c for c in verification.keys() if verification[c].get('llm_confidence', 0) > 0]
    sent2_claims = [c for c in verification.keys()]
    
    # 簡單的可信度計算
    credibility_1 = np.mean([verification[c]['llm_confidence'] for c in sent1_claims]) if sent1_claims else 50
    credibility_2 = np.mean([verification[c]['llm_confidence'] for c in sent2_claims]) if sent2_claims else 50
    
    # 計算嚴重性
    if contradiction_score > 0.8 and abs(credibility_1 - credibility_2) > 30:
        severity = 'high'
    elif contradiction_score > 0.7 or abs(credibility_1 - credibility_2) > 20:
        severity = 'medium'
    else:
        severity = 'low'
    
    # 生成證據總結
    evidence_parts = []
    
    if google_results:
        for claim, results in google_results.items():
            if results:
                evidence_parts.append(f"Google搜索'{claim}'：找到{len(results)}條相關結果")
    
    if verification:
        low_confidence_claims = [c for c, v in verification.items() if v['llm_confidence'] < 40]
        if low_confidence_claims:
            evidence_parts.append(f"LLM對以下claim置信度低：{', '.join(low_confidence_claims[:2])}")
    
    evidence_summary = ";".join(evidence_parts) if evidence_parts else f"NLI檢測到矛盾(得分:{contradiction_score:.2f})"
    
    return {
        'contradiction': True,
        'type': classification.get('contradiction_type', '未分類'),
        'severity': severity,
        'sent1': {
            'text': sent1,
            'credibility': int(credibility_1),
            'claim_count': len([c for c in verification.values() if c['priority'] == 'high'])
        },
        'sent2': {
            'text': sent2,
            'credibility': int(credibility_2),
            'claim_count': len([c for c in verification.values()])
        },
        'evidence_summary': evidence_summary,
        'google_api_used': len(google_results) > 0,
        'nli_contradiction_score': float(contradiction_score),
        'classification_confidence': float(classification.get('confidence', 0.5))
    }


# ========================
# 主程式：內部矛盾檢測
# ========================
def detect_internal_contradictions(article: str, use_google: bool = False) -> List[Dict]:
    """
    檢測文章內部的矛盾
    
    返回: 按嚴重性排序的矛盾列表
    
    優化：使用 batch inference 進行 NLI 預測，而非逐條預測
    """
    
    sentences = split_sentences(article)
    
    if len(sentences) < 2:
        return []
    
    print(f"[Info] Article split into {len(sentences)} sentences")
    
    # ===== 第一步：生成所有句子對 =====
    sentence_pairs = []
    pair_indices = []
    
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            sent1 = sentences[i]
            sent2 = sentences[j]
            
            # 跳過太短的句子
            if len(sent1) < 7 or len(sent2) < 7:
                continue
            
            sentence_pairs.append((sent1, sent2))
            pair_indices.append((i, j))
    
    if not sentence_pairs:
        print("[Info] No valid sentence pairs to compare")
        return []
    
    print(f"[Info] Generated {len(sentence_pairs)} sentence pairs for comparison")
    
    # ===== 第二步：批量 NLI 推理 =====
    print(f"[Info] Running batch NLI inference on {len(sentence_pairs)} pairs...")
    premises = [pair[0] for pair in sentence_pairs]
    hypotheses = [pair[1] for pair in sentence_pairs]
    
    nli_results = nli_predict_batch(premises, hypotheses)
    
    # ===== 第三步：篩選有矛盾嫌疑的句子對 =====
    contradictions = []
    contradiction_candidates = []
    
    for idx, (nli_result, (sent1, sent2), (i, j)) in enumerate(
        zip(nli_results, sentence_pairs, pair_indices)
    ):
        contradiction_score = nli_result['prob_distribution'].get('contradiction', 0)
        
        # 只保留可能有矛盾的句子對（加速篩選）
        if contradiction_score >= 0.5:
            print("sent1:", sent1)
            print("sent2:", sent2)
            contradiction_candidates.append({
                'sent1': sent1,
                'sent2': sent2,
                'i': i,
                'j': j,
                'nli_score': contradiction_score,
                'nli_result': nli_result
            })
    
    print(f"[Info] Found {len(contradiction_candidates)} potential contradictions (score >= 0.5)")
    
    # ===== 第四步：對候選矛盾進行詳細驗證 =====
    excluded_candidates = []  # 被排除的候選（for debugging）
    for candidate_idx, candidate in enumerate(contradiction_candidates):
        sent1 = candidate['sent1']
        sent2 = candidate['sent2']
        i = candidate['i']
        j = candidate['j']
        
        print(f"\n[Verifying {candidate_idx+1}/{len(contradiction_candidates)}] Sentence {i} vs {j}")
        
        # 使用已有的 NLI 結果，避免重複預測
        classification = classify_contradiction_type(sent1, sent2, article)
        
        print(f"  Classification: {classification.get('contradiction_type', 'N/A')}")
        print(f"  Is Real Contradiction: {classification.get('is_real_contradiction', False)}")
        print(f"  Explanation: {classification.get('explanation', 'N/A')}")
        
        if not classification.get('is_real_contradiction', False):
            # 記錄被排除的候選，供調試
            excluded_candidates.append({
                'sent1': sent1,
                'sent2': sent2,
                'i': i,
                'j': j,
                'nli_score': candidate['nli_score'],
                'classification': classification,
                'reason': '分類器認為不是真實矛盾'
            })
            print(f"  ❌ Excluded (is_real_contradiction=False)\n")
            continue
        
        # 提取 claim 並驗證
        claims1 = extract_factual_claims(sent1, article)
        claims2 = extract_factual_claims(sent2, article)
        all_claims = claims1 + claims2
        
        print(f"  Extracted {len(all_claims)} claims")
        
        if not all_claims:
            # 沒有可驗證的claim，使用NLI結果
            result = {
                'contradiction': True,
                'type': classification.get('contradiction_type', '未分類'),
                'severity': 'medium',
                'sent1': {'text': sent1, 'credibility': 50},
                'sent2': {'text': sent2, 'credibility': 50},
                'evidence_summary': '無法提取具體claim，基於NLI判斷存在矛盾',
                'google_api_used': False,
                'sent1_idx': i,
                'sent2_idx': j,
                'classification_explanation': classification.get('explanation', '')
            }
            contradictions.append(result)
            print(f"  ✅ Added (no claims extracted)\n")
            continue
        
        # LLM 自身判斷
        verification = {}
        for claim_obj in all_claims:
            claim_text = claim_obj.get('claim', '')
            if not claim_text:
                continue
            
            llm_conf = query_llm_confidence(claim_text)
            verification[claim_text] = {
                'llm_confidence': llm_conf.get('confidence', 50),
                'source_type': llm_conf.get('source_type', 'uncertain'),
                'claim_type': claim_obj.get('type', 'other'),
                'priority': claim_obj.get('priority', 'medium'),
                'reasoning': llm_conf.get('reasoning', ''),
                'search_query': claim_obj.get('search_query', '')
            }
        
        # Google API 驗證
        google_results = {}
        claims_needing_google = [
            claim_text for claim_text, info in verification.items()
            if info['llm_confidence'] < 60 and info['priority'] == 'high'
        ]
        
        if use_google and claims_needing_google:
            print(f"  [Google API] Verifying {len(claims_needing_google)} claims")
            for claim_text in claims_needing_google:
                google_results[claim_text] = google_search_claim(claim_text)
        
        # 綜合判斷
        final_verdict = synthesize_verdict(
            sent1, sent2,
            verification,
            google_results,
            classification,
            candidate['nli_score']
        )
        final_verdict['sent1_idx'] = i
        final_verdict['sent2_idx'] = j
        final_verdict['classification_explanation'] = classification.get('explanation', '')
        final_verdict['claims_verified'] = verification
        
        contradictions.append(final_verdict)
        print(f"  ✅ Added (severity: {final_verdict.get('severity', 'N/A')})\n")
    
    # 按嚴重性排序
    severity_order = {'high': 0, 'medium': 1, 'low': 2}
    contradictions.sort(
        key=lambda x: severity_order.get(x.get('severity', 'low'), 3)
    )
    
    # 返回包含確認矛盾和被排除候選的完整結果
    return {
        'confirmed': contradictions,
        'excluded': excluded_candidates,
        'total_candidates': len(contradiction_candidates),
        'total_confirmed': len(contradictions),
        'total_excluded': len(excluded_candidates)
    }


# ========================
# 主函數
# ========================
if __name__ == "__main__":
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Info] Using device: {DEVICE}")
    
    # 載入模型
    MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    print(f"[Info] Loading model: {MODEL_NAME}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    # Debug
    print(f"[Debug] 模型標籤映射 (id2label): {model.config.id2label}")
    
    model.to(DEVICE)
    model.eval()
    
    # 測試文章
    article = """
    昨日上午，台北車站發生嚴重的地下瓦斯管線氣爆事件，導致整個車站大廳屋頂完全坍塌，現場死傷慘重。許多目擊民眾在社群媒體上表示，當時聽到巨大的震天巨響，感到非常驚恐。然而，台北市政府消防局隨後發布的官方聲明卻指出，台北車站昨日全天營運狀況一切正常，並未接獲任何火災或爆炸的報案紀錄。針對這起事件的責任歸屬，有土木專家痛批市府的管線維護極度草率，但也有市府官員反駁這純屬不可抗力的意外，雙方各執一詞。令人不解的是，交通部部長在昨晚的緊急記者會上不僅確認了氣爆的發生，更宣布因受損嚴重，全台高鐵將自今日起無限期全面停駛。
    """
    
    print("\n" + "="*60)
    print("Method 3: 混合驗證內部矛盾檢測")
    print("="*60 + "\n")
    
    # 運行檢測
    result = detect_internal_contradictions(article, use_google=False)
    
    # 提取結果
    contradictions = result['confirmed']
    excluded_candidates = result['excluded']
    
    # 輸出結果
    output = {
        'article_summary': article[:100] + "...",
        'total_candidates': result['total_candidates'],
        'total_confirmed': result['total_confirmed'],
        'total_excluded': result['total_excluded'],
        'confirmed_contradictions': contradictions,
        'excluded_candidates': excluded_candidates
    }
    
    output_str = json.dumps(output, indent=2, ensure_ascii=False)
    print("\n" + "="*60)
    print("檢測結果")
    print("="*60)
    print(output_str)
    
    # 保存到檔案
    with open("nli_truth_log.json", "w", encoding="utf-8") as f:
        f.write(output_str)
    
    print("\n[Info] Results saved to nli_truth_log.json")
    
    # 生成可讀的報告
    with open("nli_truth_report_3.md", "w", encoding="utf-8") as f:
        f.write("# 新聞邏輯矛盾檢測報告\n\n")
        f.write(f"## 摘要\n\n")
        f.write(f"- 文章：{article[:80]}...\n")
        f.write(f"- 檢測到的候選矛盾：{result['total_candidates']}\n")
        f.write(f"- 確認的矛盾：{result['total_confirmed']} ✅\n")
        f.write(f"- 被排除的候選：{result['total_excluded']} ⚠️\n\n")
        
        if contradictions:
            f.write("## 確認矛盾詳細列表\n\n")
            for idx, contra in enumerate(contradictions, 1):
                f.write(f"### 矛盾 #{idx}\n\n")
                f.write(f"**位置**：句子 {contra.get('sent1_idx', 'N/A')} vs 句子 {contra.get('sent2_idx', 'N/A')}\n\n")
                f.write(f"**類型**：{contra.get('type', 'N/A')}\n\n")
                f.write(f"**嚴重性**：{contra.get('severity', 'N/A')}\n\n")
                f.write(f"**NLI 矛盾分數**：{contra.get('nli_contradiction_score', 'N/A')}\n\n")
                f.write(f"**分類解釋**：{contra.get('classification_explanation', 'N/A')}\n\n")
                f.write(f"**句子1（可信度 {contra['sent1'].get('credibility', 0)}%）**：\n")
                f.write(f"> {contra['sent1']['text']}\n\n")
                f.write(f"**句子2（可信度 {contra['sent2'].get('credibility', 0)}%）**：\n")
                f.write(f"> {contra['sent2']['text']}\n\n")
                
                # 詳細的 claim 驗證
                if 'claims_verified' in contra and contra['claims_verified']:
                    f.write(f"**驗證的聲明** ({len(contra['claims_verified'])} 個)：\n\n")
                    for claim_text, claim_info in contra['claims_verified'].items():
                        f.write(f"- **聲明**：{claim_text}\n")
                        f.write(f"  - LLM 置信度：{claim_info.get('llm_confidence', 'N/A')}\n")
                        f.write(f"  - 類型：{claim_info.get('claim_type', 'N/A')}\n")
                        f.write(f"  - 優先級：{claim_info.get('priority', 'N/A')}\n")
                        f.write(f"  - 推理：{claim_info.get('reasoning', 'N/A')}\n")
                        if claim_info.get('search_query'):
                            f.write(f"  - 搜尋關鍵字：{claim_info.get('search_query', 'N/A')}\n")
                        f.write("\n")
                
                f.write(f"**證據總結**：{contra.get('evidence_summary', 'N/A')}\n\n")
                f.write("---\n\n")
        else:
            f.write("## 確認矛盾\n\n未檢測到確認的邏輯矛盾。\n\n")
        
        # 顯示被排除的候選
        if excluded_candidates:
            f.write("## 被排除的候選矛盾\n\n")
            f.write(f"以下 {len(excluded_candidates)} 個候選在 LLM 分類步驟被認為「不是真實矛盾」而被排除。")
            f.write("可能是觀點差異或其他非事實型矛盾：\n\n")
            
            for idx, excluded in enumerate(excluded_candidates, 1):
                f.write(f"### 被排除候選 #{idx}\n\n")
                f.write(f"**位置**：句子 {excluded.get('i', 'N/A')} vs 句子 {excluded.get('j', 'N/A')}\n\n")
                f.write(f"**NLI 矛盾分數**：{excluded.get('nli_score', 'N/A'):.3f}\n\n")
                f.write(f"**分類結果**：{excluded.get('classification', {}).get('contradiction_type', 'N/A')}\n\n")
                f.write(f"**分類解釋**：{excluded.get('classification', {}).get('explanation', 'N/A')}\n\n")
                f.write(f"**排除原因**：{excluded.get('reason', 'N/A')}\n\n")
                f.write(f"**句子1**：\n")
                f.write(f"> {excluded['sent1']}\n\n")
                f.write(f"**句子2**：\n")
                f.write(f"> {excluded['sent2']}\n\n")
                f.write("---\n\n")
        
        f.write("## 檢測參數\n\n")
        f.write("- 矛盾分數閾值（NLI）：0.5\n")
        f.write("- 批量推理：已啟用（效能優化）\n")
        f.write("- Google 搜尋：未啟用\n")
    
    print("[Info] Report saved to nli_truth_report.md")
