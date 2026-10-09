import json
import re
from openai import OpenAI
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from sentence_transformers import SentenceTransformer
import numpy as np

# 嘗試載入 spaCy，若無則使用 Regex
try:
    import spacy
    # 優先嘗試載入中文模型
    try:
        nlp = spacy.load("zh_core_web_trf")
    except:
        try:
            nlp = spacy.load("zh_core_web_sm")
        except:
            nlp = None
except ImportError:
    nlp = None

client = OpenAI(api_key="")

embed_model = SentenceTransformer('all-MiniLM-L6-v2')

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# NLI 判斷函式
def nli_predict(premise: str, hypothesis: str):
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


# 將長文本切分成多個chunk，並對每個chunk進行NLI判斷，最後綜合結果
def split_into_token_chunks(text, tokenizer, max_tokens=512, overlap=50):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=False
    )

    input_ids = inputs["input_ids"][0]

    chunks = []
    start = 0

    while start < len(input_ids):
        end = start + max_tokens
        chunk_ids = input_ids[start:end]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk_text)

        start += max_tokens - overlap

    return chunks

# 用於取得chunk中的最佳證據句的函式
def split_sentences(text):
    if nlp:
        doc = nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    else:
        return [s.strip() for s in re.split(r'(?<=[。！？.!?])\s*', text) if s.strip()]

# token overlap ratio：計算兩段文本的token重疊程度，若過高則可能是原句，需進行清理
def token_overlap_ratio(a, b):
    a_tokens = set(a.split())
    b_tokens = set(b.split())

    overlap = len(a_tokens & b_tokens)
    return overlap / max(len(a_tokens), 1)



# 在 final_chunk 中找最支持的 sentence
def get_best_sentence(final_chunk, hypothesis_text):
    sentences = split_sentences(final_chunk)

    best_score = -1
    best_sentence = None
    best_result = None

    for sent in sentences:
        
        res = nli_predict(sent, hypothesis_text)
        # 以 entailment 或 contradiction 最高分作為判斷依據
        score = max(res['prob_distribution']['entailment'], res['prob_distribution']['contradiction'])
        if score > best_score:
            best_score = score
            best_sentence = sent
            best_result = res

    return best_sentence, best_result

# 滑動窗口NLI：將文章切成多個chunk，對每個chunk與假設進行NLI判斷，找出最強的entailment和contradiction作為最終判斷依據(best score, best proven sentence)
def sliding_window_nli(chunks, hypothesis):

    best_entailment_score = -1
    best_entailment_chunk = None
    best_entailment_result = None

    best_contradiction_score = -1
    best_contradiction_chunk = None
    best_contradiction_result = None

    for chunk in chunks:
        result = nli_predict(chunk, hypothesis)

        entail_score = result["prob_distribution"]["entailment"]
        contra_score = result["prob_distribution"]["contradiction"]

        # 追蹤最佳 entailment
        if entail_score > best_entailment_score:
            best_entailment_score = entail_score
            best_entailment_chunk = chunk
            best_entailment_result = result

        # 追蹤最佳 contradiction
        if contra_score > best_contradiction_score:
            best_contradiction_score = contra_score
            best_contradiction_chunk = chunk
            best_contradiction_result = result

    # 🔥 最終決策規則
    if best_contradiction_score > 0.7:
        final_label = "contradiction"
        final_chunk = best_contradiction_chunk
        final_result = best_contradiction_result

    elif best_entailment_score > 0.7:
        final_label = "entailment"
        final_chunk = best_entailment_chunk
        final_result = best_entailment_result

    else:
        final_label = "neutral"
        final_chunk = best_entailment_chunk
        final_result = best_entailment_result

    return {
        "final_label": final_label,
        "best_premise_chunk": final_chunk,
        "verification_result": final_result,
        "best_entailment_score": best_entailment_score,
        "best_contradiction_score": best_contradiction_score
    }

# ========================
# 清除 + chunk 函式
# ========================
def prepare_chunks_for_hypothesis(
    original_sentences,
    hyp_text,
    tokenizer,
    embed_model=None,
    sentence_embeddings=None,
    max_tokens=512,
    overlap=50,
    overlap_threshold=0.8,
    sim_threshold=0.92
):
    """
    為單一 hypothesis 建立專屬 cleaned text 並做 chunk。

    Parameters
    ----------
    original_sentences : List[str]
        已 split 的全文句子（不要每次重 split）
    hyp_text : str
        當前 hypothesis 文字
    tokenizer :
        用於 chunk 的 tokenizer
    embed_model :
        sentence transformer 模型（可選）
    sentence_embeddings : List[np.array]
        預先計算好的 sentence embeddings（可選）
    """

    filtered_sentences = []

    # 若使用 embedding，先算 hypothesis embedding
    if embed_model is not None:
        hyp_embedding = embed_model.encode(hyp_text)

    for idx, sent in enumerate(original_sentences):

        # 1️⃣ 排除完全相同
        if sent.strip() == hyp_text.strip():
            continue

        # 2️⃣ lexical overlap 過濾
        if token_overlap_ratio(sent, hyp_text) > overlap_threshold:
            continue

        # 3️⃣ semantic similarity 過濾（可選）
        if embed_model is not None and sentence_embeddings is not None:
            sim = cosine_sim(sentence_embeddings[idx], hyp_embedding)
            if sim > sim_threshold:
                continue

        filtered_sentences.append(sent)

    cleaned_text = " ".join(filtered_sentences)

    # 4️⃣ chunk
    chunks = split_into_token_chunks(
        cleaned_text,
        tokenizer,
        max_tokens=max_tokens,
        overlap=overlap
    )

    return chunks


# 整合並執行新的英文NLI驗證流程
def run_english_nli_pipeline(article: str):
    """
    執行基於英文NLI模型的事實核查流程。
    該流程依賴一次GPT呼叫來同時生成英文前提(Premise)和多個英文假設(Hypotheses)。
    """
    
    # 1. 設計一個強大的Prompt，指示LLM執行多任務並返回JSON
    prompt = f"""
You are an analytical assistant for fact-checking. Your task is to process a Traditional Chinese text, create a concise English summary (premise), extract its core factual claims, and translate those claims into English (hypotheses).

Here is the Traditional Chinese text:
---
{article}
---

Please perform the following actions and provide the output ONLY in a single, valid JSON object:
1. Full Translation (premise_en):
Translate the entire Chinese article into fluent and faithful English.
Do NOT summarize, compress, reinterpret, or omit any information.
Preserve all numerical values, dates, percentages, quantities, attributions, and hedging expressions exactly as stated.
The translation must retain the original meaning, tone, and logical structure.
This will be the value for the key "premise_en".
-------------------------------------------------------
2. Extract and Translate Key Claims (hypotheses_en):
Identify 1 to 4 central statements explicitly presented in the original Chinese text.
Each statement must:
- Be directly stated or clearly implied in the article
- Represent a concrete assertion about events, actions, data, or outcomes
- Preserve all numerical details, dates, and attributions if present
- Avoid vague generalizations or purely subjective commentary
Do NOT evaluate whether the statements are true or false.
Do NOT perform verification.
Simply extract the claims as written.
Translate each selected statement into clear and faithful English.
Return them as a list of objects under the key "hypotheses_en".
-------------------------------------------------------
Your JSON output must be in the following format:
{{
  "premise_en": "<Full English translation of the article>",
  "hypotheses_en": [
    {{
      "id": 1,
      "text": "<The first verifiable English factual statement>"
    }},
    {{
      "id": 2,
      "text": "<The second verifiable English factual statement>"
    }}
  ]
}}
"""

    print("--- Sending request to LLM ---")
    # 2. 進行單次GPT呼叫
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # 使用更強大的模型以確保遵循指令
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}, # 啟用JSON模式
        )
        content = response.choices[0].message.content
        nli_data = json.loads(content)
        
        premise_en = nli_data.get("premise_en")
        hypotheses_en = nli_data.get("hypotheses_en")

        if not (premise_en and hypotheses_en):
            raise ValueError("LLM response did not contain required 'premise_en' or 'hypotheses_en' keys.")

        print("LLM Response Parsed Successfully.")
        print(f"Premise: {premise_en}")
        print("---")

    except Exception as e:
        print(f"Error processing LLM response: {e}")
        return []

    original_sentences = split_sentences(premise_en)

    # 若使用 embedding，先算好 sentence embeddings（加速）
    sentence_embeddings = embed_model.encode(original_sentences)
    
    # 3. 遍歷每一個假設，並與前提進行NLI判斷
    results = []

    for hyp in hypotheses_en:
        hyp_text = hyp.get("text")
        hyp_id = hyp.get("id")

        chunks = prepare_chunks_for_hypothesis(
            original_sentences=original_sentences,
            hyp_text=hyp_text,
            tokenizer=tokenizer,
            embed_model=embed_model,
            sentence_embeddings=sentence_embeddings
        )

        nli_result = sliding_window_nli(chunks, hyp_text) # will get best chunk
        
        # 從 best chunk 找支撐句
        best_sentence, best_result = get_best_sentence(nli_result['best_premise_chunk'], hyp_text)

        results.append({
            "hypothesis_id": hyp_id,
            "hypothesis_text": hyp_text,
            "nli_label": nli_result["final_label"],
            "best_supporting_sentence": best_sentence,
            "entailment_score": nli_result["best_entailment_score"],
            "contradiction_score": nli_result["best_contradiction_score"],
            "best_supporting_result": best_result, 
        })

    return results

if __name__ == "__main__": 
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    MODEL_NAME = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli" # 保持使用多語言模型，因為它很強大
    # 若要切換為純英文模型，可改為 "roberta-large-mnli" 或 "bart-large-mnli"
    # MODEL_NAME = "roberta-large-mnli"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    model.eval()

    # labels = ["contradiction", "neutral", "entailment"]
    
    # 待驗證文章
    article = ["""
    政府近日推出新型教育政策，聲稱將提升全國學生的學習成效。然而，多家媒體報導指出，部分學校反映，課程安排增加了教師負擔，使教學壓力上升。有評論認為，學生的學習表現仍維持原有水準，並未出現明顯改善。另一方面，有教育專家表示，政策初期效果不明，但政策改革可能導致學校資源分配不均，影響部分地區學生的學習環境。部分家長則認為，新的課程設計讓孩子接觸更多活動，但也可能降低基礎知識的累積速度。整體而言，政策實施後，學生表現波動仍存在，政策效果仍需時間觀察。
    """
    ]
    # 執行新的NLI流程
    results = []
    for i, art in enumerate(article):
        print(f"Processing article {i+1}")
        results.extend(run_english_nli_pipeline(art))

    output_str = json.dumps(results, indent=2, ensure_ascii=False)
    print("\n--- Final Verification Results ---")
    print(output_str)

    # Save to log file
    with open("nli_log2.md", "w", encoding="utf-8") as f:
        f.write("# NLI Analysis Log (English Pipeline)\n\n")
        print("=============================")
        f.write(output_str)    
    print("\n[Info] Results have been saved to nli_log2.md")

    # Testing
    # premise = "The government launched a new education policy."
    # hypothesis = "A new education policy was introduced by the government."

    # print(nli_predict(premise, hypothesis))