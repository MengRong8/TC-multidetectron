import os
import re
import json
import time
import statistics
from typing import List, Dict
from openai import OpenAI


# -----------------------------
#  初始化 OpenAI
# -----------------------------
API_KEY = os.environ.get("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("請在環境變數中設定 OPENAI_API_KEY")

client = OpenAI(api_key=API_KEY)

# -----------------------------
# OpenAI 呼叫函式
# -----------------------------
def call_openai(prompt, model="gpt-5-mini", retry=5):
    for attempt in range(retry):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            wait_sec = 2 ** attempt
            print(f"OpenAI 呼叫失敗，嘗試重試 {attempt+1}，等待 {wait_sec} 秒… 錯誤訊息：{e}")
            time.sleep(wait_sec)
    print("無法獲得回覆，跳過此筆")
    return
                  

# -----------------------------
#  砍掉廣告新聞、無訓練作用新聞 Prompt
# -----------------------------
def filter_ad_prompt(title, content):
    return f"""
你是一位資深新聞檢查編輯。  
請閱讀以下新聞標題與內容，判斷是否是以廣告為主的新聞，介紹商品為主要內容的新聞。
不要只看標題或片段，請完整閱讀全文後再做判斷。  
如果是廣告新聞或無法作為訓練資料，請回覆 單個數字 1，否則回覆 單個數字 0。
【新聞標題】：
{title}
【新聞全文】：
{content}
"""

# -----------------------------
#  廣告判斷
# -----------------------------
def is_ad_news(response_text: str) -> bool:
    """
    根據模型回覆，穩定判定是否為廣告新聞。
    回覆中第一個 0 或 1 即視為結果。
    """
    if not response_text:
        return True  # 沒回覆 → 當成不可用資料
    
    # 全形 → 半形
    response_text = response_text.replace("０", "0").replace("１", "1")

    # 搜尋第一個出現的 0 或 1
    m = re.search(r"[01]", response_text)
    if not m:
        return True  # 沒找到 → 保守認定為無法訓練

    return m.group(0) == "1"  # True = 廣告 / 不可訓練

# -----------------------------
#  摘要 Prompt
# -----------------------------
def summarize_prompt(content, target_len):
    return f"""
你是一位精準新聞摘要模型。  
請將以下新聞內容摘要成大約 {target_len} 字的文本。  

摘要必須：  
- 完全保留資訊、事實、時間、地點、人物  
- 不必保留句子結構  
- 不要模仿原文  
- 用你自己的方式重寫  
- 不加入新資訊、不評論、不解讀  
- 僅保留核心敘述 
- 某些()括號內容可適度保留補充說明，像是英文人名、重要資訊等 

請產生「純摘要文字」，不要加標題、不加額外說明。

【新聞全文】：
{content}
"""


# -----------------------------
#  擴寫 Prompt
# -----------------------------
def rewrite_prompt(summary, target_len):
    return f"""
請根據以下摘要，重新撰寫一篇完整新聞稿。  

要求：
- 長度約為 {target_len} 字（±10%）
- 不照抄原文或摘要任何句子
- 使用你自己的句型與表達方式
- 保留所有事實，但可以調整敘事順序
- 適度補充背景，使新聞敘事完整
- 不加入不存在的人名、資料或事件

格式要求：
- 可以適度加入括號（）、補註性說明，例如（資料來源）、（日期）、（地名）、（示意）、（補充資訊）等，但不需固定出現
- 中文名若是英文翻譯，則在名字後附上(英文人名)
- 可以使用自然的時間或背景補充語，但不可虛構

輸出為純新聞文本，不要加標題、不加說明。

【摘要】：
{summary}
"""


# -----------------------------
#  Main generator
# -----------------------------
def generate_aigc_news(input_path, output_path, sample_size=1000):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # test = [ # 測試用資料集
    #       {
    #         "id": 1015,
    #         "category": "財經",
    #         "url": "https://www.chinatimes.com/newspapers/20200117000313-260202?chdtv",
    #         "title": "加速危老重建 容積獎勵再延五年",
    #         "content": "內政部部務會報16日通過《危老條例》修正草案，將再延長限時容積獎勵5年，但從第4年起（即今年5月10日起）限時獎勵減半為5％，每年遞減一個百分點，到第9年全部歸零；另新增規模容積獎勵，約130坪（400平方公尺）、相當兩個雙併公寓，可獲2％規模容積獎勵，採雙軌制，但限時及規模獎勵兩者合計不得逾基準容積10％，鼓勵更多房屋進行危老重建。\n依現行危老條例，提供前3年申請危老重建者10％限時容獎，這項優惠今年5月9日屆期，業者早在1年前就不斷呼籲內政部修法延長限時容積獎勵。\n內政部認為，若一次性取消限時容獎，可能大幅削減民眾申請的意願，因此採取逐年遞減限時容獎的方式處理。\n其次，新的危老條例中增加規模獎勵，增訂危老建築基地加計合併鄰地面積（加計之鄰地面積，不得超過危老建築物基地面積或1,000公尺）達到400平方公尺者，給予2％的容積獎勵，每增加100平方公尺，再給予0.5％的容獎。但是時程與規模容積獎勵，兩者合計不得超過基準容積的10％，這與現行限時獎勵的額度上限相同。\n據內政部統計，約有39.45％危老重建案申請限時容積獎勵及危險建築獎勵，顯示民眾對限時容獎有相當需求。另約有51.83％案件，是在400平方公尺以下，且小面積的基地案件數量有越來越多趨勢，可能造成改建基地太過零星化的問題，因此本次修法時，特別提出規模獎勵，鼓勵大面積基地申請危老重建。\n花敬群說，達到400平方公尺，約130坪的案子，就可以獲得規模獎勵，實際上來看，就是兩個雙併公寓的規模，規模越大、獎勵越多。\n此外，這次修法另一大亮點是，將鄰地合併面積鬆綁，鼓勵擴大重建。現行鄰地超過危老建物基地面積部分不得合併重建，使得重建效益受限，或間接造成畸零地或地籍零碎，這次刪除合併鄰地面積限制，允許合併，惟危老小基地併鄰地大基地，不能享受限時和規模容獎；至於危老大基地併鄰地小基地，則可以享受容獎，但鄰地最大不能逾1,000平方公尺。\n危老重建條例修正案是蔡總統連任後推動第一個財經政策法案，花敬群說，部務會報通過後，還要送交行政院審查，可能要等春節後行政院會才會通過，並送交立法院審議。他希望這項福國利民的修法，能在5月9日限時容獎到期前完成三讀，順利接軌。",
    #         "label": 0,
    #         "image_path": "./chinatime_news/加速危老重建 容積獎.jpg"
    #     },
    #     {
    #         "id": 1054,
    #         "category": "財經",
    #         "url": "https://www.chinatimes.com/newspapers/20200229001686-260511?chdtv",
    #         "title": "拉亞漢堡 打造最「罩」早午餐",
    #         "content": "無畏新冠肺炎疫情持續延燒，森邦集團旗下知名拉亞漢堡於2月28日至3月2日舉辦的台北國際連鎖加盟春季展，以強而有力的總部支援，運用多元活潑的行銷宣傳，幫助加盟主面對這波疫情仍能業績穩定成長，成為「開店一把罩、創業金鐘罩」早午餐品牌。\n搶攻早午餐商機，拉亞漢堡推出「50萬零利率、開店成功加碼再送市價超過10萬元全自動咖啡機」專案，鼓勵有開店夢想但資金不寬裕的年輕族群，能安心創業、勇敢追夢！\n因應這波疫情，拉亞漢堡總部第一時間嚴格要求夥伴們加強環境清潔消毒，食材配送及製餐過程全程均須戴口罩，店內張貼防疫宣導公告，再搭配即將推出的外帶優惠活動及安心保證貼紙，鼓勵消費者外帶可提升門市營業額，讓加盟主免除後顧之憂，安心且專注經營門市，並再創美好佳績。\n總部長期致力新品研發及行銷活動，創造品牌差異化與提升消費者心中好感度，進而帶動加盟主的銷售業績。如2020年初以「絕色玩味」為主題，搭配美味色票「LAYATONE」推出的全新餐點，如起司芋泥黃金堡、鹹蛋黃流沙、韓式泡菜、抹茶系列飲品，皆大獲好評。",
    #         "label": 0,
    #         "image_path": "./chinatime_news/拉亞漢堡 打造最「罩.jpg"
    #     },
    #     {
    #         "id": 1055,
    #         "category": "財經",
    #         "url": "https://www.chinatimes.com/realtimenews/20200214004828-260410?chdtv",
    #         "title": "兆豐銀四大宅經濟優惠登場 最高16％回饋",
    #         "content": "新冠肺炎疫情持續延燒，民眾出門購物意願降低，行政院於2月13日拍板發放「振興抵用券」刺激消費。腦筋動得快的銀行業者嗅出商機，目前包括兆豐及台新等銀行皆已發動宅經濟刷卡優惠應戰，其中台新以網購、電玩及外送最高6％現金回饋搶市，而日前創金融業先例，率先推出因應疫情可申請卡費緩繳最長三個月的兆豐銀行為響應政府政策，繼推出7天有薪防疫照顧假及理財e把兆機器人服務，再加碼針對四大類宅經濟消費，一舉送出最高16％現金回饋，包括海內外網購、六大影音、外送平台與線上量販，讓民眾免出門也能在家輕鬆大賺回饋。\n網購業者統計，近來宅配量大增，包括居家食品與運動用品，成長幅度均逾二成，顯示民眾即使不出門，也會透過線上購物，在家打造舒適的娛樂休閒環境。兆豐銀行針對其網購神卡e秒刷推出加碼活動，原海外網購最高4％回饋，每逢周一再加碼6％，總回饋將直衝10％之高；國內則與Yahoo!購物中心、momo、PChome、蝦皮及松果合作滿額活動，最高回饋高達16％。台新則推指定卡國外網購及訂房網滿萬享3％回饋，惟須注意該行網購卡@GoGo與FlyGo卡都無法參加加碼活動。\n在家休閒娛樂部分，各家銀行也推出眾多追劇優惠讓民眾盡情宅在家看劇，如兆豐銀行針對六大影視平台，推出為期三個月的限時加碼10％回饋活動，包括愛奇藝、Netflix、CATCHPLAY等，只要刷兆豐卡，登錄即可獲得10％回饋。華南則針對i網購卡推出指定線上娛樂等通路，送出8％回饋；國泰世華則針對影音娛樂平台送出7％回饋。\n至於飲食部分同樣有刷卡優惠可拿，想吃熱食的話，可以考量外送平台優惠，如兆豐便與foodpanda及Uber Eats合作，只要刷兆豐卡，最高可獲得53％回饋。如果需要採購量販生鮮用品，也可利用家樂福線上購物，刷兆豐卡單筆滿千送60元刷卡金。\n除了祭出最高16％現金回饋活動外，兆豐銀行為善盡企業社會責任，即日起再為應主管機關要求居家隔離的卡友提供延遲繳款信用卡帳單30天服務，並將申請期間延長至5月底，一同為防疫盡一份心力，與民眾共同度過非常時期。",
    #         "label": 0,
    #         "image_path": "./chinatime_news/兆豐銀四大宅經濟優惠.jpg"
    #     },
    #     {
    #         "id": 1067,
    #         "category": "財經",
    #         "url": "https://www.chinatimes.com/newspapers/20200216000211-260202?chdtv",
    #         "title": "工總：電子業3月初恐斷鏈",
    #         "content": "經濟部正盤點新冠肺炎對台灣製造業衝擊，工業總會針對旗個產業公會及會員代表廠商進行全面調查結果，15日正式公布，發現製造業面臨供應鏈斷鏈、產線停擺、訂單減少，及資金周轉困難四大問題，廠商評估，3月初電子業就可能出現斷鏈，希望政府提高600億紓困金額，並對製造業強化資金紓困，解決斷鏈、斷料等問題，穩定台灣經濟發展。\n政院高層15日透露，經濟部正加速研議對製造業的紓困方案，包含提供服務業的三項資金紓困措施，協助舊有貸款展延利息補貼減免、新增貸款補貼利息及向銀行融資周轉，提供十足保證與補貼利息，免保證手續費等，都比照服務業，只是大企業無法享受利息補貼，這部分服務業和製造業補貼利息的經費共計30.8億元，而100億元保證專款，以十倍保證祭出防疫千億保證額度，製造業也適用。\n除此，行政院15日繼續討論防疫紓困特別條例草案，勞動部提出給予居家檢疫、居家隔離者適當補償；財政部則祭出租稅優惠，企業支薪給防疫照顧假員工增加的薪資成本可「加倍」列為費用，隔年申報營所稅時減除。\n工總表示，雖然台灣生產線持續運作，但因產品原物料需從大陸進口，受停工影響，目前已面臨原料短缺問題。目前大陸復工率最多三成，許多廠商認為，2月底前難以恢復生產。製造業最先需要政府紓困的是資金問題。短期部分，行政院匡列600億特別預算，但政府應考慮這次疫情衝擊較SARS時期更全面，提高紓困金額。值得注意的是，此次政府產業紓困方案多集中在五大項內需服務業，對製造業的紓困金額嚴重不足，也要強化對製造業的紓困金額。\n中長期則包括貸款額度提高、還款期限展延、利息補貼等。此外，應考慮房屋稅、地價稅、營所稅等稅負優惠或減免；在營所稅申報上，准予企業認列疫情產生的損失。此外，比照SARS紓困政策，今年到期的貸款延後一年還款寬限期。\n針對原物料及零組件需從中國大陸進口短缺問題，建議政府快速掌握各產業別對原物料的需求狀況，解決物流問題，並協助廠商找尋可替代的原物料來源。中長期則希望降低部分進口產品關稅，讓廠商降低成本及增加競爭力。進一步協助廠商在台灣生產關鍵零組件，建議政府盤點需協助的產業別，以專案處理的方式簡化重大共通零組件。\n此外，為因應疫情，有些企業會將產能轉移回台灣。因此，政府應在彈性工時、派遣比重、外勞比重、加班等政策上適度放寬，以協助廠商提高在台產能利用率，擴大轉單效益。",
    #         "label": 0,
    #         "image_path": "./chinatime_news/工總：電子業3月初恐.jpg"
    #     },
    # ]

    output = []
    for idx, item in enumerate(data[:sample_size]):   # 只取前 sample_size 筆
        print(f"正在處理第 {idx+1} 筆…")

        content = item["content"].strip()
        title = item["title"].strip()
        orig_len = len(content)

        # ======= 廣告新聞過濾 =======
        filter_pmt = filter_ad_prompt(title, content)
        filter_result = call_openai(filter_pmt)
        if is_ad_news(filter_result):
            rewritten = ""
            print(f"⚠️ 第 {idx+1} 筆被判定為廣告新聞，跳過生成。")
            new_item = {
                "id": item["id"],
                "category": item["category"],
                "url": item["url"],
                "title": item["title"],
                "content": rewritten,
                "label": 1,           # AIGC
                "image_path": ""      # 空
            }
            output.append(new_item)
            
            # 儲存每筆生成文本資料
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)
        else:
            #   長文本處理（content > 2000字）
            if orig_len > 2100:
                print("偵測到長文本，啟用長文本模式處理…")

                # Clip 避免 GPT 遺忘後半段
                clipped_content = content[:1800]

                # 建議長文摘要長度比率：35%
                target_summary_len = max(120, int(orig_len * 0.35))

                # Step 1: 摘要
                summary_prompt = summarize_prompt(clipped_content, target_summary_len)
                summary = call_openai(summary_prompt)

                # Step 2: 擴寫成固定中長度（約 1000 字）
                target_rewrite_len = 1500
                rewrite_pmt = rewrite_prompt(summary, target_rewrite_len)
                rewritten = call_openai(rewrite_pmt)
            else: 
                # ======= Step 1: 摘要（25% 長度）======
                target_summary_len = max(60, orig_len // 4)
                summary_prompt = summarize_prompt(content, target_summary_len)
                summary = call_openai(summary_prompt)

                # ======= Step 2: 擴寫 =======
                target_rewrite_len = orig_len
                rewrite_pmt = rewrite_prompt(summary, target_rewrite_len)
                rewritten = call_openai(rewrite_pmt)

            # ======= Step 3: 建立新資料 =======
            new_item = {
                "id": item["id"],
                "category": item["category"],
                "url": item["url"],
                "title": item["title"],
                "content": rewritten,
                "label": 1,           # AIGC
                "image_path": ""      # 空
            }
            output.append(new_item)
            # 儲存每筆生成文本資料
            print(f"第 {idx+1} 筆擴寫，進行生成文本暫存...")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)
            

    # ======= 寫出結果 =======
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"已完成，輸出檔案：{output_path}")


# -----------------------------
#  執行
# -----------------------------
if __name__ == "__main__":
    generate_aigc_news(
        input_path="chinatime_cleaned2.json",
        output_path="chinatime_fake.json",
        sample_size=1000
    )

# def analyze_lengths(json_path: str, top_n: int = 10):
#     with open(json_path, "r", encoding="utf-8") as f:
#         data = json.load(f)

#     lengths = []
#     records = []

#     for item in data:
#         content = item.get("content", "") or ""
#         text = content.replace("\n", "").strip()
#         length = len(text)

#         lengths.append(length)
#         records.append({
#             "id": item.get("id"),
#             "length": length,
#             "content": text
#         })

#     # 基本統計
#     stats = {
#         "count": len(lengths),
#         "min": min(lengths),
#         "max": max(lengths),
#         "mean": round(statistics.mean(lengths), 2),
#         "median": statistics.median(lengths),
#         "stdev": round(statistics.stdev(lengths), 2) if len(lengths) > 1 else 0
#     }

#     # 取最短、最長
#     shortest = sorted(records, key=lambda x: x["length"])[:top_n]
#     longest = sorted(records, key=lambda x: x["length"], reverse=True)[:top_n]

#     return stats, shortest, longest

# import json
# from collections import defaultdict
# import math

# def analyze_length_distribution(json_path):
#     with open(json_path, "r", encoding="utf-8") as f:
#         data = json.load(f)

#     # 建立分布區間
#     bins = defaultdict(int)

#     for item in data:
#         text = item.get("content", "")
#         length = len(text)

#         # 只計算 0~3000 之間
#         if length <= 3000:
#             bin_index = math.floor(length / 100)  # 100 字為一區
#             bins[bin_index] += 1
#         else:
#             bins[30] += 1  # 超過 3000 字的歸為最後一區

#     # 依區間排序輸出
#     for i in range(1, 31):
#         start = (i - 1) * 100
#         end = i * 100
#         count = bins.get(i, 0)
#         print(f"{start:4d}-{end:4d} 字： {count} 篇")


# if __name__ == "__main__":
#     stats, shortest, longest = analyze_lengths("ltn_cleaned2.json")
    
#     print("=== 基本統計 ===")
#     print(stats)
    
#     print("\n=== 長度分布 ===")
#     analyze_length_distribution("ltn_cleaned2.json")



