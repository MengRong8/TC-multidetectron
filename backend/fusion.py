"""
模擬決策融合層和最終決策器S
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DecisionFusionLayer(nn.Module):
    """
    線性融合層
    將三個模型的分數融合為最終分數

    公式: final_score = w1 * text_aigc + w2 * text_intent + w3 * image_aigc + bias
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        bias: float = 0.0,
        learnable: bool = False
    ):
        """
        Args:
            weights: 權重字典 {'text_aigc': w1, 'text_intent': w2, 'image_aigc': w3}
            bias: 偏置項
            learnable: 是否作為可學習參數（用於訓練）
        """
        super().__init__()

        # 默認權重（均等）
        if weights is None:
            weights = {
                'text_aigc': 0.4,      # RoBERTa 權重
                'text_intent': 0.3,    # CB-LLM 權重
                'image_aigc': 0.3      # CBM 權重
            }

        self.weight_names = ['text_aigc', 'text_intent', 'image_aigc']

        if learnable:
            # 可學習參數
            self.weights = nn.Parameter(torch.tensor([
                weights['text_aigc'],
                weights['text_intent'],
                weights['image_aigc']
            ], dtype=torch.float32))
            self.bias = nn.Parameter(torch.tensor(bias, dtype=torch.float32))
        else:
            # 固定參數
            self.register_buffer('weights', torch.tensor([
                weights['text_aigc'],
                weights['text_intent'],
                weights['image_aigc']
            ], dtype=torch.float32))
            self.register_buffer('bias', torch.tensor(bias, dtype=torch.float32))

        logger.info(f"DecisionFusionLayer initialized with weights: {weights}, bias: {bias}")

    def forward(
        self,
        text_aigc_score: Optional[float],
        text_intent_score: Optional[float],
        image_aigc_score: Optional[float]
    ) -> torch.Tensor:
        """
        前向傳播

        Args:
            text_aigc_score: Text AIGC 分數 (0-1)
            text_intent_score: Text Intent 分數 (0-1)
            image_aigc_score: Image AIGC 分數 (0-1)

        Returns:
            final_score: 最終分數 (0-1)
        """
        # 處理缺失值
        scores = []
        active_weights = []

        for i, (score, name) in enumerate([
            (text_aigc_score, 'text_aigc'),
            (text_intent_score, 'text_intent'),
            (image_aigc_score, 'image_aigc')
        ]):
            if score is not None:
                scores.append(score)
                active_weights.append(self.weights[i])

        if not scores:
            # 所有模型都失敗了
            return torch.tensor(0.5)  # 返回中立分數

        # 轉換為 tensor
        scores_tensor = torch.tensor(scores, dtype=torch.float32)
        weights_tensor = torch.stack(active_weights)

        # 歸一化權重
        weights_normalized = weights_tensor / weights_tensor.sum()

        # 加權求和
        final_score = (scores_tensor * weights_normalized).sum() + self.bias

        # Clamp 到 [0, 1]
        final_score = torch.clamp(final_score, 0.0, 1.0)

        return final_score

    def get_weights(self) -> Dict[str, float]:
        """獲取當前權重"""
        return {
            name: self.weights[i].item()
            for i, name in enumerate(self.weight_names)
        }


class FusionStrategy:
    """
    融合策略類
    提供多種融合方法
    """

    @staticmethod
    def linear_fusion(
        scores: Dict[str, Optional[float]],
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        線性加權融合

        Args:
            scores: {'text_aigc': 0.8, 'text_intent': 0.6, 'image_aigc': 0.9}
            weights: {'text_aigc': 0.4, 'text_intent': 0.3, 'image_aigc': 0.3}

        Returns:
            final_score: float (0-1)
        """
        if weights is None:
            weights = {
                'text_aigc': 0.4,
                'text_intent': 0.3,
                'image_aigc': 0.3
            }

        # 過濾有效分數
        valid_scores = {k: v for k, v in scores.items() if v is not None}

        if not valid_scores:
            return 0.5  # 默認中立分數

        # 歸一化權重（只考慮有效模型）
        total_weight = sum(weights[k] for k in valid_scores.keys() if k in weights)
        normalized_weights = {
            k: weights[k] / total_weight
            for k in valid_scores.keys()
            if k in weights
        }

        # 加權求和
        final_score = sum(
            valid_scores[k] * normalized_weights[k]
            for k in valid_scores.keys()
            if k in normalized_weights
        )

        return float(np.clip(final_score, 0.0, 1.0))

    @staticmethod
    def max_fusion(scores: Dict[str, Optional[float]]) -> float:
        """
        最大值融合（保守策略：任意一個模型認為是假新聞）
        """
        valid_scores = [v for v in scores.values() if v is not None]
        if not valid_scores:
            return 0.5
        return float(max(valid_scores))

    @staticmethod
    def min_fusion(scores: Dict[str, Optional[float]]) -> float:
        """
        最小值融合（樂觀策略：所有模型都認為是假新聞才判定）
        """
        valid_scores = [v for v in scores.values() if v is not None]
        if not valid_scores:
            return 0.5
        return float(min(valid_scores))

    @staticmethod
    def avg_fusion(scores: Dict[str, Optional[float]]) -> float:
        """
        平均值融合（均等權重）
        """
        valid_scores = [v for v in scores.values() if v is not None]
        if not valid_scores:
            return 0.5
        return float(np.mean(valid_scores))

    @staticmethod
    def voting_fusion(
        scores: Dict[str, Optional[float]],
        threshold: float = 0.5
    ) -> float:
        """
        投票融合（多數決策）

        Args:
            scores: 各模型分數
            threshold: 判定閾值

        Returns:
            1.0 如果多數認為是假新聞，否則 0.0
        """
        valid_scores = [v for v in scores.values() if v is not None]
        if not valid_scores:
            return 0.5

        # 統計投票
        fake_votes = sum(1 for score in valid_scores if score > threshold)
        real_votes = len(valid_scores) - fake_votes

        if fake_votes > real_votes:
            return 1.0
        elif real_votes > fake_votes:
            return 0.0
        else:
            return 0.5  # 平局


class DecisionMaker:
    """
    最終決策器
    整合模型預測結果並生成最終報告
    """

    def __init__(
        self,
        fusion_method: str = 'linear',
        weights: Optional[Dict[str, float]] = None,
        threshold: float = 0.5
    ):
        """
        Args:
            fusion_method: 融合方法 ('linear', 'max', 'min', 'avg', 'voting')
            weights: 線性融合的權重（僅當 fusion_method='linear' 時使用）
            threshold: 判定閾值
        """
        self.fusion_method = fusion_method
        self.weights = weights or {
            'text_aigc': 0.4,
            'text_intent': 0.3,
            'image_aigc': 0.3
        }
        self.threshold = threshold
        self.strategy = FusionStrategy()

        logger.info(f"DecisionMaker initialized with method={fusion_method}, threshold={threshold}")

    def make_decision(
        self,
        model_results: Dict[str, Optional[Dict]]
    ) -> Dict:
        """
        做出最終決策

        Args:
            model_results: {
                'text_aigc': {'score': 0.8, 'confidence': 0.9, 'label': 'AIGC'},
                'text_intent': {'score': 0.6, 'confidence': 0.7, 'label': 'Manipulative'},
                'image_aigc': {'score': 0.9, 'confidence': 0.85, 'label': 'AI-generated'}
            }

        Returns:
            {
                'final_score': float (0-1),
                'final_label': str,
                'final_confidence': float,
                'method': str,
                'individual_scores': dict,
                'report': str
            }
        """
        # 提取分數
        scores = {}
        for model_name, result in model_results.items():
            if result is not None and 'score' in result:
                scores[model_name] = result['score']
            else:
                scores[model_name] = None

        # 選擇融合方法
        if self.fusion_method == 'linear':
            final_score = self.strategy.linear_fusion(scores, self.weights)
        elif self.fusion_method == 'max':
            final_score = self.strategy.max_fusion(scores)
        elif self.fusion_method == 'min':
            final_score = self.strategy.min_fusion(scores)
        elif self.fusion_method == 'avg':
            final_score = self.strategy.avg_fusion(scores)
        elif self.fusion_method == 'voting':
            final_score = self.strategy.voting_fusion(scores, self.threshold)
        else:
            logger.warning(f"Unknown fusion method: {self.fusion_method}, using 'avg'")
            final_score = self.strategy.avg_fusion(scores)

        # 判定標籤
        if final_score > 0.66:
            final_label = "Highly Likely Fake"
            severity = "high"
        elif final_score > 0.5:
            final_label = "Possibly Fake"
            severity = "medium"
        elif final_score > 0.33:
            final_label = "Uncertain"
            severity = "low"
        else:
            final_label = "Likely Real"
            severity = "minimal"

        # 計算置信度
        final_confidence = abs(final_score - 0.5) * 2  # 距離 0.5 越遠，confidence 越高

        # 生成報告
        report = self._generate_report(model_results, final_score, final_label, severity)

        return {
            'final_score': float(final_score),
            'final_label': final_label,
            'final_confidence': float(final_confidence),
            'method': self.fusion_method,
            'individual_scores': scores,
            'report': report
        }

    def _generate_report(
        self,
        model_results: Dict[str, Optional[Dict]],
        final_score: float,
        final_label: str,
        severity: str
    ) -> str:
        """生成分析報告"""
        lines = []

        lines.append(f"**Final Verdict: {final_label}**")
        lines.append(f"**Overall Score: {final_score:.2f}** (0=Real, 1=Fake)")
        lines.append("")
        lines.append("### Individual Model Assessments:")
        lines.append("")

        # Text AIGC
        if model_results.get('text_aigc'):
            result = model_results['text_aigc']
            lines.append(f"**1. Text AIGC Detection (RoBERTa):**")
            lines.append(f"   - Score: {result.get('score', 0):.2f}")
            lines.append(f"   - Label: {result.get('label', 'Unknown')}")
            lines.append(f"   - Confidence: {result.get('confidence', 0):.2f}")
            lines.append("")

        # Text Intent
        if model_results.get('text_intent'):
            result = model_results['text_intent']
            lines.append(f"**2. Text Intent Detection (CB-LLM):**")
            lines.append(f"   - Score: {result.get('score', 0):.2f}")
            lines.append(f"   - Label: {result.get('label', 'Unknown')}")
            lines.append(f"   - Confidence: {result.get('confidence', 0):.2f}")
            lines.append("")

        # Image AIGC
        if model_results.get('image_aigc'):
            result = model_results['image_aigc']
            lines.append(f"**3. Image AIGC Detection (CBM):**")
            lines.append(f"   - Score: {result.get('score', 0):.2f}")
            lines.append(f"   - Label: {result.get('label', 'Unknown')}")
            lines.append(f"   - Confidence: {result.get('confidence', 0):.2f}")
            lines.append("")

        # 總結
        lines.append("### Recommendation:")
        if severity == "high":
            lines.append("⚠️ **High Risk**: Multiple models indicate potential deception. Exercise extreme caution.")
        elif severity == "medium":
            lines.append("⚡ **Medium Risk**: Some indicators of falsity detected. Verify with additional sources.")
        elif severity == "low":
            lines.append("💡 **Low Risk**: Mixed signals. Content appears somewhat credible but warrants fact-checking.")
        else:
            lines.append("✅ **Minimal Risk**: Models indicate content is likely authentic.")

        return "\n".join(lines)


# 測試代碼
if __name__ == "__main__":
    # 測試融合策略
    print("Testing FusionStrategy...")

    scores = {
        'text_aigc': 0.8,
        'text_intent': 0.6,
        'image_aigc': 0.9
    }

    strategy = FusionStrategy()

    print(f"Linear: {strategy.linear_fusion(scores):.3f}")
    print(f"Max: {strategy.max_fusion(scores):.3f}")
    print(f"Min: {strategy.min_fusion(scores):.3f}")
    print(f"Avg: {strategy.avg_fusion(scores):.3f}")
    print(f"Voting: {strategy.voting_fusion(scores):.3f}")

    # 測試 DecisionMaker
    print("\nTesting DecisionMaker...")

    model_results = {
        'text_aigc': {'score': 0.8, 'confidence': 0.9, 'label': 'AIGC'},
        'text_intent': {'score': 0.6, 'confidence': 0.7, 'label': 'Manipulative'},
        'image_aigc': {'score': 0.9, 'confidence': 0.85, 'label': 'AI-generated'}
    }

    decision_maker = DecisionMaker(fusion_method='linear')
    decision = decision_maker.make_decision(model_results)

    print(f"\nFinal Score: {decision['final_score']:.3f}")
    print(f"Final Label: {decision['final_label']}")
    print(f"Final Confidence: {decision['final_confidence']:.3f}")
    print(f"\nReport:\n{decision['report']}")