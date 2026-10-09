"""
Backend Package
假新聞檢測 API 後端
"""
__version__ = "0.1.0"
__author__ = "Fake News Detection Team"

# 避免循環導入，不在 __init__.py 中導入
# 這些模組在需要時直接從各自的檔案導入

__all__ = [
    'app',
    'ModelInference',
    'DecisionMaker',
    'NewsCrawler'
]
