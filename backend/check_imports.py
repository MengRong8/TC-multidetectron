"""
快速驗證腳本
檢查所有模塊是否可以正常導入
"""
import sys
from pathlib import Path

# 添加項目根目錄到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 60)
print(" Backend Components Import Test")
print("=" * 60)

# 測試導入
components = []

# 1. Crawler
print("\n1. Testing crawler import...")
try:
    from backend.crawler import NewsCrawler
    print("   NewsCrawler imported successfully")
    components.append(("NewsCrawler", True))
except Exception as e:
    print(f"   Failed to import NewsCrawler: {e}")
    components.append(("NewsCrawler", False))

# 2. Models
print("\n2. Testing models import...")
try:
    from backend.models import (
        TextAIGCModel,
        TextIntentModel,
        ImageAIGCModel,
        ModelInference
    )
    print("   Model classes imported successfully")
    components.append(("Models", True))
except Exception as e:
    print(f"   Failed to import Models: {e}")
    components.append(("Models", False))

# 3. Fusion
print("\n3. Testing fusion import...")
try:
    from backend.fusion import (
        DecisionFusionLayer,
        FusionStrategy,
        DecisionMaker
    )
    print("   Fusion classes imported successfully")
    components.append(("Fusion", True))
except Exception as e:
    print(f"   Failed to import Fusion: {e}")
    components.append(("Fusion", False))

# 4. App
print("\n4. Testing app import...")
try:
    from backend.app import app
    print("   Flask app imported successfully")
    components.append(("Flask App", True))
except Exception as e:
    print(f"   Failed to import Flask app: {e}")
    components.append(("Flask App", False))

# 5. Config
print("\n5. Testing config import...")
try:
    from backend.config import MODEL_PATHS, FUSION_CONFIG, API_CONFIG
    print("   Config imported successfully")
    print(f"      - Model paths configured: {len(MODEL_PATHS)}")
    print(f"      - Fusion method: {FUSION_CONFIG['method']}")
    print(f"      - API port: {API_CONFIG['port']}")
    components.append(("Config", True))
except Exception as e:
    print(f"   Failed to import Config: {e}")
    components.append(("Config", False))

print("\n" + "=" * 60)
print(" Summary")
print("=" * 60)

passed = sum(1 for _, success in components if success)
failed = sum(1 for _, success in components if not success)

for name, success in components:
    status = " PASS" if success else " FAIL"
    print(f"{name:20s} {status}")

print("=" * 60)
print(f"Total: {len(components)}")
print(f"Passed: {passed}")
print(f"Failed: {failed}")
print("=" * 60)

if failed == 0:
    print("\n All components imported successfully!")
    print("You can now start the API server with:")
    print("   cd backend && ./start.sh")
else:
    print("\n  Some components failed to import.")
    print("Please check the error messages above.")

sys.exit(0 if failed == 0 else 1)
