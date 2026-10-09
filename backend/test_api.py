"""
測試腳本
用於測試後端 API 的各個端點
"""
import requests
import json
import base64
from pathlib import Path

# API 基礎 URL
BASE_URL = "http://localhost:5000"


def test_health():
    """測試健康檢查端點"""
    print("=" * 50)
    print("Testing /health endpoint...")
    print("=" * 50)
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        print(f"Status Code: {response.status_code}")
        print("Response:")
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))
        return response.status_code == 200
    except Exception as e:
        print(f" Error: {e}")
        return False


def test_crawl():
    """測試爬蟲端點"""
    print("\n" + "=" * 50)
    print("Testing /api/crawl endpoint...")
    print("=" * 50)
    
    # 使用一個測試 URL（可能需要替換為實際可訪問的 URL）
    test_url = "https://www.ettoday.net/news/20240101/2645678.htm"
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/crawl",
            json={"url": test_url},
            timeout=30
        )
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("Response:")
            print(f"  Title: {data.get('title', 'N/A')[:50]}...")
            print(f"  Content length: {len(data.get('content', ''))} chars")
            print(f"  Images found: {len(data.get('imageUrls', []))}")
            print(f"  Site: {data.get('siteName', 'N/A')}")
            return True
        else:
            print("Response:")
            print(json.dumps(response.json(), indent=2, ensure_ascii=False))
            return False
            
    except Exception as e:
        print(f" Error: {e}")
        return False


def test_detect_manual():
    """測試手動檢測端點"""
    print("\n" + "=" * 50)
    print("Testing /api/detect endpoint (manual input)...")
    print("=" * 50)
    
    # 測試數據
    test_data = {
        "title": "測試新聞標題",
        "content": "這是一則測試新聞的內容本文旨在測試假新聞檢測系統的功能",
        "sourceUrl": "https://example.com/test"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/detect",
            json=test_data,
            timeout=60
        )
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("Response:")
            print(f"  Title: {data.get('title')}")
            print(f"  Final Score: {data.get('finalScore'):.3f}")
            print(f"  Final Label: {data.get('finalLabel')}")
            print(f"  Final Confidence: {data.get('finalConfidence'):.3f}")
            print("\n  Model Scores:")
            print(f"    - Text AIGC: {data.get('textAigcScore', {}).get('score', 0):.3f} ({data.get('textAigcScore', {}).get('label', 'N/A')})")
            print(f"    - Text Intent: {data.get('textIntentScore', {}).get('score', 0):.3f} ({data.get('textIntentScore', {}).get('label', 'N/A')})")
            print(f"    - Image AIGC: {data.get('imageAigcScore', {}).get('score', 0):.3f} ({data.get('imageAigcScore', {}).get('label', 'N/A')})")
            
            print("\n  Report Preview:")
            report_lines = data.get('report', '').split('\n')[:5]
            for line in report_lines:
                print(f"    {line}")
            
            return True
        else:
            print("Response:")
            print(json.dumps(response.json(), indent=2, ensure_ascii=False))
            return False
            
    except Exception as e:
        print(f" Error: {e}")
        return False


def test_detect_with_image():
    """測試帶圖片的檢測"""
    print("\n" + "=" * 50)
    print("Testing /api/detect endpoint (with image URL)...")
    print("=" * 50)
    
    # 測試數據（使用一個公開的圖片 URL）
    test_data = {
        "title": "帶圖片的測試新聞",
        "content": "這是一則帶有圖片的測試新聞",
        "imageUrl": "https://picsum.photos/400/300",  # 測試圖片
        "sourceUrl": "https://example.com/test-with-image"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/detect",
            json=test_data,
            timeout=60
        )
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("Response:")
            print(f"  Final Score: {data.get('finalScore'):.3f}")
            print(f"  Final Label: {data.get('finalLabel')}")
            print(f"  Image AIGC Score: {data.get('imageAigcScore', {}).get('score', 0):.3f}")
            return True
        else:
            print("Response:")
            print(json.dumps(response.json(), indent=2, ensure_ascii=False))
            return False
            
    except Exception as e:
        print(f" Error: {e}")
        return False


def test_detect_from_url():
    """測試從 URL 檢測"""
    print("\n" + "=" * 50)
    print("Testing /api/detect-from-url endpoint...")
    print("=" * 50)
    
    # 使用一個測試 URL
    test_url = "https://www.ettoday.net/news/20240101/2645678.htm"
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/detect-from-url",
            json={"url": test_url},
            timeout=60
        )
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("Response:")
            print(f"  Title: {data.get('title', 'N/A')[:50]}...")
            print(f"  Final Score: {data.get('finalScore'):.3f}")
            print(f"  Final Label: {data.get('finalLabel')}")
            print(f"  Source URL: {data.get('sourceUrl', 'N/A')[:50]}...")
            return True
        else:
            print("Response:")
            print(json.dumps(response.json(), indent=2, ensure_ascii=False))
            return False
            
    except Exception as e:
        print(f" Error: {e}")
        return False


def main():
    """運行所有測試"""
    print("\n" + "=" * 50)
    print(" API Testing Suite")
    print("=" * 50)
    print(f"Target: {BASE_URL}")
    print("=" * 50)
    
    results = {
        "Health Check": test_health(),
        "Crawl Endpoint": test_crawl(),
        "Detect Manual": test_detect_manual(),
        "Detect with Image": test_detect_with_image(),
        "Detect from URL": test_detect_from_url()
    }
    
    # 彙總結果
    print("\n" + "=" * 50)
    print(" Test Results Summary")
    print("=" * 50)
    
    passed = 0
    failed = 0
    
    for test_name, result in results.items():
        status = " PASS" if result else " FAIL"
        print(f"{test_name:25s} {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print("=" * 50)
    print(f"Total: {len(results)} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print("=" * 50)
    
    return failed == 0


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
