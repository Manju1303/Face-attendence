"""
Automated Model Downloader for BioAttendAI.
Downloads pre-trained deep learning face detection, face recognition, and emotion models.
"""

import os
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent

MODELS = {
    'face_detection_yunet_2023mar.onnx': 'https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx',
    'face_recognition_sface_2021dec.onnx': 'https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx',
    'emotion_ferplus.onnx': 'https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx',
    'haarcascade_frontalface_default.xml': 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml',
    'deploy.prototxt': 'https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt',
    'res10_300x300_ssd_iter_140000.caffemodel': 'https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel'
}

def ensure_models():
    os.makedirs(MODELS_DIR, exist_ok=True)
    for filename, url in MODELS.items():
        dest = MODELS_DIR / filename
        if dest.exists() and dest.stat().st_size > 1000:
            print(f"[OK] {filename} already present ({dest.stat().st_size:,} bytes).")
            continue

        print(f"[DOWNLOADING] {filename} from {url}...")
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as resp, open(dest, 'wb') as out_f:
                total = int(resp.headers.get('Content-Length', 0))
                downloaded = 0
                while True:
                    chunk = resp.read(1024 * 256)
                    if not chunk:
                        break
                    out_f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0 and downloaded % (1024 * 1024 * 5) < (1024 * 256):
                        print(f"  {downloaded / 1024 / 1024:.1f} MB / {total / 1024 / 1024:.1f} MB")
            print(f"[DONE] {filename} downloaded successfully ({dest.stat().st_size:,} bytes).")
        except Exception as e:
            print(f"[ERROR] Failed to download {filename}: {e}")

if __name__ == '__main__':
    ensure_models()

