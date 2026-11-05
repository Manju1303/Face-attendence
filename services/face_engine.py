import os
import cv2
import numpy as np
import base64
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class FaceEngine:
    """
    Unified Face Recognition and Facial Emotion Analysis Engine.
    Uses:
      - OpenCV YuNet (Deep Neural Network Face Detection & 5-point landmark alignment)
      - OpenCV SFace (State-of-the-Art 128-Dimensional Face Feature Vector / Embedding)
      - FER+ ONNX (FER2013-derived Real-time Facial Emotion Recognition)
      - Fallback Haar Cascade & Caffe SSD support
    """
    
    def __init__(self, models_dir=None):
        if models_dir is None:
            models_dir = Path(__file__).resolve().parent.parent / 'models_data'
        else:
            models_dir = Path(models_dir)
            
        self.models_dir = models_dir
        self.yunet_path = models_dir / 'face_detection_yunet_2023mar.onnx'
        self.sface_path = models_dir / 'face_recognition_sface_2021dec.onnx'
        self.emotion_path = models_dir / 'emotion_ferplus.onnx'
        self.haarcascade_path = models_dir / 'haarcascade_frontalface_default.xml'
        
        self.detector = None
        self.recognizer = None
        self.emotion_net = None
        self.haar_cascade = None
        
        self.emotion_labels = [
            'Neutral', 'Happy', 'Surprise', 'Sad', 
            'Angry', 'Disgust', 'Fear', 'Contempt'
        ]
        
        self._load_models()

    def _load_models(self):
        try:
            # Auto-download models if missing
            if not self.yunet_path.exists() or not self.sface_path.exists():
                try:
                    from models_data.download_models import ensure_models
                    ensure_models()
                except Exception as ex:
                    logger.warning(f"Could not auto-download models: {ex}")

            if self.yunet_path.exists():
                self.detector = cv2.FaceDetectorYN.create(
                    str(self.yunet_path), "", (320, 320), 0.6, 0.3, 5000
                )
                logger.info("YuNet Face Detector initialized.")
            else:
                logger.warning(f"YuNet model not found at {self.yunet_path}")

            if self.sface_path.exists():
                self.recognizer = cv2.FaceRecognizerSF.create(str(self.sface_path), "")
                logger.info("SFace Face Recognizer initialized.")
            else:
                logger.warning(f"SFace model not found at {self.sface_path}")

            if self.emotion_path.exists():
                self.emotion_net = cv2.dnn.readNetFromONNX(str(self.emotion_path))
                logger.info("FER+ Emotion Recognition Model initialized.")
            else:
                logger.warning(f"Emotion model not found at {self.emotion_path}")

            if hasattr(cv2, 'CascadeClassifier') and self.haarcascade_path.exists():
                self.haar_cascade = cv2.CascadeClassifier(str(self.haarcascade_path))
        except Exception as e:
            logger.error(f"Error loading models: {e}")

    @staticmethod
    def decode_base64_image(b64_string):
        """Decode base64 image (from HTML5 canvas or webcam snapshot) into OpenCV BGR image."""
        try:
            if ',' in b64_string:
                b64_string = b64_string.split(',', 1)[1]
            img_bytes = base64.b64decode(b64_string)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            logger.error(f"Base64 decode error: {e}")
            return None

    def detect_faces(self, image_bgr, min_confidence=0.6):
        """
        Detect faces in image.
        Returns list of dicts: [{
            'bbox': (x, y, w, h),
            'confidence': float,
            'landmarks': [(x1,y1), (x2,y2), ...],
            'raw_face': face_array
        }]
        """
        if image_bgr is None:
            return []

        h, w, _ = image_bgr.shape
        if self.detector is not None:
            self.detector.setInputSize((w, h))
            self.detector.setScoreThreshold(min_confidence)
            retval, faces = self.detector.detect(image_bgr)
            
            if faces is None or len(faces) == 0:
                return []

            results = []
            for face in faces:
                x, y, box_w, box_h = map(int, face[:4])
                # Ensure box stays within frame boundaries
                x = max(0, x)
                y = max(0, y)
                box_w = min(w - x, box_w)
                box_h = min(h - y, box_h)
                
                # Filter out tiny artifacts
                if box_w < 30 or box_h < 30:
                    continue

                conf = float(face[-1])
                landmarks = [(int(face[i]), int(face[i+1])) for i in range(4, 14, 2)]
                
                results.append({
                    'bbox': (x, y, box_w, box_h),
                    'confidence': conf,
                    'landmarks': landmarks,
                    'raw_face': face
                })
            return results

        # Fallback to Haar Cascade if available
        if self.haar_cascade is not None:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            faces = self.haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            results = []
            for (x, y, bw, bh) in faces:
                results.append({
                    'bbox': (x, y, bw, bh),
                    'confidence': 0.8,
                    'landmarks': [],
                    'raw_face': None
                })
            return results

        return []

    def extract_embedding(self, image_bgr, face_data):
        """
        Align face and extract 128-dimensional L2-normalized feature vector.
        """
        if self.recognizer is None or image_bgr is None:
            return None

        try:
            raw_face = face_data.get('raw_face')
            if raw_face is not None:
                aligned_face = self.recognizer.alignCrop(image_bgr, raw_face)
                feature = self.recognizer.feature(aligned_face)
                # Flatten to 1D float list
                return feature[0].astype(float).tolist()
            else:
                # If no landmarks available, crop directly
                x, y, w, h = face_data['bbox']
                crop = image_bgr[y:y+h, x:x+w]
                resized = cv2.resize(crop, (112, 112))
                feature = self.recognizer.feature(resized)
                return feature[0].astype(float).tolist()
        except Exception as e:
            logger.error(f"Error extracting embedding: {e}")
            return None

    def predict_emotion(self, image_bgr, bbox):
        """
        Predict facial emotion using FER+ ONNX model on cropped face.
        Returns: (dominant_emotion, confidence, dict_of_all_probs)
        """
        if self.emotion_net is None or image_bgr is None:
            return "Neutral", 1.0, {"Neutral": 1.0}

        try:
            x, y, w, h = bbox
            h_img, w_img = image_bgr.shape[:2]
            # Clip bounds
            x = max(0, min(x, w_img - 1))
            y = max(0, min(y, h_img - 1))
            w = max(10, min(w, w_img - x))
            h = max(10, min(h, h_img - y))

            crop = image_bgr[y:y+h, x:x+w]
            if crop.size == 0:
                return "Neutral", 1.0, {"Neutral": 1.0}

            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            blob = cv2.dnn.blobFromImage(gray, 1.0, (64, 64))

            self.emotion_net.setInput(blob)
            out = self.emotion_net.forward()[0]

            # Softmax
            exp_out = np.exp(out - np.max(out))
            probs = exp_out / exp_out.sum()

            dominant_idx = int(np.argmax(probs))
            dominant_emotion = self.emotion_labels[dominant_idx]
            confidence = float(probs[dominant_idx])

            all_probs = {self.emotion_labels[i]: round(float(probs[i]), 3) for i in range(len(self.emotion_labels))}
            return dominant_emotion, confidence, all_probs
        except Exception as e:
            logger.error(f"Error predicting emotion: {e}")
            return "Neutral", 0.0, {}

    def compute_similarity(self, emb1, emb2):
        """
        Compute Cosine Similarity between two 128D embeddings.
        Returns float between -1.0 and 1.0. Higher is closer.
        """
        if not emb1 or not emb2:
            return 0.0

        vec1 = np.array(emb1, dtype=np.float32)
        vec2 = np.array(emb2, dtype=np.float32)

        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def find_best_match(self, query_embedding, enrolled_students, threshold=0.60, uncertainty_margin=0.08, ambiguity_margin=0.05):
        """
        Compares query_embedding against all active students' enrolled samples.
        Strict verification:
          - Rejects inactive students
          - Checks top match meets calibrated threshold
          - Checks ambiguity: rejects if top two matching students have similarity difference < ambiguity_margin
          - Rejects uncertain and unknown faces
        Returns:
            match_status: 'MATCH' | 'AMBIGUOUS_MATCH' | 'UNCERTAIN' | 'NO_MATCH'
            matched_student: Student instance or None
            highest_score: float (highest similarity found)
            candidate_student: Student instance if candidate found or None
        """
        if not enrolled_students or not query_embedding:
            return 'NO_MATCH', None, 0.0, None

        best_score = -1.0
        best_student = None
        second_best_score = -1.0
        second_best_student = None

        for student in enrolled_students:
            # Enforce active student check
            if not getattr(student, 'active_status', True):
                continue
            for enc in student.encodings:
                emb = enc.get_embedding()
                if not emb:
                    continue
                score = self.compute_similarity(query_embedding, emb)
                if score > best_score:
                    if best_student is not None and best_student.id != student.id:
                        second_best_score = best_score
                        second_best_student = best_student
                    best_score = score
                    best_student = student
                elif score > second_best_score and student.id != (best_student.id if best_student else None):
                    second_best_score = score
                    second_best_student = student

        if best_student is None or best_score <= 0.0:
            return 'NO_MATCH', None, 0.0, None

        uncertain_floor = max(0.0, threshold - uncertainty_margin)

        if best_score >= threshold:
            # Check for ambiguity between two different students
            if second_best_student is not None and (best_score - second_best_score) < ambiguity_margin:
                logger.warning(
                    f"Ambiguous biometric match between Student {best_student.id} ({best_score:.3f}) "
                    f"and Student {second_best_student.id} ({second_best_score:.3f})"
                )
                return 'AMBIGUOUS_MATCH', None, best_score, best_student
            return 'MATCH', best_student, best_score, None
        elif best_score >= uncertain_floor:
            # Score is in the uncertain range
            return 'UNCERTAIN', None, best_score, best_student
        else:
            return 'NO_MATCH', None, best_score, None

    def check_duplicate_face(self, query_embedding, enrolled_students, threshold=0.55):
        """
        Checks if a newly captured face during registration already matches an existing student.
        Prevents accidental double registration.
        """
        if not enrolled_students or not query_embedding:
            return None, 0.0

        for student in enrolled_students:
            for enc in student.encodings:
                emb = enc.get_embedding()
                if not emb:
                    continue
                score = self.compute_similarity(query_embedding, emb)
                if score >= threshold:
                    return student, score

        return None, 0.0


# Global singleton instance
face_engine = FaceEngine()

