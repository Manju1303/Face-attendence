import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

class Config:
    # Security
    SECRET_KEY = os.environ.get('SECRET_KEY', 'biometric-attendance-secure-key-2026-prod-fallback')
    
    # Database - SQLite by default, easily switchable to PostgreSQL via DATABASE_URL
    # e.g., postgresql+psycopg2://user:password@localhost/attendance_db
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', f'sqlite:///{BASE_DIR / "attendance.db"}')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Face Recognition Engine configuration
    # Threshold for Face match (Cosine distance: lower means stricter, typical 0.36 - 0.40)
    # Cosine Similarity: higher means closer (typical threshold: 0.60 or cosine distance <= 0.363)
    FACE_SIMILARITY_THRESHOLD = float(os.environ.get('FACE_SIMILARITY_THRESHOLD', 0.60))
    # Threshold for uncertain match warning
    FACE_UNCERTAINTY_MARGIN = float(os.environ.get('FACE_UNCERTAINTY_MARGIN', 0.08))
    
    # Model directories
    MODELS_DIR = BASE_DIR / 'models_data'
    UPLOAD_FOLDER = BASE_DIR / 'uploads'
    STUDENT_PHOTOS_DIR = UPLOAD_FOLDER / 'students'
    
    # Session timeout (in seconds, e.g., 2 hours)
    PERMANENT_SESSION_LIFETIME = 7200
    
    # Session status defaults
    DEFAULT_SESSION_NAME = "General Attendance Session"

