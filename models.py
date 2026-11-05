from datetime import datetime, date, time
import json
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()

class Admin(db.Model):
    __tablename__ = 'admins'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f'<Admin {self.username}>'


class Student(db.Model):
    __tablename__ = 'students'

    id = db.Column(db.Integer, primary_key=True)
    register_number = db.Column(db.String(50), unique=True, nullable=False, index=True)
    full_name = db.Column(db.String(120), nullable=False, index=True)
    department = db.Column(db.String(80), nullable=False, index=True)
    class_year = db.Column(db.String(50), nullable=False, index=True)
    email = db.Column(db.String(120), nullable=True)
    phone_number = db.Column(db.String(30), nullable=True)
    active_status = db.Column(db.Boolean, default=True, nullable=False, index=True)
    photo_filename = db.Column(db.String(255), nullable=True)
    password_hash = db.Column(db.String(255), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        if not self.password_hash:
            # Default password fallback: 'student123' or their register number
            return password == 'student123' or password.upper() == self.register_number.upper()
        return check_password_hash(self.password_hash, password)

    # Relationships
    encodings = db.relationship('FaceEncoding', backref='student', lazy=True, cascade="all, delete-orphan")
    attendance_records = db.relationship('Attendance', backref='student', lazy=True, cascade="all, delete-orphan")

    def to_dict(self):
        return {
            'id': self.id,
            'register_number': self.register_number,
            'full_name': self.full_name,
            'department': self.department,
            'class_year': self.class_year,
            'email': self.email or f"{self.register_number.lower()}@university.edu",
            'phone_number': self.phone_number or "+91 98765 43210",
            'active_status': self.active_status,
            'photo_filename': self.photo_filename,
            'samples_count': len(self.encodings),
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S') if self.created_at else None
        }

    def __repr__(self):
        return f'<Student {self.register_number} - {self.full_name}>'


class FaceEncoding(db.Model):
    __tablename__ = 'face_encodings'

    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('students.id', ondelete='CASCADE'), nullable=False, index=True)
    # Numerical embedding vector stored safely as JSON list of floats
    embedding_data = db.Column(db.Text, nullable=False)
    sample_index = db.Column(db.Integer, default=0)
    model_version = db.Column(db.String(50), default="sface_v1_128d", nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    def get_embedding(self):
        try:
            return json.loads(self.embedding_data)
        except Exception:
            return []

    def set_embedding(self, embedding_list):
        if hasattr(embedding_list, 'tolist'):
            embedding_list = embedding_list.tolist()
        self.embedding_data = json.dumps([float(x) for x in embedding_list])


class ClassSession(db.Model):
    __tablename__ = 'class_sessions'

    id = db.Column(db.Integer, primary_key=True)
    session_name = db.Column(db.String(120), nullable=False)
    department = db.Column(db.String(80), nullable=False, index=True)
    class_year = db.Column(db.String(50), nullable=False, index=True)
    session_date = db.Column(db.Date, default=date.today, nullable=False, index=True)
    start_time = db.Column(db.Time, nullable=False)
    end_time = db.Column(db.Time, nullable=False)
    created_by = db.Column(db.Integer, db.ForeignKey('admins.id'), nullable=True)
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    attendance_records = db.relationship('Attendance', backref='session', lazy=True, cascade="all, delete-orphan")

    def to_dict(self):
        return {
            'id': self.id,
            'session_name': self.session_name,
            'department': self.department,
            'class_year': self.class_year,
            'session_date': self.session_date.strftime('%Y-%m-%d') if self.session_date else None,
            'start_time': self.start_time.strftime('%H:%M') if self.start_time else None,
            'end_time': self.end_time.strftime('%H:%M') if self.end_time else None,
            'is_active': self.is_active,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S') if self.created_at else None
        }

    def __repr__(self):
        return f'<ClassSession {self.session_name} - {self.session_date}>'


class Attendance(db.Model):
    __tablename__ = 'attendance'

    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('students.id', ondelete='CASCADE'), nullable=False, index=True)
    session_id = db.Column(db.Integer, db.ForeignKey('class_sessions.id', ondelete='CASCADE'), nullable=False, index=True)
    attendance_date = db.Column(db.Date, default=date.today, nullable=False, index=True)
    check_in_time = db.Column(db.Time, default=lambda: datetime.now().time(), nullable=False)
    status = db.Column(db.String(20), default='Present', nullable=False)  # 'Present', 'Absent', 'Late', 'Excused'
    verification_method = db.Column(db.String(50), default='Face Recognition', nullable=False)
    confidence_or_distance = db.Column(db.Float, nullable=True)
    
    # Audit trail for corrections
    is_manually_corrected = db.Column(db.Boolean, default=False)
    corrected_by_admin = db.Column(db.String(80), nullable=True)
    correction_note = db.Column(db.String(255), nullable=True)
    corrected_at = db.Column(db.DateTime, nullable=True)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        # Ensure a student can only have 1 attendance record per class session
        db.UniqueConstraint('student_id', 'session_id', 'attendance_date', name='uq_student_session_date'),
        db.Index('idx_attendance_date_session', 'attendance_date', 'session_id'),
    )

    def to_dict(self):
        return {
            'id': self.id,
            'student_id': self.student_id,
            'register_number': self.student.register_number if self.student else None,
            'student_name': self.student.full_name if self.student else None,
            'department': self.student.department if self.student else None,
            'class_year': self.student.class_year if self.student else None,
            'session_id': self.session_id,
            'session_name': self.session.session_name if self.session else None,
            'attendance_date': self.attendance_date.strftime('%Y-%m-%d') if self.attendance_date else None,
            'check_in_time': self.check_in_time.strftime('%H:%M:%S') if self.check_in_time else None,
            'status': self.status,
            'verification_method': self.verification_method,
            'confidence': round(self.confidence_or_distance, 4) if self.confidence_or_distance is not None else None,
            'is_manually_corrected': self.is_manually_corrected,
            'correction_note': self.correction_note,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S') if self.created_at else None
        }


class AttendanceAuditLog(db.Model):
    __tablename__ = 'attendance_audit_logs'

    id = db.Column(db.Integer, primary_key=True)
    attendance_id = db.Column(db.Integer, db.ForeignKey('attendance.id', ondelete='CASCADE'), nullable=False, index=True)
    action = db.Column(db.String(50), nullable=False) # e.g. "Status changed Present -> Absent"
    previous_status = db.Column(db.String(20), nullable=True)
    new_status = db.Column(db.String(20), nullable=True)
    reason = db.Column(db.String(255), nullable=True)
    admin_username = db.Column(db.String(80), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)

