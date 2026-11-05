import os
from flask import Flask, send_from_directory, render_template
from config import Config
from models import db, Admin, ClassSession, Student
from datetime import date, time
from routes import auth_bp, admin_bp, attendance_bp, student_bp

def create_app(config_override=None):
    app = Flask(__name__)
    app.config.from_object(Config)
    if config_override:
        app.config.update(config_override)

    # Ensure required directories exist
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(Config.STUDENT_PHOTOS_DIR, exist_ok=True)
    os.makedirs(Config.MODELS_DIR, exist_ok=True)

    # Initialize database
    db.init_app(app)

    # Register blueprints   
    app.register_blueprint(auth_bp)
    app.register_blueprint(admin_bp)
    app.register_blueprint(attendance_bp)
    app.register_blueprint(student_bp)

    # Serve student uploaded photos securely
    @app.route('/uploads/students/<filename>')
    def student_photo(filename):
        return send_from_directory(Config.STUDENT_PHOTOS_DIR, filename)

    # Initialize Database and default admin
    with app.app_context():
        db.create_all()
        
        # Seed default admin if none exists
        if not Admin.query.filter_by(username='admin').first():
            default_admin = Admin(username='admin')
            default_admin.set_password('admin123')
            db.session.add(default_admin)
            db.session.commit()
            print("[INFO] Default admin created: username='admin', password='admin123'")

        # Seed default class session for today if none exists
        today = date.today()
        if not ClassSession.query.filter_by(session_date=today).first():
            default_session = ClassSession(
                session_name="Computer Science - Morning Lecture",
                department="Computer Science & Engineering",
                class_year="Year 3 - Semester 5",
                session_date=today,
                start_time=time(9, 0),
                end_time=time(17, 0),
                is_active=True
            )
            db.session.add(default_session)
            db.session.commit()
            print("[INFO] Initialized default class session for today.")

        # Seed registered portal students if none exist in database and not in testing mode
        if not app.config.get('TESTING') and Student.query.count() == 0:
            sample_portal_students = [
                {
                    'reg': 'CS2026021',
                    'name': 'R. Dharanika',
                    'dept': 'Information Technology',
                    'class': 'Year 3 - Semester 5',
                    'email': 'dharanika@university.edu',
                    'phone': '+91 98765 43210'
                },
                {
                    'reg': 'CS2026022',
                    'name': 'K. Dharshini',
                    'dept': 'Computer Science & Engineering',
                    'class': 'Year 3 - Semester 5',
                    'email': 'dharshini@university.edu',
                    'phone': '+91 98765 43211'
                },
                {
                    'reg': 'CS2026023',
                    'name': 'S. Gowshika',
                    'dept': 'Artificial Intelligence & Data Science',
                    'class': 'Year 3 - Semester 5',
                    'email': 'gowshika@university.edu',
                    'phone': '+91 98765 43212'
                }
            ]
            from models import FaceEncoding
            from services.face_engine import face_engine
            import cv2

            for s_data in sample_portal_students:
                s = Student(
                    register_number=s_data['reg'],
                    full_name=s_data['name'],
                    department=s_data['dept'],
                    class_year=s_data['class'],
                    email=s_data['email'],
                    phone_number=s_data['phone'],
                    active_status=True
                )
                s.set_password(f"Pass{s_data['reg']}")
                db.session.add(s)
                db.session.flush()

                # If test image exists, attach face profile & extract embedding
                if os.path.exists('test_images/lena.jpg') and s_data['reg'] == 'CS2026021':
                    img = cv2.imread('test_images/lena.jpg')
                    if img is not None:
                        photo_fname = f"{s_data['reg']}_profile.jpg"
                        photo_path = Config.STUDENT_PHOTOS_DIR / photo_fname
                        cv2.imwrite(str(photo_path), img)
                        s.photo_filename = photo_fname
                        faces = face_engine.detect_faces(img)
                        if faces:
                            emb = face_engine.extract_embedding(img, faces[0])
                            if emb:
                                fe = FaceEncoding(student_id=s.id, sample_index=0, model_version="sface_v1_128d")
                                fe.set_embedding(emb)
                                db.session.add(fe)
            db.session.commit()
            print("[INFO] Seeded registered portal students with complete profile details.")

    return app

app = create_app()

if __name__ == '__main__':
    # Listen on localhost port 5000
    print("\n" + "="*60)
    print(" Biometric Attendance Management System")
    print(" Admin Portal:     http://127.0.0.1:5000/admin/login")
    print(" Student Portal:   http://127.0.0.1:5000/attendance")
    print(" Default Admin:    admin / admin123")
    print("="*60 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=True)
