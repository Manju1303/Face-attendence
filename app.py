import os
from flask import Flask, send_from_directory, render_template
from config import Config
from models import db, Admin, ClassSession, Student
from datetime import date, time
from routes import auth_bp, admin_bp, attendance_bp, student_bp

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

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
