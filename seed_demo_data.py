import os
import cv2
import numpy as np
from datetime import date, datetime, time
from app import create_app
from models import db, Admin, Student, FaceEncoding, ClassSession, Attendance
from services.face_engine import face_engine

def seed():
    app = create_app()
    with app.app_context():
        # Check if already seeded
        if Student.query.filter_by(register_number='CS2026001').first():
            print("[INFO] Demo data already seeded.")
            return

        print("[INFO] Seeding initial demo students and biometric samples...")
        
        # 1. Student 1: Lena Forsen (with real face embedding from test_images/lena.jpg)
        lena_img = cv2.imread('test_images/lena.jpg')
        if lena_img is not None:
            faces = face_engine.detect_faces(lena_img)
            if faces:
                emb = face_engine.extract_embedding(lena_img, faces[0])
                
                # Save photo
                photo_fname = "CS2026001_demo.jpg"
                cv2.imwrite(str(app.config['STUDENT_PHOTOS_DIR'] / photo_fname), lena_img)

                s1 = Student(
                    register_number='CS2026001',
                    full_name='Lena Forsen',
                    department='Computer Science & Engineering',
                    class_year='Year 3 - Semester 5',
                    email='lena.forsen@university.edu',
                    photo_filename=photo_fname,
                    active_status=True
                )
                db.session.add(s1)
                db.session.flush()

                fe = FaceEncoding(
                    student_id=s1.id,
                    sample_index=0,
                    model_version='sface_v1_128d'
                )
                fe.set_embedding(emb)
                db.session.add(fe)

        # 2. Student 2: Alex Rivera
        s2 = Student(
            register_number='CS2026002',
            full_name='Alex Rivera',
            department='Computer Science & Engineering',
            class_year='Year 3 - Semester 5',
            email='alex.rivera@university.edu',
            active_status=True
        )
        db.session.add(s2)

        # 3. Student 3: Priya Sharma
        s3 = Student(
            register_number='EC2026045',
            full_name='Priya Sharma',
            department='Electronics & Communication',
            class_year='Year 2 - Semester 3',
            email='priya.sharma@university.edu',
            active_status=True
        )
        db.session.add(s3)
        db.session.flush()

        # Add initial attendance record for Lena today
        session = ClassSession.query.filter_by(session_date=date.today()).first()
        if session and 's1' in locals():
            att = Attendance(
                student_id=s1.id,
                session_id=session.id,
                attendance_date=date.today(),
                check_in_time=datetime.now().time(),
                status='Present',
                verification_method='Face Recognition (YuNet + SFace)',
                confidence_or_distance=0.998
            )
            db.session.add(att)

        db.session.commit()
        print("[SUCCESS] Initial demo students and sessions successfully seeded!")

if __name__ == '__main__':
    seed()

