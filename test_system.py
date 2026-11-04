import unittest
import json
import base64
import os
import cv2
import numpy as np
from datetime import date, datetime

from app import create_app
from models import db, Admin, Student, FaceEncoding, ClassSession, Attendance, AttendanceAuditLog
from services.face_engine import face_engine

class BiometricAttendanceTestCase(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.app.config['TESTING'] = True
        self.app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        self.app.config['WTF_CSRF_ENABLED'] = False
        self.client = self.app.test_client()

        with self.app.app_context():
            db.create_all()
            # Seed test admin
            admin = Admin(username='testadmin')
            admin.set_password('password123')
            db.session.add(admin)
            
            # Seed test session
            test_sess = ClassSession(
                session_name="Computer Networks Test Session",
                department="Computer Science & Engineering",
                class_year="Year 3 - Semester 5",
                session_date=date.today(),
                start_time=datetime.strptime("09:00", "%H:%M").time(),
                end_time=datetime.strptime("11:00", "%H:%M").time(),
                is_active=True
            )
            db.session.add(test_sess)
            db.session.commit()

    def tearDown(self):
        with self.app.app_context():
            db.session.remove()
            db.drop_all()

    def test_01_admin_login_and_logout(self):
        """Test secure admin login with hashed password and session management."""
        # Invalid password
        resp = self.client.post('/admin/login', data={'username': 'testadmin', 'password': 'wrongpassword'}, follow_redirects=True)
        self.assertIn(b'Invalid username or password', resp.data)

        # Valid login
        resp = self.client.post('/admin/login', data={'username': 'testadmin', 'password': 'password123'}, follow_redirects=True)
        self.assertIn(b'Welcome back, testadmin!', resp.data)

        # Access protected dashboard
        resp = self.client.get('/admin/dashboard')
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b'Administrative Dashboard', resp.data)

        # Logout
        resp = self.client.get('/admin/logout', follow_redirects=True)
        self.assertIn(b'You have been logged out securely', resp.data)

        # Verify dashboard now redirects to login
        resp = self.client.get('/admin/dashboard', follow_redirects=True)
        self.assertIn(b'Please log in with administrator credentials', resp.data)

    def test_02_student_registration_and_duplicate_check(self):
        """Test registering student with biometric samples and preventing duplicates."""
        # Log in first
        self.client.post('/admin/login', data={'username': 'testadmin', 'password': 'password123'})

        # Load Lena test photo as sample
        lena_img = cv2.imread('test_images/lena.jpg')
        _, buffer = cv2.imencode('.jpg', lena_img)
        lena_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode()

        # Register Student 1
        resp = self.client.post('/admin/students/add', data={
            'register_number': 'CS2026101',
            'full_name': 'Lena Forsen',
            'department': 'Computer Science & Engineering',
            'class_year': 'Year 3 - Semester 5',
            'email': 'lena@university.edu',
            'face_samples': json.dumps([lena_b64, lena_b64])
        }, follow_redirects=True)

        self.assertIn(b'successfully registered', resp.data)

        with self.app.app_context():
            student = Student.query.filter_by(register_number='CS2026101').first()
            self.assertIsNotNone(student)
            self.assertEqual(student.full_name, 'Lena Forsen')
            self.assertEqual(len(student.encodings), 2)
            self.assertEqual(len(student.encodings[0].get_embedding()), 128)

        # Try registering with duplicate Register Number
        resp = self.client.post('/admin/students/add', data={
            'register_number': 'CS2026101',
            'full_name': 'Another Student',
            'department': 'Electrical Engineering',
            'class_year': 'Year 1',
            'face_samples': json.dumps([lena_b64])
        }, follow_redirects=True)
        self.assertIn(b'already exists', resp.data)

        # Try registering a different student with same face (Duplicate face warning)
        resp = self.client.post('/admin/students/add', data={
            'register_number': 'CS2026102',
            'full_name': 'Twin Student',
            'department': 'Computer Science & Engineering',
            'class_year': 'Year 3 - Semester 5',
            'face_samples': json.dumps([lena_b64])
        }, follow_redirects=True)
        self.assertIn(b'Duplicate Face Warning', resp.data)

    def test_03_biometric_attendance_verification(self):
        """Test real-time face verification and server-side duplicate prevention."""
        # 1. Register Lena
        self.client.post('/admin/login', data={'username': 'testadmin', 'password': 'password123'})
        lena_img = cv2.imread('test_images/lena.jpg')
        _, buffer = cv2.imencode('.jpg', lena_img)
        lena_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode()

        self.client.post('/admin/students/add', data={
            'register_number': 'CS2026101',
            'full_name': 'Lena Forsen',
            'department': 'Computer Science & Engineering',
            'class_year': 'Year 3 - Semester 5',
            'face_samples': json.dumps([lena_b64])
        })

        with self.app.app_context():
            sess = ClassSession.query.first()
            session_id = sess.id

        # 2. Test No Face Detected
        black_img = np.zeros((320, 320, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', black_img)
        black_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode()

        resp = self.client.post('/api/verify-attendance', json={
            'image': black_b64,
            'session_id': session_id
        })
        data = resp.get_json()
        self.assertEqual(data['status'], 'NO_FACE')
        self.assertIn('No face detected', data['message'])

        # 3. Test Successful Attendance Verification with Lena's face
        resp = self.client.post('/api/verify-attendance', json={
            'image': lena_b64,
            'session_id': session_id
        })
        data = resp.get_json()
        self.assertTrue(data['success'])
        self.assertEqual(data['status'], 'SUCCESS')
        self.assertEqual(data['student']['register_number'], 'CS2026101')
        self.assertEqual(data['student']['name'], 'Lena Forsen')
        self.assertGreater(data['confidence'], 90.0)
        self.assertIn('emotion', data)

        with self.app.app_context():
            rec = Attendance.query.filter_by(session_id=session_id).first()
            self.assertIsNotNone(rec)
            self.assertEqual(rec.status, 'Present')
            self.assertGreater(rec.confidence_or_distance, 0.6)

        # 4. Test Duplicate Attendance Rule (repeated scan in same session)
        resp2 = self.client.post('/api/verify-attendance', json={
            'image': lena_b64,
            'session_id': session_id
        })
        data2 = resp2.get_json()
        self.assertTrue(data2['success'])
        self.assertEqual(data2['status'], 'ALREADY_MARKED')
        self.assertTrue(data2['already_marked'])
        self.assertIn('Attendance already recorded', data2['message'])

        # Verify only 1 attendance row exists in DB
        with self.app.app_context():
            count = Attendance.query.filter_by(session_id=session_id).count()
            self.assertEqual(count, 1)

    def test_04_attendance_audit_correction(self):
        """Test admin attendance correction with audit trail."""
        self.client.post('/admin/login', data={'username': 'testadmin', 'password': 'password123'})
        
        with self.app.app_context():
            student = Student(
                register_number='CS2026105',
                full_name='Test Student',
                department='Mechanical',
                class_year='Year 2'
            )
            db.session.add(student)
            db.session.flush()

            sess = ClassSession.query.first()
            att = Attendance(
                student_id=student.id,
                session_id=sess.id,
                attendance_date=date.today(),
                check_in_time=datetime.now().time(),
                status='Present',
                verification_method='Face Recognition'
            )
            db.session.add(att)
            db.session.commit()
            att_id = att.id

        # Admin corrects status to 'Absent' with reason
        resp = self.client.post(f'/admin/attendance/correct/{att_id}', data={
            'status': 'Absent',
            'reason': 'Left lecture without permission during verification check'
        }, follow_redirects=True)

        self.assertIn(b'corrected from Present to Absent', resp.data)

        with self.app.app_context():
            updated = db.session.get(Attendance, att_id)
            self.assertEqual(updated.status, 'Absent')
            self.assertTrue(updated.is_manually_corrected)
            self.assertEqual(updated.correction_note, 'Left lecture without permission during verification check')

            audit = AttendanceAuditLog.query.filter_by(attendance_id=att_id).first()
            self.assertIsNotNone(audit)
            self.assertEqual(audit.previous_status, 'Present')
            self.assertEqual(audit.new_status, 'Absent')
            self.assertEqual(audit.admin_username, 'testadmin')

    def test_05_csv_and_excel_export(self):
        """Test CSV and Excel export endpoints."""
        self.client.post('/admin/login', data={'username': 'testadmin', 'password': 'password123'})

        # CSV export
        resp = self.client.get(f'/admin/attendance/export/csv?date={date.today().strftime("%Y-%m-%d")}')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.mimetype, 'text/csv')
        self.assertIn(b'Register Number,Student Name', resp.data)

        # Excel export
        resp = self.client.get(f'/admin/attendance/export/excel?date={date.today().strftime("%Y-%m-%d")}')
        self.assertEqual(resp.status_code, 200)
        self.assertIn('spreadsheetml', resp.mimetype)

if __name__ == '__main__':
    unittest.main()
