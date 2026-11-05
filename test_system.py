import unittest
import json
import base64
import os
import cv2
import numpy as np
from datetime import date, datetime, time, timedelta

from app import create_app
from models import db, Admin, Student, FaceEncoding, ClassSession, Attendance, AttendanceAuditLog
from services.face_engine import face_engine

class BiometricAttendanceSystemTestSuite(unittest.TestCase):
    """
    Comprehensive Test Suite for Biometric Attendance Management System.
    Validates:
      - Unified Authentication (Admin & Student)
      - Role-Based Access Control (RBAC) & Session Management
      - Face Detection (YuNet) & 128D Embeddings (SFace)
      - Facial Emotion Analysis (FER+)
      - Student Biometric Enrollment, Duplicate Face Guard & CRUD
      - Attendance Verification Kiosk & Duplicate Check-in Prevention
      - Manual Correction Audit Trails & Reporting
      - CSV & Excel Export APIs
      - UI Template Integrity & Security Bounds
    """

    @classmethod
    def setUpClass(cls):
        # Prepare reusable base64 test image from test_images/lena.jpg
        lena_img = cv2.imread('test_images/lena.jpg')
        if lena_img is None:
            raise FileNotFoundError("test_images/lena.jpg not found for test suite.")
        _, buffer = cv2.imencode('.jpg', lena_img)
        cls.lena_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode()

        # Prepare dummy non-face image (solid black)
        black_img = np.zeros((320, 320, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', black_img)
        cls.black_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode()

        # Prepare a different synthetic face (blurred lena) to test unrecognized condition if needed
        unrelated_img = cv2.GaussianBlur(lena_img, (51, 51), 0)
        _, buffer = cv2.imencode('.jpg', unrelated_img)
        cls.blurred_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode()

    def setUp(self):
        self.app = create_app({
            'TESTING': True,
            'SQLALCHEMY_DATABASE_URI': 'sqlite:///:memory:',
            'WTF_CSRF_ENABLED': False
        })
        self.client = self.app.test_client()

        with self.app.app_context():
            db.create_all()

            # Seed Admin
            admin = Admin(username='sysadmin')
            admin.set_password('AdminSecurePass2026!')
            db.session.add(admin)

            # Seed Student with password
            student = Student(
                register_number='CS2026001',
                full_name='Test Student Alex',
                department='Computer Science & Engineering',
                class_year='Year 3 - Semester 5',
                email='alex@university.edu',
                active_status=True
            )
            student.set_password('AlexPass123')
            db.session.add(student)

            # Seed Class Session
            sess = ClassSession(
                session_name="Neural Networks & Deep Learning",
                department="Computer Science & Engineering",
                class_year="Year 3 - Semester 5",
                session_date=date.today(),
                start_time=time(9, 0),
                end_time=time(11, 0),
                is_active=True
            )
            db.session.add(sess)
            db.session.commit()

    def tearDown(self):
        with self.app.app_context():
            db.session.remove()
            db.drop_all()

    # =========================================================================
    # MODULE 1: UNIFIED AUTHENTICATION & ACCESS CONTROL
    # =========================================================================

    def test_01_unified_login_page_renders_cleanly(self):
        """Verify unified login page GET response and elements."""
        resp = self.client.get('/login')
        self.assertEqual(resp.status_code, 200)
        html = resp.data.decode('utf-8')
        self.assertIn('Sign In', html)
        self.assertIn('Student', html)
        self.assertIn('Administrator', html)
        self.assertIn('login_role', html)

    def test_02_admin_login_success_and_logout(self):
        """Verify successful admin login via unified portal and session logout."""
        resp = self.client.post('/login', data={
            'login_role': 'admin',
            'admin_username': 'sysadmin',
            'admin_password': 'AdminSecurePass2026!'
        }, follow_redirects=True)
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b'Welcome back, sysadmin!', resp.data)
        self.assertIn(b'Administrative Dashboard', resp.data)

        # Logout
        resp_logout = self.client.get('/logout', follow_redirects=True)
        self.assertEqual(resp_logout.status_code, 200)
        self.assertIn(b'You have been logged out securely', resp_logout.data)

    def test_03_admin_login_invalid_credentials(self):
        """Verify admin login failure on bad password."""
        resp = self.client.post('/login', data={
            'login_role': 'admin',
            'admin_username': 'sysadmin',
            'admin_password': 'WrongPassword!'
        }, follow_redirects=True)
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b'Invalid username or password', resp.data)

    def test_04_student_login_success_and_dashboard_access(self):
        """Verify student login with Register Number and password into student portal."""
        resp = self.client.post('/login', data={
            'login_role': 'student',
            'student_reg_no': 'CS2026001',
            'student_password': 'AlexPass123'
        }, follow_redirects=True)
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b'Welcome back, Test Student Alex!', resp.data)
        self.assertIn(b'STUDENT PORTAL', resp.data)
        self.assertIn(b'CS2026001', resp.data)

    def test_05_student_login_invalid_password(self):
        """Verify rejection of invalid student password."""
        resp = self.client.post('/login', data={
            'login_role': 'student',
            'student_reg_no': 'CS2026001',
            'student_password': 'WrongPass'
        }, follow_redirects=True)
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b'Invalid Student ID or password', resp.data)

    def test_06_deactivated_student_login_blocked(self):
        """Verify that a deactivated student cannot log in."""
        with self.app.app_context():
            student = Student.query.filter_by(register_number='CS2026001').first()
            student.active_status = False
            db.session.commit()

        resp = self.client.post('/login', data={
            'login_role': 'student',
            'student_reg_no': 'CS2026001',
            'student_password': 'AlexPass123'
        }, follow_redirects=True)
        self.assertIn(b'currently deactivated', resp.data)

    def test_07_role_segregation_and_route_guards(self):
        """Verify unauthenticated/student users cannot access admin dashboard."""
        # Unauthenticated access to admin dashboard
        resp = self.client.get('/admin/dashboard', follow_redirects=True)
        self.assertIn(b'Please log in with administrator credentials', resp.data)

        # Student logged in attempting admin route
        self.client.post('/login', data={
            'login_role': 'student',
            'student_reg_no': 'CS2026001',
            'student_password': 'AlexPass123'
        })
        resp = self.client.get('/admin/dashboard', follow_redirects=True)
        self.assertIn(b'Please log in with administrator credentials', resp.data)

    # =========================================================================
    # MODULE 2: STUDENT MANAGEMENT & BIOMETRIC ENROLLMENT
    # =========================================================================

    def test_08_student_enrollment_with_face_encodings(self):
        """Verify admin can register a student with webcam face samples."""
        self.client.post('/login', data={
            'login_role': 'admin',
            'admin_username': 'sysadmin',
            'admin_password': 'AdminSecurePass2026!'
        })

        resp = self.client.post('/admin/students/add', data={
            'register_number': 'CS2026002',
            'full_name': 'Sarah Connor',
            'department': 'Computer Science & Engineering',
            'class_year': 'Year 3 - Semester 5',
            'email': 'sarah@university.edu',
            'face_samples': json.dumps([self.lena_b64, self.lena_b64])
        }, follow_redirects=True)

        self.assertEqual(resp.status_code, 200)
        self.assertIn(b'successfully registered', resp.data)

        with self.app.app_context():
            student = Student.query.filter_by(register_number='CS2026002').first()
            self.assertIsNotNone(student)
            self.assertEqual(student.full_name, 'Sarah Connor')
            self.assertEqual(len(student.encodings), 2)
            emb = student.encodings[0].get_embedding()
            self.assertEqual(len(emb), 128)
            # Default password should match register number
            self.assertTrue(student.check_password('CS2026002'))

    def test_09_duplicate_student_id_prevention(self):
        """Verify duplicate register numbers are rejected."""
        self.client.post('/login', data={
            'login_role': 'admin',
            'admin_username': 'sysadmin',
            'admin_password': 'AdminSecurePass2026!'
        })

        resp = self.client.post('/admin/students/add', data={
            'register_number': 'CS2026001', # Already exists
            'full_name': 'Duplicate Person',
            'department': 'Electrical Engineering',
            'class_year': 'Year 1',
            'face_samples': json.dumps([self.lena_b64])
        }, follow_redirects=True)

        self.assertIn(b'already exists', resp.data)

    def test_10_biometric_duplicate_face_warning(self):
        """Verify that enrolling the same face under a new ID triggers duplicate warning."""
        self.client.post('/login', data={
            'login_role': 'admin',
            'admin_username': 'sysadmin',
            'admin_password': 'AdminSecurePass2026!'
        })

        # Register Student A with Lena face
        self.client.post('/admin/students/add', data={
            'register_number': 'CS2026010',
            'full_name': 'Original Student',
            'department': 'Computer Science & Engineering',
            'class_year': 'Year 3',
            'face_samples': json.dumps([self.lena_b64])
        })

        # Try registering Student B with the same Lena face
        resp = self.client.post('/admin/students/add', data={
            'register_number': 'CS2026011',
            'full_name': 'Impostor / Twin Student',
            'department': 'Computer Science & Engineering',
            'class_year': 'Year 3',
            'face_samples': json.dumps([self.lena_b64])
        }, follow_redirects=True)

        self.assertIn(b'Duplicate Face Warning', resp.data)

    def test_11_student_update_and_re_enrollment(self):
        """Verify updating student information and re-enrolling face biometrics."""
        self.client.post('/login', data={
            'login_role': 'admin',
            'admin_username': 'sysadmin',
            'admin_password': 'AdminSecurePass2026!'
        })

        with self.app.app_context():
            student = Student.query.filter_by(register_number='CS2026001').first()
            s_id = student.id

        # Update details
        resp = self.client.post(f'/admin/students/{s_id}/edit', data={
            'full_name': 'Alex Updated Name',
            'department': 'Information Technology',
            'class_year': 'Year 4 - Semester 7',
            'email': 'alex.updated@university.edu',
            'active_status': '1'
        }, follow_redirects=True)

        self.assertIn(b'updated successfully', resp.data)
        with self.app.app_context():
            updated = db.session.get(Student, s_id)
            self.assertEqual(updated.full_name, 'Alex Updated Name')
            self.assertEqual(updated.department, 'Information Technology')

        # Re-enroll face
        resp_reenroll = self.client.post(f'/admin/students/{s_id}/re-enroll', data={
            'face_samples': json.dumps([self.lena_b64, self.lena_b64])
        }, follow_redirects=True)

        self.assertIn(b'successfully re-enrolled', resp_reenroll.data)
        with self.app.app_context():
            updated = db.session.get(Student, s_id)
            self.assertEqual(len(updated.encodings), 2)

    def test_12_student_status_toggle_and_delete(self):
        """Verify deactivating and deleting a student record."""
        self.client.post('/login', data={
            'login_role': 'admin',
            'admin_username': 'sysadmin',
            'admin_password': 'AdminSecurePass2026!'
        })

        with self.app.app_context():
            student = Student.query.filter_by(register_number='CS2026001').first()
            s_id = student.id

        # Deactivate
        resp = self.client.post(f'/admin/students/{s_id}/toggle-status', follow_redirects=True)
        self.assertIn(b'has been deactivated', resp.data)

        with self.app.app_context():
            self.assertFalse(db.session.get(Student, s_id).active_status)

        # Delete
        resp_del = self.client.post(f'/admin/students/{s_id}/delete', follow_redirects=True)
        self.assertIn(b'permanently removed', resp_del.data)

        with self.app.app_context():
            self.assertIsNone(db.session.get(Student, s_id))

    # =========================================================================
    # MODULE 3: BIOMETRIC ATTENDANCE VERIFICATION & DUPLICATE PREVENTION
    # =========================================================================

    def test_13_attendance_verification_no_face_detected(self):
        """Verify scanner returns NO_FACE when solid black frame is submitted."""
        with self.app.app_context():
            sess = ClassSession.query.first()
            session_id = sess.id

        resp = self.client.post('/api/verify-attendance', json={
            'image': self.black_b64,
            'session_id': session_id
        })
        data = resp.get_json()
        self.assertEqual(resp.status_code, 200)
        self.assertFalse(data['success'])
        self.assertEqual(data['status'], 'NO_FACE')
        self.assertIn('No face detected', data['message'])

    def test_14_attendance_verification_unrecognized_face(self):
        """Verify scanner returns UNRECOGNIZED when an unregistered face appears."""
        with self.app.app_context():
            sess = ClassSession.query.first()
            session_id = sess.id
            # Ensure no student has lena face enrolled yet
            FaceEncoding.query.delete()
            db.session.commit()

        # Send lena face when no encodings are registered
        resp = self.client.post('/api/verify-attendance', json={
            'image': self.lena_b64,
            'session_id': session_id
        })
        data = resp.get_json()
        self.assertEqual(resp.status_code, 200)
        self.assertFalse(data['success'])
        # Either NO_STUDENTS or UNRECOGNIZED
        self.assertIn(data['status'], ['UNRECOGNIZED', 'NO_STUDENTS'])

    def test_15_successful_attendance_marking_and_emotion_capture(self):
        """Verify successful face match, attendance record creation, and emotion detection."""
        # Enroll Lena for Test Student Alex
        self.client.post('/login', data={
            'login_role': 'admin',
            'admin_username': 'sysadmin',
            'admin_password': 'AdminSecurePass2026!'
        })
        with self.app.app_context():
            student = Student.query.filter_by(register_number='CS2026001').first()
            s_id = student.id

        self.client.post(f'/admin/students/{s_id}/re-enroll', data={
            'face_samples': json.dumps([self.lena_b64])
        })

        with self.app.app_context():
            sess = ClassSession.query.first()
            session_id = sess.id

        # Scan face
        resp = self.client.post('/api/verify-attendance', json={
            'image': self.lena_b64,
            'session_id': session_id
        })
        data = resp.get_json()
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(data['success'])
        self.assertEqual(data['status'], 'SUCCESS')
        self.assertEqual(data['student']['register_number'], 'CS2026001')
        self.assertGreater(data['confidence'], 90.0)
        self.assertIn('emotion', data)

        with self.app.app_context():
            att = Attendance.query.filter_by(student_id=s_id, session_id=session_id).first()
            self.assertIsNotNone(att)
            self.assertEqual(att.status, 'Present')
            self.assertIn('Face Recognition', att.verification_method)
            self.assertGreater(att.confidence_or_distance, 0.60)

    def test_16_duplicate_attendance_scan_prevention(self):
        """Verify that scanning a student again in the same class session returns ALREADY_MARKED."""
        # Set up student with Lena face
        with self.app.app_context():
            student = Student.query.filter_by(register_number='CS2026001').first()
            sess = ClassSession.query.first()
            session_id = sess.id
            s_id = student.id

        self.client.post('/login', data={
            'login_role': 'admin',
            'admin_username': 'sysadmin',
            'admin_password': 'AdminSecurePass2026!'
        })
        self.client.post(f'/admin/students/{s_id}/re-enroll', data={
            'face_samples': json.dumps([self.lena_b64])
        })

        # First scan -> Success
        self.client.post('/api/verify-attendance', json={
            'image': self.lena_b64,
            'session_id': session_id
        })

        # Second scan in same session -> ALREADY_MARKED
        resp2 = self.client.post('/api/verify-attendance', json={
            'image': self.lena_b64,
            'session_id': session_id
        })
        data2 = resp2.get_json()
        self.assertTrue(data2['success'])
        self.assertEqual(data2['status'], 'ALREADY_MARKED')
        self.assertTrue(data2['already_marked'])
        self.assertIn('Attendance already recorded', data2['message'])

        # Verify only 1 attendance row exists in DB
        with self.app.app_context():
            count = Attendance.query.filter_by(student_id=s_id, session_id=session_id).count()
            self.assertEqual(count, 1)

    def test_16b_access_denied_unrecognized_and_multiple_faces(self):
        """Verify strict ACCESS DENIED rejection for unregistered faces and multiple faces in frame."""
        with self.app.app_context():
            sess = ClassSession.query.first()
            session_id = sess.id
            student = Student.query.filter_by(register_number='CS2026001').first()
            # Register synthetic orthogonal face encoding for student
            dummy_emb = np.zeros(128, dtype=np.float32)
            dummy_emb[0] = 1.0  # orthogonal to real faces
            FaceEncoding.query.filter_by(student_id=student.id).delete()
            fe = FaceEncoding(student_id=student.id)
            fe.set_embedding(dummy_emb)
            db.session.add(fe)
            db.session.commit()

        # 1. Real Lena face does not match orthogonal embedding -> UNRECOGNIZED
        resp = self.client.post('/api/verify-attendance', json={
            'image': self.lena_b64,
            'session_id': session_id
        })
        data = resp.get_json()
        self.assertEqual(resp.status_code, 200)
        self.assertFalse(data['success'])
        self.assertEqual(data['status'], 'UNRECOGNIZED')
        self.assertIn('ACCESS DENIED – FACE NOT RECOGNIZED', data['message'])

        # Ensure NO attendance was marked
        with self.app.app_context():
            self.assertEqual(Attendance.query.count(), 0)

        # 2. Multi-face frame (side-by-side Lena faces) -> MULTIPLE_FACES
        lena_img = cv2.imread('test_images/lena.jpg')
        two_faces = np.hstack([lena_img, lena_img])
        _, buf = cv2.imencode('.jpg', two_faces)
        two_faces_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode()

        resp_multi = self.client.post('/api/verify-attendance', json={
            'image': two_faces_b64,
            'session_id': session_id
        })
        data_multi = resp_multi.get_json()
        self.assertEqual(resp_multi.status_code, 200)
        self.assertFalse(data_multi['success'])
        self.assertEqual(data_multi['status'], 'MULTIPLE_FACES')
        self.assertIn('ACCESS DENIED – MULTIPLE FACES', data_multi['message'])

    def test_16c_individual_face_isolation_and_no_cross_contamination(self):
        """Verify that when a member scans their face, only that exact person's details are retrieved and recorded."""
        self.client.post('/login', data={
            'login_role': 'admin',
            'admin_username': 'sysadmin',
            'admin_password': 'AdminSecurePass2026!'
        })

        with self.app.app_context():
            # Add Member 1: R. Dharanika
            dharanika = Student(
                register_number='CS2026021',
                full_name='R. Dharanika',
                department='Information Technology',
                class_year='Year 3 - Semester 5',
                email='dharanika@university.edu',
                active_status=True
            )
            dharanika.set_password('PassDharanika123')
            db.session.add(dharanika)

            # Add Member 2: K. Dharshini
            dharshini = Student(
                register_number='CS2026022',
                full_name='K. Dharshini',
                department='Computer Science & Engineering',
                class_year='Year 3 - Semester 5',
                email='dharshini@university.edu',
                active_status=True
            )
            dharshini.set_password('PassDharshini123')
            db.session.add(dharshini)
            db.session.commit()

            dharanika_id = dharanika.id
            dharshini_id = dharshini.id
            sess_id = ClassSession.query.first().id

        # Enroll face for Dharanika
        self.client.post(f'/admin/students/{dharanika_id}/re-enroll', data={
            'face_samples': json.dumps([self.lena_b64])
        })

        # Scan face (Lena)
        resp = self.client.post('/api/verify-attendance', json={
            'image': self.lena_b64,
            'session_id': sess_id
        })
        data = resp.get_json()
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(data['success'])
        self.assertEqual(data['status'], 'SUCCESS')

        # Verify ONLY Dharanika's details are returned
        self.assertEqual(data['student']['name'], 'R. Dharanika')
        self.assertEqual(data['student']['register_number'], 'CS2026021')
        self.assertEqual(data['student']['department'], 'Information Technology')
        self.assertNotEqual(data['student']['name'], 'K. Dharshini')
        self.assertNotEqual(data['student']['register_number'], 'CS2026022')

        # Verify DB records: Dharanika has 1 attendance, Dharshini has 0
        with self.app.app_context():
            att_dharanika = Attendance.query.filter_by(student_id=dharanika_id).all()
            att_dharshini = Attendance.query.filter_by(student_id=dharshini_id).all()
            self.assertEqual(len(att_dharanika), 1)
            self.assertEqual(len(att_dharshini), 0)
            self.assertEqual(att_dharanika[0].student.full_name, 'R. Dharanika')

    # =========================================================================
    # MODULE 4: AUDIT TRAIL, SESSION SCHEDULING & REPORTING
    # =========================================================================

    def test_17_attendance_audit_log_correction(self):
        """Verify admin can manually correct attendance record and generates immutable audit entry."""
        self.client.post('/login', data={
            'login_role': 'admin',
            'admin_username': 'sysadmin',
            'admin_password': 'AdminSecurePass2026!'
        })

        with self.app.app_context():
            student = Student.query.filter_by(register_number='CS2026001').first()
            sess = ClassSession.query.first()
            att = Attendance(
                student_id=student.id,
                session_id=sess.id,
                attendance_date=date.today(),
                check_in_time=time(9, 15),
                status='Present',
                verification_method='Face Recognition',
                confidence_or_distance=0.95
            )
            db.session.add(att)
            db.session.commit()
            att_id = att.id

        # Admin corrects Present -> Absent
        resp = self.client.post(f'/admin/attendance/correct/{att_id}', data={
            'status': 'Absent',
            'reason': 'Student departed premises without approval.'
        }, follow_redirects=True)

        self.assertIn(b'corrected from Present to Absent', resp.data)

        with self.app.app_context():
            updated = db.session.get(Attendance, att_id)
            self.assertEqual(updated.status, 'Absent')
            self.assertTrue(updated.is_manually_corrected)
            self.assertEqual(updated.correction_note, 'Student departed premises without approval.')

            audit = AttendanceAuditLog.query.filter_by(attendance_id=att_id).first()
            self.assertIsNotNone(audit)
            self.assertEqual(audit.previous_status, 'Present')
            self.assertEqual(audit.new_status, 'Absent')
            self.assertEqual(audit.admin_username, 'sysadmin')
            self.assertEqual(audit.reason, 'Student departed premises without approval.')

    def test_18_class_session_creation(self):
        """Verify admin can schedule new class sessions."""
        self.client.post('/login', data={
            'login_role': 'admin',
            'admin_username': 'sysadmin',
            'admin_password': 'AdminSecurePass2026!'
        })

        resp = self.client.post('/admin/sessions', data={
            'session_name': 'Operating Systems Lab',
            'department': 'Computer Science & Engineering',
            'class_year': 'Year 3 - Semester 5',
            'session_date': date.today().strftime('%Y-%m-%d'),
            'start_time': '14:00',
            'end_time': '16:00'
        }, follow_redirects=True)

        self.assertIn(b'scheduled for', resp.data)
        with self.app.app_context():
            sess = ClassSession.query.filter_by(session_name='Operating Systems Lab').first()
            self.assertIsNotNone(sess)
            self.assertEqual(sess.department, 'Computer Science & Engineering')

    def test_19_reports_page_and_export_endpoints(self):
        """Verify summary reports, CSV export, and Excel export."""
        self.client.post('/login', data={
            'login_role': 'admin',
            'admin_username': 'sysadmin',
            'admin_password': 'AdminSecurePass2026!'
        })

        # 1. Reports page
        resp_rep = self.client.get('/admin/reports')
        self.assertEqual(resp_rep.status_code, 200)
        self.assertIn(b'Attendance Analytics', resp_rep.data)

        # 2. CSV Export
        today_str = date.today().strftime('%Y-%m-%d')
        resp_csv = self.client.get(f'/admin/attendance/export/csv?date={today_str}')
        self.assertEqual(resp_csv.status_code, 200)
        self.assertEqual(resp_csv.mimetype, 'text/csv')
        self.assertIn(b'Register Number,Student Name', resp_csv.data)

        # 3. Excel Export
        resp_excel = self.client.get(f'/admin/attendance/export/excel?date={today_str}')
        self.assertEqual(resp_excel.status_code, 200)
        self.assertIn('spreadsheetml', resp_excel.mimetype)

    # =========================================================================
    # MODULE 5: UI & TEMPLATE INTEGRITY ACROSS ALL SCREENS
    # =========================================================================

    def test_20_all_ui_views_render_without_error(self):
        """Verify all portal pages return HTTP 200 and render clean, proper UI."""
        # 1. Face Scanner Kiosk
        resp = self.client.get('/attendance')
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b'Face Attendance Scanner', resp.data)

        # Log in admin to test admin UI screens
        self.client.post('/login', data={
            'login_role': 'admin',
            'admin_username': 'sysadmin',
            'admin_password': 'AdminSecurePass2026!'
        })

        admin_routes = [
            '/admin/dashboard',
            '/admin/students',
            '/admin/students/add',
            '/admin/attendance',
            '/admin/sessions',
            '/admin/reports'
        ]

        for route in admin_routes:
            with self.subTest(route=route):
                r = self.client.get(route)
                self.assertEqual(r.status_code, 200, f"Route {route} failed with status {r.status_code}")
                # Confirm no technical tracebacks or unhandled server errors
                self.assertNotIn(b'Traceback (most recent call last)', r.data)
                self.assertNotIn(b'Internal Server Error', r.data)

if __name__ == '__main__':
    unittest.main()
