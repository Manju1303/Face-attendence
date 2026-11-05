from flask import Blueprint, render_template, request, jsonify, current_app
from datetime import datetime, date, time
from models import db, Student, FaceEncoding, ClassSession, Attendance
from services.face_engine import face_engine
from config import Config

attendance_bp = Blueprint('attendance', __name__)

@attendance_bp.route('/')
@attendance_bp.route('/attendance')
def scan_page():
    # Fetch active sessions for today
    today = date.today()
    active_sessions = ClassSession.query.filter_by(
        session_date=today,
        is_active=True
    ).order_by(ClassSession.start_time.asc()).all()

    # If no session scheduled for today, fallback to any active session or create a default session
    if not active_sessions:
        default_session = ClassSession.query.filter_by(is_active=True).first()
        if not default_session:
            # Create standard daily session
            default_session = ClassSession(
                session_name="Daily General Attendance",
                department="All Departments",
                class_year="All Classes",
                session_date=today,
                start_time=time(8, 0),
                end_time=time(18, 0),
                is_active=True
            )
            db.session.add(default_session)
            db.session.commit()
        active_sessions = [default_session]

    return render_template(
        'attendance/scan.html',
        sessions=active_sessions,
        today=today.strftime('%A, %B %d, %Y')
    )

@attendance_bp.route('/api/verify-attendance', methods=['POST'])
def verify_attendance():
    """
    Real-time face verification API endpoint.
    Receives webcam frame snapshot (base64) and session ID.
    Performs face detection, 128D embedding extraction, emotion classification,
    and server-side duplicate prevention.
    """
    data = request.get_json() or {}
    image_b64 = data.get('image', '')
    session_id = data.get('session_id')

    if not image_b64:
        return jsonify({
            'success': False,
            'status': 'ERROR',
            'message': 'No image frame received from camera.'
        }), 400

    # 1. Decode image
    image_bgr = face_engine.decode_base64_image(image_b64)
    if image_bgr is None:
        return jsonify({
            'success': False,
            'status': 'ERROR',
            'message': 'Camera frame could not be decoded.'
        }), 400

    # 2. Detect face
    faces = face_engine.detect_faces(image_bgr, min_confidence=0.55)
    if not faces:
        return jsonify({
            'success': False,
            'status': 'NO_FACE',
            'message': 'No face detected. Please position your face clearly in the camera view.'
        })

    # Pick the most prominent/highest confidence face
    face_data = max(faces, key=lambda f: f['confidence'])
    bbox = face_data['bbox']

    # 3. Extract 128D Face Embedding
    query_embedding = face_engine.extract_embedding(image_bgr, face_data)
    if not query_embedding:
        return jsonify({
            'success': False,
            'status': 'UNCERTAIN',
            'message': 'Face detected but features could not be analyzed. Please adjust your angle and lighting.'
        })

    # 4. Predict facial emotion (from FER+ deep model)
    dominant_emotion, emotion_conf, emotion_probs = face_engine.predict_emotion(image_bgr, bbox)

    # 5. Query enrolled active students
    enrolled_students = Student.query.filter_by(active_status=True).all()
    if not enrolled_students:
        return jsonify({
            'success': False,
            'status': 'NO_STUDENTS',
            'message': 'No students are currently enrolled in the system. Please register students in the Admin portal.'
        })

    # 6. Match face against database embeddings
    threshold = Config.FACE_SIMILARITY_THRESHOLD
    uncertainty_margin = Config.FACE_UNCERTAINTY_MARGIN

    match_status, matched_student, similarity_score, candidate = face_engine.find_best_match(
        query_embedding,
        enrolled_students,
        threshold=threshold,
        uncertainty_margin=uncertainty_margin
    )

    # Handle uncertain match
    if match_status == 'UNCERTAIN':
        return jsonify({
            'success': False,
            'status': 'UNCERTAIN',
            'message': 'Recognition uncertain. Please look straight into the camera, avoid backlight, and try again.',
            'confidence': round(similarity_score * 100, 1),
            'bbox': bbox
        })

    # Handle no match
    if match_status == 'NO_MATCH' or matched_student is None:
        return jsonify({
            'success': False,
            'status': 'UNRECOGNIZED',
            'message': 'Face Not Registered. Attendance is not marked.',
            'confidence': round(similarity_score * 100, 1) if similarity_score > 0 else 0.0,
            'bbox': bbox
        })

    # 7. Match Successful - Identify Class Session
    today = date.today()
    if session_id:
        target_session = db.session.get(ClassSession, session_id)
    else:
        # Auto-resolve session
        target_session = ClassSession.query.filter_by(session_date=today, is_active=True).first()
        if not target_session:
            target_session = ClassSession.query.filter_by(is_active=True).first()

    if not target_session:
        return jsonify({
            'success': False,
            'status': 'NO_SESSION',
            'message': 'No active class session available for attendance recording.'
        })

    # 8. Server-Side Duplicate Check: Ensure student receives attendance ONLY ONCE per session
    existing_attendance = Attendance.query.filter_by(
        student_id=matched_student.id,
        session_id=target_session.id,
        attendance_date=today
    ).first()

    if existing_attendance:
        return jsonify({
            'success': True,
            'status': 'ALREADY_MARKED',
            'already_marked': True,
            'message': f'Attendance already recorded for {matched_student.full_name} ({matched_student.register_number}) in this session at {existing_attendance.check_in_time.strftime("%I:%M %p")}.',
            'student': {
                'id': matched_student.id,
                'name': matched_student.full_name,
                'register_number': matched_student.register_number,
                'department': matched_student.department,
                'class_year': matched_student.class_year,
                'photo': matched_student.photo_filename
            },
            'session_name': target_session.session_name,
            'check_in_time': existing_attendance.check_in_time.strftime('%I:%M:%S %p'),
            'confidence': round(existing_attendance.confidence_or_distance * 100, 1) if existing_attendance.confidence_or_distance else 95.0,
            'emotion': dominant_emotion,
            'bbox': bbox
        })

    # 9. Record New Attendance in Database
    now = datetime.now()
    new_attendance = Attendance(
        student_id=matched_student.id,
        session_id=target_session.id,
        attendance_date=today,
        check_in_time=now.time(),
        status='Present',
        verification_method='Face Recognition (YuNet + SFace)',
        confidence_or_distance=similarity_score
    )

    try:
        db.session.add(new_attendance)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        # In case of concurrent request collision on unique constraint
        existing_attendance = Attendance.query.filter_by(
            student_id=matched_student.id,
            session_id=target_session.id,
            attendance_date=today
        ).first()
        if existing_attendance:
            return jsonify({
                'success': True,
                'status': 'ALREADY_MARKED',
                'already_marked': True,
                'message': f'Attendance already recorded at {existing_attendance.check_in_time.strftime("%I:%M %p")}.',
                'student': matched_student.to_dict()
            })
        return jsonify({
            'success': False,
            'status': 'ERROR',
            'message': f'Database error while recording attendance: {str(e)}'
        }), 500

    # 10. Return Clear Immediate Success Response
    return jsonify({
        'success': True,
        'status': 'SUCCESS',
        'already_marked': False,
        'message': f'Attendance successfully verified & recorded for {matched_student.full_name}!',
        'student': {
            'id': matched_student.id,
            'name': matched_student.full_name,
            'register_number': matched_student.register_number,
            'department': matched_student.department,
            'class_year': matched_student.class_year,
            'photo': matched_student.photo_filename
        },
        'session_name': target_session.session_name,
        'check_in_time': now.strftime('%I:%M:%S %p'),
        'confidence': round(similarity_score * 100, 1),
        'emotion': dominant_emotion,
        'bbox': bbox
    })
