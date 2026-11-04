from flask import Blueprint, render_template, redirect, url_for, flash, session
from functools import wraps
from datetime import date
from models import db, Student, Attendance, ClassSession

student_bp = Blueprint('student', __name__, url_prefix='/student')

def student_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('student_id'):
            flash('Please log in with your Student ID to access the student portal.', 'warning')
            return redirect(url_for('auth.login', role='student'))
        return f(*args, **kwargs)
    return decorated_function

@student_bp.route('/dashboard')
@student_required
def dashboard():
    student_id = session.get('student_id')
    student = db.session.get(Student, student_id)
    if not student:
        session.clear()
        return redirect(url_for('auth.login'))

    today = date.today()
    # Fetch personal attendance records
    records = Attendance.query.filter_by(student_id=student.id).order_by(Attendance.attendance_date.desc()).all()
    
    total_sessions = len(records)
    present_count = sum(1 for r in records if r.status == 'Present')
    rate = round((present_count / total_sessions * 100), 1) if total_sessions > 0 else 0.0

    # Check today's status
    today_record = Attendance.query.filter_by(student_id=student.id, attendance_date=today).first()
    today_sessions = ClassSession.query.filter_by(session_date=today, is_active=True).all()

    return render_template(
        'student/dashboard.html',
        student=student,
        records=records,
        total_sessions=total_sessions,
        present_count=present_count,
        attendance_rate=rate,
        today_record=today_record,
        today_sessions=today_sessions,
        today=today
    )
