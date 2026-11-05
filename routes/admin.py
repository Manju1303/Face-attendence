from flask import Blueprint, render_template, request, redirect, url_for, flash, session, jsonify, Response, send_file
from datetime import datetime, date, timedelta
import io
import csv
import json
import os
from pathlib import Path
from werkzeug.utils import secure_filename
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

from models import db, Admin, Student, FaceEncoding, ClassSession, Attendance, AttendanceAuditLog
from routes.auth import admin_required
from services.face_engine import face_engine
from config import Config

admin_bp = Blueprint('admin', __name__, url_prefix='/admin')

@admin_bp.route('/')
@admin_bp.route('/dashboard')
@admin_required
def dashboard():
    today = date.today()
    
    # 1. Metric: Total registered students
    total_students = Student.query.count()
    active_students = Student.query.filter_by(active_status=True).count()
    
    # 2. Metric: Today's attendance
    today_records = Attendance.query.filter_by(attendance_date=today).all()
    present_today = sum(1 for a in today_records if a.status == 'Present')
    
    # Students absent today (active students minus present students)
    absent_today = max(0, active_students - present_today)
    
    # 3. Metric: Attendance percentage
    attendance_pct = round((present_today / active_students * 100), 1) if active_students > 0 else 0.0
    
    # 4. Recent attendance records (last 10)
    recent_records = Attendance.query.order_by(Attendance.created_at.desc()).limit(10).all()
    
    # 5. Active class sessions today
    today_sessions = ClassSession.query.filter_by(session_date=today).all()
    
    # 6. Chart Data: Past 7 days attendance trends
    trend_labels = []
    trend_present = []
    trend_absent = []
    
    for i in range(6, -1, -1):
        day = today - timedelta(days=i)
        trend_labels.append(day.strftime('%b %d'))
        day_present = Attendance.query.filter_by(attendance_date=day, status='Present').count()
        trend_present.append(day_present)
        trend_absent.append(max(0, active_students - day_present))
        
    # 7. Department breakdown
    dept_query = db.session.query(Student.department, db.func.count(Student.id)).group_by(Student.department).all()
    dept_labels = [d[0] for d in dept_query]
    dept_counts = [d[1] for d in dept_query]

    return render_template(
        'admin/dashboard.html',
        total_students=total_students,
        active_students=active_students,
        present_today=present_today,
        absent_today=absent_today,
        attendance_pct=attendance_pct,
        recent_records=recent_records,
        today_sessions=today_sessions,
        trend_labels=trend_labels,
        trend_present=trend_present,
        trend_absent=trend_absent,
        dept_labels=dept_labels,
        dept_counts=dept_counts,
        today=today
    )

# ----------------- STUDENT REGISTRATION & MANAGEMENT ----------------- #

@admin_bp.route('/students')
@admin_required
def students():
    query = Student.query
    
    search = request.args.get('search', '').strip()
    department = request.args.get('department', '').strip()
    class_year = request.args.get('class_year', '').strip()
    status = request.args.get('status', '').strip()
    
    if search:
        query = query.filter(
            (Student.register_number.ilike(f'%{search}%')) |
            (Student.full_name.ilike(f'%{search}%')) |
            (Student.email.ilike(f'%{search}%'))
        )
    if department:
        query = query.filter_by(department=department)
    if class_year:
        query = query.filter_by(class_year=class_year)
    if status != '':
        query = query.filter_by(active_status=(status == 'active'))
        
    all_students = query.order_by(Student.register_number.asc()).all()
    
    # Unique departments and classes for filter dropdowns
    departments = [d[0] for d in db.session.query(Student.department).distinct().all()]
    classes = [c[0] for c in db.session.query(Student.class_year).distinct().all()]
    
    return render_template(
        'admin/students.html',
        students=all_students,
        departments=departments,
        classes=classes,
        search=search,
        selected_dept=department,
        selected_class=class_year,
        selected_status=status
    )

@admin_bp.route('/students/add', methods=['GET', 'POST'])
@admin_required
def add_student():
    if request.method == 'POST':
        register_number = request.form.get('register_number', '').strip().upper()
        full_name = request.form.get('full_name', '').strip()
        department = request.form.get('department', '').strip()
        class_year = request.form.get('class_year', '').strip()
        email = request.form.get('email', '').strip()
        phone_number = request.form.get('phone_number', '').strip()
        force_register = request.form.get('force_register') == '1'
        
        # Captured samples from webcam (JSON string array of base64 images)
        samples_json = request.form.get('face_samples', '[]')
        
        # Validations
        if not register_number or not full_name or not department or not class_year:
            flash('Register Number, Full Name, Department, and Class are required.', 'danger')
            return render_template('admin/add_student.html')
            
        # Check duplicate register number
        if Student.query.filter_by(register_number=register_number).first():
            flash(f'Student ID / Register Number "{register_number}" already exists. Please choose a unique ID.', 'danger')
            return render_template('admin/add_student.html')
            
        try:
            samples = json.loads(samples_json)
        except Exception:
            samples = []
            
        if not samples or len(samples) < 1:
            flash('At least one clear face sample must be captured via the webcam or uploaded.', 'warning')
            return render_template('admin/add_student.html')
            
        # Process face samples and extract embeddings
        valid_embeddings = []
        best_crop_b64 = None
        
        for sample_b64 in samples:
            img = face_engine.decode_base64_image(sample_b64)
            if img is None:
                continue
            faces = face_engine.detect_faces(img, min_confidence=0.55)
            if not faces:
                continue
                
            # Pick highest confidence face
            face_data = max(faces, key=lambda f: f['confidence'])
            embedding = face_engine.extract_embedding(img, face_data)
            if embedding and len(embedding) == 128:
                valid_embeddings.append(embedding)
                if best_crop_b64 is None:
                    best_crop_b64 = sample_b64
                    
        if len(valid_embeddings) == 0:
            flash('No usable face detected in the captured samples. Please position your face clearly with good lighting and try again.', 'danger')
            return render_template('admin/add_student.html')
            
        # Check duplicate face match against existing enrolled students
        existing_students = Student.query.all()
        matching_student, match_score = face_engine.check_duplicate_face(
            valid_embeddings[0], existing_students, threshold=0.58
        )
        
        if matching_student and not force_register:
            flash(
                f'Warning: Captured face closely matches an already registered student: '
                f'{matching_student.full_name} ({matching_student.register_number}) '
                f'with {match_score*100:.1f}% similarity. Check the override box if this is intended.',
                'warning'
            )
            return render_template(
                'admin/add_student.html',
                matched_student=matching_student,
                match_score=round(match_score * 100, 1),
                register_number=register_number,
                full_name=full_name,
                department=department,
                class_year=class_year,
                email=email,
                samples_json=samples_json
            )
            
        # Save Student Record
        new_student = Student(
            register_number=register_number,
            full_name=full_name,
            department=department,
            class_year=class_year,
            email=email if email else f"{register_number.lower()}@university.edu",
            phone_number=phone_number if phone_number else "+91 98765 43210",
            active_status=True
        )
        
        # Save photo file if sample available
        if best_crop_b64:
            os.makedirs(Config.STUDENT_PHOTOS_DIR, exist_ok=True)
            photo_fname = f"{register_number}_{int(datetime.utcnow().timestamp())}.jpg"
            photo_path = Config.STUDENT_PHOTOS_DIR / photo_fname
            img_bgr = face_engine.decode_base64_image(best_crop_b64)
            if img_bgr is not None:
                import cv2
                cv2.imwrite(str(photo_path), img_bgr)
                new_student.photo_filename = photo_fname
                
        db.session.add(new_student)
        db.session.flush() # Populate new_student.id
        
        # Store Face Embeddings
        for idx, emb in enumerate(valid_embeddings):
            fe = FaceEncoding(
                student_id=new_student.id,
                sample_index=idx,
                model_version="sface_v1_128d"
            )
            fe.set_embedding(emb)
            db.session.add(fe)
            
        db.session.commit()
        flash(f'Student "{full_name}" ({register_number}) successfully registered with {len(valid_embeddings)} biometric face sample(s)!', 'success')
        return redirect(url_for('admin.students'))
        
    return render_template('admin/add_student.html')

@admin_bp.route('/students/<int:student_id>')
@admin_required
def student_detail(student_id):
    student = db.get_or_404(Student, student_id)
    attendance_records = Attendance.query.filter_by(student_id=student.id).order_by(Attendance.attendance_date.desc()).all()
    
    total_classes = len(attendance_records)
    present_count = sum(1 for a in attendance_records if a.status == 'Present')
    rate = round((present_count / total_classes * 100), 1) if total_classes > 0 else 0.0
    
    return render_template(
        'admin/student_detail.html',
        student=student,
        attendance_records=attendance_records,
        total_classes=total_classes,
        present_count=present_count,
        attendance_rate=rate
    )

@admin_bp.route('/students/<int:student_id>/edit', methods=['GET', 'POST'])
@admin_required
def edit_student(student_id):
    student = db.get_or_404(Student, student_id)
    
    if request.method == 'POST':
        register_number = request.form.get('register_number', '').strip().upper()
        full_name = request.form.get('full_name', '').strip()
        department = request.form.get('department', '').strip()
        class_year = request.form.get('class_year', '').strip()
        email = request.form.get('email', '').strip()
        phone_number = request.form.get('phone_number', '').strip()
        active_status = request.form.get('active_status') == '1'
        
        # Check duplicate register_number
        existing = Student.query.filter(Student.register_number == register_number, Student.id != student.id).first()
        if existing:
            flash(f'Register Number "{register_number}" is already used by another student.', 'danger')
            return render_template('admin/edit_student.html', student=student)
            
        student.register_number = register_number
        student.full_name = full_name
        student.department = department
        student.class_year = class_year
        student.email = email if email else f"{register_number.lower()}@university.edu"
        student.phone_number = phone_number if phone_number else (student.phone_number or "+91 98765 43210")
        student.active_status = active_status
        
        db.session.commit()
        flash(f'Student "{full_name}" information updated successfully.', 'success')
        return redirect(url_for('admin.student_detail', student_id=student.id))
        
    return render_template('admin/edit_student.html', student=student)

@admin_bp.route('/students/<int:student_id>/re-enroll', methods=['GET', 'POST'])
@admin_required
def re_enroll_student(student_id):
    student = db.get_or_404(Student, student_id)
    
    if request.method == 'POST':
        samples_json = request.form.get('face_samples', '[]')
        try:
            samples = json.loads(samples_json)
        except Exception:
            samples = []
            
        if not samples:
            flash('Please capture at least one clear face sample to re-enrol.', 'warning')
            return render_template('admin/re_enroll.html', student=student)
            
        valid_embeddings = []
        best_crop_b64 = None
        for sample_b64 in samples:
            img = face_engine.decode_base64_image(sample_b64)
            if img is None:
                continue
            faces = face_engine.detect_faces(img, min_confidence=0.55)
            if not faces:
                continue
            face_data = max(faces, key=lambda f: f['confidence'])
            emb = face_engine.extract_embedding(img, face_data)
            if emb and len(emb) == 128:
                valid_embeddings.append(emb)
                if best_crop_b64 is None:
                    best_crop_b64 = sample_b64
                    
        if not valid_embeddings:
            flash('No usable face detected. Please ensure good lighting and try again.', 'danger')
            return render_template('admin/re_enroll.html', student=student)
            
        # Remove old face encodings
        FaceEncoding.query.filter_by(student_id=student.id).delete()
        
        # Save new embeddings
        for idx, emb in enumerate(valid_embeddings):
            fe = FaceEncoding(student_id=student.id, sample_index=idx, model_version="sface_v1_128d")
            fe.set_embedding(emb)
            db.session.add(fe)
            
        if best_crop_b64:
            os.makedirs(Config.STUDENT_PHOTOS_DIR, exist_ok=True)
            photo_fname = f"{student.register_number}_{int(datetime.utcnow().timestamp())}.jpg"
            photo_path = Config.STUDENT_PHOTOS_DIR / photo_fname
            img_bgr = face_engine.decode_base64_image(best_crop_b64)
            if img_bgr is not None:
                import cv2
                cv2.imwrite(str(photo_path), img_bgr)
                student.photo_filename = photo_fname
                
        db.session.commit()
        flash(f'Biometric face data for "{student.full_name}" has been successfully re-enrolled with {len(valid_embeddings)} new samples.', 'success')
        return redirect(url_for('admin.student_detail', student_id=student.id))
        
    return render_template('admin/re_enroll.html', student=student)

@admin_bp.route('/students/<int:student_id>/toggle-status', methods=['POST'])
@admin_required
def toggle_student_status(student_id):
    student = db.get_or_404(Student, student_id)
    student.active_status = not student.active_status
    db.session.commit()
    state = "activated" if student.active_status else "deactivated"
    flash(f'Student "{student.full_name}" has been {state}.', 'info')
    return redirect(url_for('admin.students'))

@admin_bp.route('/students/<int:student_id>/delete', methods=['POST'])
@admin_required
def delete_student(student_id):
    student = db.get_or_404(Student, student_id)
    name = student.full_name
    reg = student.register_number
    db.session.delete(student)
    db.session.commit()
    flash(f'Student "{name}" ({reg}) and all corresponding biometric data have been permanently removed.', 'info')
    return redirect(url_for('admin.students'))

# ----------------- ATTENDANCE MANAGEMENT ----------------- #

@admin_bp.route('/attendance')
@admin_required
def attendance():
    date_str = request.args.get('date', date.today().strftime('%Y-%m-%d'))
    try:
        selected_date = datetime.strptime(date_str, '%Y-%m-%d').date()
    except Exception:
        selected_date = date.today()
        
    session_id = request.args.get('session_id', type=int)
    department = request.args.get('department', '').strip()
    class_year = request.args.get('class_year', '').strip()
    status_filter = request.args.get('status', '').strip()
    
    # Query attendance records
    att_query = Attendance.query.filter_by(attendance_date=selected_date)
    if session_id:
        att_query = att_query.filter_by(session_id=session_id)
        
    records = att_query.join(Student).order_by(Attendance.check_in_time.desc()).all()
    
    # Filter by department or class
    if department:
        records = [r for r in records if r.student and r.student.department == department]
    if class_year:
        records = [r for r in records if r.student and r.student.class_year == class_year]
    if status_filter:
        records = [r for r in records if r.status == status_filter]
        
    # Get all active students for computing absent list
    active_students_query = Student.query.filter_by(active_status=True)
    if department:
        active_students_query = active_students_query.filter_by(department=department)
    if class_year:
        active_students_query = active_students_query.filter_by(class_year=class_year)
    total_active_students = active_students_query.all()
    
    present_student_ids = {r.student_id for r in records if r.status == 'Present'}
    absent_students = [s for s in total_active_students if s.id not in present_student_ids]
    
    # Filter options
    sessions = ClassSession.query.filter_by(session_date=selected_date).all()
    departments = [d[0] for d in db.session.query(Student.department).distinct().all()]
    classes = [c[0] for c in db.session.query(Student.class_year).distinct().all()]
    
    return render_template(
        'admin/attendance.html',
        records=records,
        absent_students=absent_students,
        selected_date=selected_date.strftime('%Y-%m-%d'),
        selected_session=session_id,
        selected_dept=department,
        selected_class=class_year,
        selected_status=status_filter,
        sessions=sessions,
        departments=departments,
        classes=classes,
        present_count=len(present_student_ids),
        absent_count=len(absent_students)
    )

@admin_bp.route('/attendance/correct/<int:attendance_id>', methods=['POST'])
@admin_required
def correct_attendance(attendance_id):
    record = db.get_or_404(Attendance, attendance_id)
    new_status = request.form.get('status')
    reason = request.form.get('reason', '').strip()
    
    if not new_status or not reason:
        flash('Correction reason and target status are both required for the audit trail.', 'danger')
        return redirect(request.referrer or url_for('admin.attendance'))
        
    old_status = record.status
    record.status = new_status
    record.is_manually_corrected = True
    record.corrected_by_admin = session.get('admin_username', 'admin')
    record.correction_note = reason
    record.corrected_at = datetime.utcnow()
    
    # Record Audit Log
    audit = AttendanceAuditLog(
        attendance_id=record.id,
        action=f"Status changed: {old_status} -> {new_status}",
        previous_status=old_status,
        new_status=new_status,
        reason=reason,
        admin_username=session.get('admin_username', 'admin')
    )
    db.session.add(audit)
    db.session.commit()
    
    flash(f'Attendance status for "{record.student.full_name}" corrected from {old_status} to {new_status}.', 'success')
    return redirect(request.referrer or url_for('admin.attendance'))

@admin_bp.route('/attendance/export/csv')
@admin_required
def export_csv():
    date_str = request.args.get('date', date.today().strftime('%Y-%m-%d'))
    selected_date = datetime.strptime(date_str, '%Y-%m-%d').date()
    
    records = Attendance.query.filter_by(attendance_date=selected_date).join(Student).order_by(Student.register_number.asc()).all()
    
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        'Register Number', 'Student Name', 'Department', 'Class/Year', 
        'Date', 'Check-In Time', 'Status', 'Verification Method', 
        'Confidence/Similarity', 'Manually Corrected', 'Audit Note'
    ])
    
    for r in records:
        writer.writerow([
            r.student.register_number if r.student else '',
            r.student.full_name if r.student else '',
            r.student.department if r.student else '',
            r.student.class_year if r.student else '',
            r.attendance_date.strftime('%Y-%m-%d'),
            r.check_in_time.strftime('%H:%M:%S') if r.check_in_time else '',
            r.status,
            r.verification_method,
            f"{r.confidence_or_distance:.4f}" if r.confidence_or_distance else '',
            'Yes' if r.is_manually_corrected else 'No',
            r.correction_note or ''
        ])
        
    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={"Content-disposition": f"attachment; filename=attendance_{date_str}.csv"}
    )

@admin_bp.route('/attendance/export/excel')
@admin_required
def export_excel():
    date_str = request.args.get('date', date.today().strftime('%Y-%m-%d'))
    selected_date = datetime.strptime(date_str, '%Y-%m-%d').date()
    
    records = Attendance.query.filter_by(attendance_date=selected_date).join(Student).order_by(Student.register_number.asc()).all()
    
    wb = Workbook()
    ws = wb.active
    ws.title = f"Attendance {date_str}"
    
    # Header styling
    header_fill = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
    header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    thin_border = Border(
        left=Side(style='thin', color='D9D9D9'),
        right=Side(style='thin', color='D9D9D9'),
        top=Side(style='thin', color='D9D9D9'),
        bottom=Side(style='thin', color='D9D9D9')
    )
    
    headers = [
        'Register No', 'Full Name', 'Department', 'Class/Year', 
        'Date', 'Check-In Time', 'Status', 'Verification Method', 
        'Confidence', 'Corrected?', 'Correction Note'
    ]
    ws.append(headers)
    
    for col_num, _ in enumerate(headers, 1):
        cell = ws.cell(row=1, column=col_num)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center")
        
    for row_idx, r in enumerate(records, 2):
        ws.append([
            r.student.register_number if r.student else '',
            r.student.full_name if r.student else '',
            r.student.department if r.student else '',
            r.student.class_year if r.student else '',
            r.attendance_date.strftime('%Y-%m-%d'),
            r.check_in_time.strftime('%H:%M:%S') if r.check_in_time else '',
            r.status,
            r.verification_method,
            round(r.confidence_or_distance, 4) if r.confidence_or_distance else '',
            'Yes' if r.is_manually_corrected else 'No',
            r.correction_note or ''
        ])
        for col_idx in range(1, len(headers) + 1):
            c = ws.cell(row=row_idx, column=col_idx)
            c.border = thin_border
            if col_idx in [1, 5, 6, 7, 10]:
                c.alignment = Alignment(horizontal="center")
                
    # Auto-adjust column widths
    for col in ws.columns:
        max_len = max(len(str(cell.value or '')) for cell in col)
        col_letter = col[0].column_letter
        ws.column_dimensions[col_letter].width = max(max_len + 3, 12)
        
    out_stream = io.BytesIO()
    wb.save(out_stream)
    out_stream.seek(0)
    
    return send_file(
        out_stream,
        as_attachment=True,
        download_name=f"attendance_report_{date_str}.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

@admin_bp.route('/reports')
@admin_required
def reports():
    # Overall summary reports
    total_students = Student.query.filter_by(active_status=True).count()
    
    # Monthly summaries
    now = datetime.now()
    first_of_month = date(now.year, now.month, 1)
    month_records = Attendance.query.filter(Attendance.attendance_date >= first_of_month).all()
    
    student_stats = []
    for s in Student.query.filter_by(active_status=True).all():
        student_records = [r for r in month_records if r.student_id == s.id]
        total_sessions = len(student_records)
        present = sum(1 for r in student_records if r.status == 'Present')
        pct = round((present / total_sessions * 100), 1) if total_sessions > 0 else 0.0
        student_stats.append({
            'student': s,
            'sessions': total_sessions,
            'present': present,
            'pct': pct
        })
        
    return render_template(
        'admin/reports.html',
        student_stats=student_stats,
        total_students=total_students,
        current_month=now.strftime('%B %Y')
    )

# ----------------- CLASS SESSIONS ----------------- #

@admin_bp.route('/sessions', methods=['GET', 'POST'])
@admin_required
def sessions():
    if request.method == 'POST':
        session_name = request.form.get('session_name', '').strip()
        department = request.form.get('department', '').strip()
        class_year = request.form.get('class_year', '').strip()
        session_date_str = request.form.get('session_date', '').strip()
        start_time_str = request.form.get('start_time', '').strip()
        end_time_str = request.form.get('end_time', '').strip()
        
        if not all([session_name, department, class_year, session_date_str, start_time_str, end_time_str]):
            flash('All session fields are required.', 'danger')
            return redirect(url_for('admin.sessions'))
            
        s_date = datetime.strptime(session_date_str, '%Y-%m-%d').date()
        s_time = datetime.strptime(start_time_str, '%H:%M').time()
        e_time = datetime.strptime(end_time_str, '%H:%M').time()
        
        new_session = ClassSession(
            session_name=session_name,
            department=department,
            class_year=class_year,
            session_date=s_date,
            start_time=s_time,
            end_time=e_time,
            created_by=session.get('admin_id'),
            is_active=True
        )
        db.session.add(new_session)
        db.session.commit()
        flash(f'Class session "{session_name}" scheduled for {s_date} ({s_time.strftime("%H:%M")} - {e_time.strftime("%H:%M")}).', 'success')
        return redirect(url_for('admin.sessions'))
        
    all_sessions = ClassSession.query.order_by(ClassSession.session_date.desc(), ClassSession.start_time.asc()).all()
    departments = [d[0] for d in db.session.query(Student.department).distinct().all()]
    classes = [c[0] for c in db.session.query(Student.class_year).distinct().all()]
    
    return render_template(
        'admin/sessions.html',
        sessions=all_sessions,
        departments=departments,
        classes=classes,
        today=date.today().strftime('%Y-%m-%d')
    )

@admin_bp.route('/sessions/<int:session_id>/toggle', methods=['POST'])
@admin_required
def toggle_session(session_id):
    s = db.get_or_404(ClassSession, session_id)
    s.is_active = not s.is_active
    db.session.commit()
    flash(f'Session "{s.session_name}" is now {"Active" if s.is_active else "Inactive"}.', 'info')
    return redirect(url_for('admin.sessions'))

# ----------------- LIVE PRE-VALIDATION API ----------------- #

@admin_bp.route('/api/validate-sample', methods=['POST'])
@admin_required
def api_validate_sample():
    data = request.get_json() or {}
    image_bgr = face_engine.decode_base64_image(data.get('image', ''))
    if image_bgr is None:
        return jsonify({'valid': False, 'message': 'Invalid image data received.'}), 400
        
    faces = face_engine.detect_faces(image_bgr, min_confidence=0.55)
    if not faces:
        return jsonify({
            'valid': False, 
            'message': 'No face detected. Please ensure your face is inside the frame with adequate lighting.'
        })
        
    face_data = max(faces, key=lambda f: f['confidence'])
    embedding = face_engine.extract_embedding(image_bgr, face_data)
    if not embedding:
        return jsonify({'valid': False, 'message': 'Unable to compute 128D face features.'})
        
    # Check duplicate against existing students
    students = Student.query.all()
    matching_student, match_score = face_engine.check_duplicate_face(embedding, students, threshold=0.58)
    
    warn = None
    if matching_student:
        warn = f"Face matches enrolled student {matching_student.full_name} ({matching_student.register_number}) with {match_score*100:.1f}% similarity."
        
    return jsonify({
        'valid': True,
        'confidence': round(face_data['confidence'] * 100, 1),
        'bbox': face_data['bbox'],
        'warning': warn
    })
