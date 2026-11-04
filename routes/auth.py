from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from functools import wraps
from models import db, Admin, Student

auth_bp = Blueprint('auth', __name__)

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('admin_id'):
            flash('Please log in with administrator credentials to access this portal.', 'warning')
            return redirect(url_for('auth.login', role='admin', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    # If already logged in, redirect accordingly
    if session.get('admin_id'):
        return redirect(url_for('admin.dashboard'))
    if session.get('student_id'):
        return redirect(url_for('student.dashboard'))

    active_role = request.args.get('role', 'student')

    if request.method == 'POST':
        login_role = request.form.get('login_role', 'student')
        active_role = login_role

        if login_role == 'admin':
            username = (request.form.get('admin_username') or request.form.get('username') or '').strip()
            password = (request.form.get('admin_password') or request.form.get('password') or '').strip()

            if not username or not password:
                flash('Invalid username or password.', 'danger')
                return render_template('auth/login.html', active_role='admin')

            admin = Admin.query.filter_by(username=username).first()
            if admin and admin.check_password(password):
                session.clear()
                session['admin_id'] = admin.id
                session['admin_username'] = admin.username
                session['role'] = 'admin'
                session.permanent = True
                flash(f'Welcome back, {admin.username}!', 'success')
                next_url = request.args.get('next')
                return redirect(next_url or url_for('admin.dashboard'))
            else:
                flash('Invalid username or password.', 'danger')
                return render_template('auth/login.html', active_role='admin')

        else: # student
            reg_no = (request.form.get('student_reg_no') or request.form.get('register_number') or '').strip().upper()
            password = (request.form.get('student_password') or request.form.get('password') or '').strip()

            if not reg_no or not password:
                flash('Please provide your Student ID / Register Number and password.', 'danger')
                return render_template('auth/login.html', active_role='student')

            student = Student.query.filter(Student.register_number.ilike(reg_no)).first()
            if student and student.check_password(password):
                if not student.active_status:
                    flash('Your student account is currently deactivated. Please contact the administrator.', 'warning')
                    return render_template('auth/login.html', active_role='student')

                session.clear()
                session['student_id'] = student.id
                session['student_name'] = student.full_name
                session['student_reg'] = student.register_number
                session['role'] = 'student'
                session.permanent = True
                flash(f'Welcome back, {student.full_name}!', 'success')
                return redirect(url_for('student.dashboard'))
            else:
                flash('Invalid Student ID or password.', 'danger')
                return render_template('auth/login.html', active_role='student')

    return render_template('auth/login.html', active_role=active_role)

@auth_bp.route('/admin/login', methods=['GET', 'POST'])
def admin_login_redirect():
    if request.method == 'POST':
        # Delegate to login handler with login_role='admin'
        username = (request.form.get('admin_username') or request.form.get('username') or '').strip()
        password = (request.form.get('admin_password') or request.form.get('password') or '').strip()
        if not username or not password:
            flash('Invalid username or password.', 'danger')
            return render_template('auth/login.html', active_role='admin')
        admin = Admin.query.filter_by(username=username).first()
        if admin and admin.check_password(password):
            session.clear()
            session['admin_id'] = admin.id
            session['admin_username'] = admin.username
            session['role'] = 'admin'
            session.permanent = True
            flash(f'Welcome back, {admin.username}!', 'success')
            next_url = request.args.get('next')
            return redirect(next_url or url_for('admin.dashboard'))
        else:
            flash('Invalid username or password.', 'danger')
            return render_template('auth/login.html', active_role='admin')
    return redirect(url_for('auth.login', role='admin'))

@auth_bp.route('/logout')
@auth_bp.route('/admin/logout')
def logout():
    session.clear()
    flash('You have been logged out securely.', 'info')
    return redirect(url_for('auth.login'))
