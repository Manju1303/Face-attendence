from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from functools import wraps
from models import db, Admin

auth_bp = Blueprint('auth', __name__, url_prefix='/admin')

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('admin_id'):
            flash('Please log in with administrator credentials to access this portal.', 'warning')
            return redirect(url_for('auth.login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if session.get('admin_id'):
        return redirect(url_for('admin.dashboard'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        if not username or not password:
            flash('Please provide both username and password.', 'danger')
            return render_template('admin/login.html')

        admin = Admin.query.filter_by(username=username).first()

        if admin and admin.check_password(password):
            session['admin_id'] = admin.id
            session['admin_username'] = admin.username
            session.permanent = True
            flash(f'Welcome back, {admin.username}!', 'success')
            next_url = request.args.get('next')
            return redirect(next_url or url_for('admin.dashboard'))
        else:
            flash('Invalid username or password. Please try again.', 'danger')

    return render_template('admin/login.html')

@auth_bp.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out securely.', 'info')
    return redirect(url_for('auth.login'))

