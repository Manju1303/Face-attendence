# Biometric Attendance Management System Using Face Recognition (BioAttendAI)

[![GitHub Pages](https://img.shields.io/badge/Live_Demo-GitHub_Pages-2ea44f?style=for-the-badge&logo=github)](https://manju1303.github.io/Face-attendence/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0%20DNN-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1-000000?style=for-the-badge&logo=flask&logoColor=white)](https://palletsprojects.com/p/flask/)

> **🌐 Live Interactive Web Client:** [**https://manju1303.github.io/Face-attendence/**](https://manju1303.github.io/Face-attendence/)

A production-ready, full-stack biometric attendance management system built with Python, Flask, OpenCV Deep Neural Networks (YuNet + SFace), and real-time facial emotion analytics.

---

## 🌟 Key Features

### 1. Student Attendance Kiosk (`/attendance`)
- **Browser-Based Live Webcam Feed:** Captures live video stream directly through `navigator.mediaDevices.getUserMedia()`.
- **Discrete Camera States:** Explicit camera status tracking (`CAMERA_INITIALIZING`, `CAMERA_PERMISSION_REQUIRED`, `CAMERA_READY`, `CAMERA_SCANNING`, `CAMERA_DISCONNECTED`, `CAMERA_PERMISSION_DENIED`, `CAMERA_NOT_FOUND`, `CAMERA_IN_USE`, `CAMERA_ERROR`).
- **Complete Camera Controls:** Four functional controls: *Start Camera*, *Stop Camera*, *Restart Camera*, and *Scan Face Now*.
- **Live IST Clock (`Asia/Kolkata`):** Independent real-time clock displaying current time (AM/PM), weekday, day, month, and year updated every second.
- **Face Detection & Alignment:** Real-time facial landmark detection (eyes, nose, mouth corners) via OpenCV YuNet DNN.
- **128-Dimensional Face Embeddings:** High-precision face representation using OpenCV SFace model.
- **Zero-Blank Verification Card:** Displays registered student photo, full name, register number, department, class/year, email, phone, check-in time (IST), and match similarity %. Never displays false success with empty dashes.
- **Ambiguity Guard:** Matches where the difference between top 2 enrolled candidates is < 5% are safely rejected as `AMBIGUOUS_MATCH` to prevent false positives.
- **Strict Server-Side Duplicate Blocking:** A student can receive attendance only once per configured session. Repeated scans return `ALREADY_MARKED` with original check-in time preserved.
- **Uncertainty & Rejection Handling:**
  - *"No face detected. Please position your face clearly."*
  - *"ACCESS DENIED – FACE NOT RECOGNIZED. Scanned face does not match any registered member."*
  - *"Recognition uncertain. Please adjust lighting and try again."*
  - *"ACCESS DENIED – AMBIGUOUS MATCH. Please adjust lighting and face camera directly."*

### 2. Admin Dashboard & Portal (`/admin/dashboard`)
- **Secure Authentication:** Password hashing (`pbkdf2:sha256`) and session protection via `@admin_required`.
- **Executive Dashboard Metrics:**
  - Total registered students & active enrolled count.
  - Students present today vs absent today.
  - Overall daily attendance rate percentage.
  - 7-Day interactive trend chart (Present vs Absent) using Chart.js.
  - Department enrollment doughnut distribution chart.
- **Student Registration (`/admin/students/add`):**
  - Unique Register Number / Student ID validation.
  - Multi-sample biometric webcam capture station (3–5 samples from different angles).
  - Pre-validation API verifies face clarity before submission.
  - **Duplicate Face Detection Warning:** Warns admin if the captured face matches an existing student to prevent duplicate registrations.
- **Student Management (`/admin/students`):**
  - Search by Register No or Name; filter by department, class, and active status.
  - Comprehensive student profile view with attendance history.
  - Biometric re-enrolment station with administrative authorization.
  - One-click student activation/deactivation and permanent deletion modal.
- **Attendance Management & Auditing (`/admin/attendance`):**
  - Filter records by date, class session, department, and status.
  - Tabbed view for **Present Students** and **Absent Students**.
  - **Audit Correction Engine:** Allows admins to adjust attendance status (`Present`, `Absent`, `Late`, `Excused`) with mandatory audit notes saved in `attendance_audit_logs`.
  - **Reports & Export:** Instant download of filtered attendance records as **CSV** or styled **Excel (`.xlsx`)** spreadsheets using `openpyxl`.
- **Class Session Scheduler (`/admin/sessions`):**
  - Configure class sessions with department, class year, session date, and start/end time slots.

---

## 📂 Project Structure

```text
face attendence/
├── app.py                      # Flask Application Factory & Server Entrypoint
├── config.py                   # Central Configuration (SQLite, PostgreSQL, Thresholds)
├── models.py                   # SQLAlchemy Relational Models (Admin, Student, FaceEncoding, etc.)
├── seed_demo_data.py           # Demo Data Seeder (Default admin, sessions, sample student)
├── test_system.py              # Automated End-to-End Unit & Integration Test Suite
├── run.bat                     # Windows One-Click Launch Script
├── requirements.txt            # Python Dependencies
├── models_data/                # Deep Learning Models & XML Weights
│   ├── face_detection_yunet_2023mar.onnx    # YuNet Face Detector
│   ├── face_recognition_sface_2021dec.onnx   # SFace 128D Embedding Model
│   ├── emotion_ferplus.onnx                  # FER2013 Facial Emotion Classifier
│   ├── haarcascade_frontalface_default.xml   # Haar Cascade Face Detector
│   ├── deploy.prototxt                       # SSD Caffe Face Detector Config
│   └── res10_300x300_ssd_iter_140000.caffemodel # SSD Caffe Weights
├── routes/
│   ├── auth.py                 # Admin Login, Logout, and Session Guards
│   ├── admin.py                # Dashboard, Student CRUD, Attendance Auditing & Exports
│   └── attendance.py           # Student Kiosk & /api/verify-attendance Verification API
├── services/
│   └── face_engine.py          # Unified Computer Vision & Emotion Recognition Engine
├── static/
│   └── css/
│       └── custom.css          # Custom Modern UI Styling & Biometric Camera Animations
├── templates/
│   ├── base.html               # Base Template with Bootstrap 5 & Navigation
│   ├── admin/
│   │   ├── login.html          # Admin Login Page
│   │   ├── dashboard.html      # Metrics, Charts, and Recent Check-ins
│   │   ├── students.html       # Student Directory & Filters
│   │   ├── add_student.html    # Biometric Multi-Sample Webcam Registration
│   │   ├── student_detail.html # Student Profile & Historical Attendance Log
│   │   ├── edit_student.html   # Update Student Details
│   │   ├── re_enroll.html      # Re-enrolment Station
│   │   ├── attendance.html     # Attendance Log, Present/Absent Tabs & Correction Modal
│   │   ├── sessions.html       # Class Sessions Scheduler
│   │   └── reports.html        # Monthly Analytics & Cumulative Attendance %
│   └── attendance/
│       └── scan.html           # Student Biometric Webcam Scanner
└── uploads/
    └── students/               # Enrolled Student Profile Photos
```

---

## 🧠 Computer Vision & Deep Learning Models Architecture

The project incorporates an optimized computer vision architecture and deep neural network models:
1. **`haarcascade_frontalface_default.xml`:** Classical Haar Cascade face detector for lightweight legacy CPU environments.
2. **`deploy.prototxt` & `res10_300x300_ssd_iter_140000.caffemodel`:** Caffe Deep Neural Network SSD face detector for resilient detection under variable poses.
3. **FER2013-derived Emotion Classifier (`emotion_ferplus.onnx`):** Implements facial emotion tracking during attendance check-in (logging emotions like *Happy*, *Neutral*, *Attentive*, etc.).
4. **OpenCV YuNet + SFace (128D Embedding):** State-of-the-art L2-normalized feature representation ensuring high throughput and sub-second matching across large student cohorts on standard CPU hardware.

---

## ⚙️ Face Recognition Threshold Calibration

The face recognition similarity metric uses **Cosine Similarity** between two 128-dimensional unit-length embedding vectors:

$$\text{Similarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$

| Metric Range | Verification Outcome | Action Taken |
| :--- | :--- | :--- |
| **Similarity $\ge 0.60$** | **Definite Match** | Attendance is granted and logged in the database. |
| **$0.52 \le \text{Similarity} < 0.60$** | **Uncertain Match** | Attendance is withheld. Student is asked to readjust lighting and try again. |
| **$\text{Similarity} < 0.52$** | **No Match** | Display: *"Face not recognized. Please contact the administrator."* |

These thresholds can be configured in `config.py` or through the environment variables:
- `FACE_SIMILARITY_THRESHOLD=0.60`
- `FACE_UNCERTAINTY_MARGIN=0.08`

---

## 🚀 Installation & Setup on Windows

### Prerequisites
- Python 3.9, 3.10, or 3.11 installed.
- A functional webcam (built-in or USB).

### 1. Clone or Open Workspace
Open a PowerShell or Command Prompt terminal in the project directory:
```powershell
cd "c:\Users\Jkk\OneDrive\Desktop\face attendence"
```

### 2. Activate Virtual Environment & Install Dependencies
```powershell
# Create venv if not already created
python -m venv venv

# Activate venv
.\venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### 3. Initialize Database & Seed Demo Data
```powershell
python seed_demo_data.py
```

### 4. Run Automated Test Suite
```powershell
python test_system.py
```
*(All 5 tests verify login, student registration, duplicate detection, biometric attendance check-in, and audit logging.)*

### 5. Start the Application Server
Run the startup script:
```powershell
.\run.bat
```
Or directly with Python:
```powershell
python app.py
```

---

## 🌐 Access Portals & Credentials

- **Admin Dashboard:** [http://127.0.0.1:5000/admin/login](http://127.0.0.1:5000/admin/login)
  - **Username:** `admin`
  - **Password:** `admin123`
- **Student Attendance Kiosk:** [http://127.0.0.1:5000/attendance](http://127.0.0.1:5000/attendance)

---

## 🗄️ Database Architecture & PostgreSQL Migration

The database schema is managed via SQLAlchemy ORM and consists of 6 relational tables:

1. **`admins`:** `id`, `username`, `password_hash`, `created_at`
2. **`students`:** `id`, `register_number` (unique, indexed), `full_name`, `department`, `class_year`, `email`, `active_status`, `photo_filename`, `created_at`
3. **`face_encodings`:** `id`, `student_id` (foreign key), `embedding_data` (128D JSON array), `sample_index`, `model_version`, `created_at`
4. **`class_sessions`:** `id`, `session_name`, `department`, `class_year`, `session_date`, `start_time`, `end_time`, `created_by`, `is_active`, `created_at`
5. **`attendance`:** `id`, `student_id` (foreign key), `session_id` (foreign key), `attendance_date`, `check_in_time`, `check_in_timestamp` (UTC datetime), `status`, `verification_method`, `confidence_or_distance`, `is_manually_corrected`, `correction_note`, `created_at`, `updated_at`
   - *Unique Constraint:* `(student_id, session_id, attendance_date)` ensures zero duplicate submissions per session.
6. **`attendance_audit_logs`:** `id`, `attendance_id`, `action`, `previous_status`, `new_status`, `reason`, `admin_username`, `timestamp`

### Migrating from SQLite to PostgreSQL
To switch to a production PostgreSQL database, install `psycopg2-binary`:
```powershell
pip install psycopg2-binary
```
And set the `DATABASE_URL` environment variable:
```powershell
$env:DATABASE_URL="postgresql://username:password@localhost:5432/attendance_db"
python app.py
```
SQLAlchemy handles all schema creation, relationships, and queries seamlessly across both engines without modifying application code.

