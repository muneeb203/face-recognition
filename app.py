import os
import pickle
import csv
import requests
import json
from io import BytesIO
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, redirect, url_for, Response, session, flash
import face_recognition
import cv2
import numpy as np
import webbrowser
import pandas as pd
import threading
import time
import hashlib
import secrets

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)  # Generate a secure secret key for sessions

# --- Configurable parameters ---
ENCODINGS_FILE = "known_faces.pkl"
ATTENDANCE_FILE = "attendance.csv"
STUDENTS_CSV = "students.csv"  # For downloaded spreadsheet data
CAPTURED_FACES_DIR = "captured_faces"  # Directory to store captured face images
USERS_FILE = "users.json"  # File to store user credentials

# Create necessary directories
if not os.path.exists(CAPTURED_FACES_DIR):
    os.makedirs(CAPTURED_FACES_DIR)

# Create attendance file if it doesn't exist
if not os.path.exists(ATTENDANCE_FILE):
    with open(ATTENDANCE_FILE, "w") as f:
        f.write("Roll Number,Name,Date,Time,Status\n")

# Initialize global variables
known_face_encodings = []
known_face_metadata = []

# Class session variables
current_session = {
    "active": False,
    "start_time": None,
    "course_code": None,
    "duration_minutes": 90,  # Default 1.5 hours (90 minutes)
    "present_threshold_minutes": 10,  # Mark as present if within first 10 minutes
    "late_threshold_minutes": 30,  # Mark as late if within first 30 minutes
    "processed_faces": set()  # Set to track already processed faces in auto mode
}

# User management functions
def load_users():
    """Load users from JSON file"""
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    else:
        # Create default admin user if file doesn't exist
        default_users = {
            "users": [
                {
                    "username": "admin",
                    "password": hash_password("admin123"),
                    "role": "teacher",
                    "name": "Admin User"
                }
            ]
        }
        save_users(default_users)
        return default_users

def save_users(users_data):
    """Save users to JSON file"""
    with open(USERS_FILE, 'w') as f:
        json.dump(users_data, f, indent=4)

def hash_password(password):
    """Hash a password for storing"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(stored_password, provided_password):
    """Verify a stored password against one provided by user"""
    return stored_password == hash_password(provided_password)

def authenticate_user(username, password):
    """Authenticate a user"""
    users_data = load_users()
    for user in users_data["users"]:
        if user["username"] == username and verify_password(user["password"], password):
            return user
    return None

def register_user(username, password, role, name, roll_no=None):
    """Register a new user"""
    users_data = load_users()
    
    # Check if username already exists
    for user in users_data["users"]:
        if user["username"] == username:
            return False, "Username already exists"
    
    # Create new user
    new_user = {
        "username": username,
        "password": hash_password(password),
        "role": role,
        "name": name
    }
    
    # Add roll number for students
    if role == "student" and roll_no:
        new_user["roll_no"] = roll_no
    
    users_data["users"].append(new_user)
    save_users(users_data)
    return True, "User registered successfully"

def get_user_by_username(username):
    """Get user data by username"""
    users_data = load_users()
    for user in users_data["users"]:
        if user["username"] == username:
            return user
    return None

def save_encodings():
    """Save face encodings to disk"""
    with open(ENCODINGS_FILE, 'wb') as f:
        pickle.dump((known_face_encodings, known_face_metadata), f)
    return f"Face data saved to {ENCODINGS_FILE}"

def load_encodings():
    """Load face encodings from disk"""
    global known_face_encodings, known_face_metadata
    try:
        with open(ENCODINGS_FILE, 'rb') as f:
            known_face_encodings, known_face_metadata = pickle.load(f)
        return f"Loaded {len(known_face_encodings)} face(s) from {ENCODINGS_FILE}"
    except FileNotFoundError:
        known_face_encodings = []
        known_face_metadata = []
        return f"No existing face data found at {ENCODINGS_FILE}"

def determine_attendance_status(session_start_time):
    """Determine attendance status based on current time and session start time"""
    now = datetime.now()
    time_diff = now - session_start_time
    minutes_late = time_diff.total_seconds() / 60
    
    if minutes_late <= current_session["present_threshold_minutes"]:
        return "Present"
    elif minutes_late <= current_session["late_threshold_minutes"]:
        return "Late"
    else:
        return "Very Late"

def mark_attendance(roll_no, name):
    """Record attendance for a student"""
    now = datetime.now()
    date = now.strftime('%Y-%m-%d')
    time = now.strftime('%H:%M:%S')
    
    # Determine status based on session time if a session is active
    if current_session["active"] and current_session["start_time"]:
        status = determine_attendance_status(current_session["start_time"])
    else:
        status = "Present"  # Default if no session is active
    
    # Append attendance record to CSV file
    with open(ATTENDANCE_FILE, "a") as f:
        f.write(f"{roll_no},{name},{date},{time},{status}\n")
    
    return {
        "roll_no": roll_no,
        "name": name,
        "date": date,
        "time": time,
        "status": status
    }

def get_registered_students():
    """Get list of all registered students"""
    return known_face_metadata

def get_attendance_records():
    """Get all attendance records"""
    records = []
    try:
        with open(ATTENDANCE_FILE, "r") as f:
            lines = f.readlines()
        
        headers = lines[0].strip().split(',')
        
        for i, line in enumerate(lines):
            if i == 0:  # Skip header
                continue
            data = line.strip().split(',')
            if len(data) >= 5:
                record = {
                    "roll_no": data[0],
                    "name": data[1],
                    "date": data[2],
                    "time": data[3],
                    "status": data[4]
                }
                records.append(record)
    except FileNotFoundError:
        pass
    
    return records

def delete_student(roll_no):
    """Delete a student from the system"""
    global known_face_encodings, known_face_metadata
    
    for i, metadata in enumerate(known_face_metadata):
        if metadata['roll_no'] == roll_no:
            known_face_encodings.pop(i)
            known_face_metadata.pop(i)
            save_encodings()
            return {
                "success": True,
                "message": f"Student with roll number {roll_no} has been deleted"
            }
    
    return {
        "success": False, 
        "message": f"Student with roll number {roll_no} not found"
    }

def download_image_from_drive(url):
    """Download image from Google Drive URL"""
    try:
        # For Google Drive links, we need to use a different approach
        # This is a simplified version and might need adjustments for your specific URLs
        if "drive.google.com" in url:
            # Extract file ID from URL
            if "/file/d/" in url:
                file_id = url.split("/file/d/")[1].split("/")[0]
            elif "id=" in url:
                file_id = url.split("id=")[1].split("&")[0]
            else:
                return None
                
            # Create direct download link
            download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
            
            # Download the image
            response = requests.get(download_url)
            if response.status_code == 200:
                return BytesIO(response.content)
        
        # For regular URLs
        response = requests.get(url)
        if response.status_code == 200:
            return BytesIO(response.content)
            
    except Exception as e:
        print(f"Error downloading image: {e}")
    
    return None

def register_student_from_image(roll_no, name, image_url):
    """Register a student using an image URL"""
    global known_face_encodings, known_face_metadata
    
    # Check if student already exists
    for metadata in known_face_metadata:
        if metadata['roll_no'] == roll_no:
            return {
                "success": False,
                "message": f"Student with roll number {roll_no} already exists"
            }
    
    # Download the image
    image_data = download_image_from_drive(image_url)
    if image_data is None:
        return {
            "success": False,
            "message": f"Failed to download image for {name} (Roll No: {roll_no})"
        }
    
    try:
        # Load the image
        image = face_recognition.load_image_file(image_data)
        
        # Find faces in the image
        face_locations = face_recognition.face_locations(image)
        
        if not face_locations:
            return {
                "success": False,
                "message": f"No face detected in the image for {name} (Roll No: {roll_no})"
            }
        
        # Get face encodings
        face_encodings = face_recognition.face_encodings(image, face_locations)
        
        if not face_encodings:
            return {
                "success": False,
                "message": f"Failed to encode face for {name} (Roll No: {roll_no})"
            }
        
        # Save the encoding and metadata
        known_face_encodings.append(face_encodings[0])
        known_face_metadata.append({
            'roll_no': roll_no,
            'name': name
        })
        
        # Save to disk
        save_encodings()
        
        return {
            "success": True,
            "message": f"Successfully registered {name} (Roll No: {roll_no})"
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Error processing image for {name} (Roll No: {roll_no}): {str(e)}"
        }

def import_students_from_spreadsheet(spreadsheet_url=None):
    """Import students from Google Spreadsheet or local CSV"""
    results = {
        "success": 0,
        "failed": 0,
        "messages": []
    }
    
    try:
        # If a Google Sheets URL is provided, download it as CSV
        if spreadsheet_url and "docs.google.com/spreadsheets" in spreadsheet_url:
            # Extract the spreadsheet ID
            sheet_id = spreadsheet_url.split("/d/")[1].split("/")[0]
            csv_export_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
            
            response = requests.get(csv_export_url)
            if response.status_code == 200:
                # Save the CSV locally
                with open(STUDENTS_CSV, 'wb') as f:
                    f.write(response.content)
            else:
                results["messages"].append(f"Failed to download spreadsheet: HTTP {response.status_code}")
                return results
        
        # Check if we have a local CSV file
        if not os.path.exists(STUDENTS_CSV):
            results["messages"].append("No student data file found")
            return results
        
        # Read the CSV file
        df = pd.read_csv(STUDENTS_CSV)
        
        # Check for required columns (case-insensitive)
        required_columns = ['roll number', 'name', 'image']
        df_columns_lower = [col.lower() for col in df.columns]
        
        # Map the actual column names to our expected names
        column_mapping = {}
        for req_col in required_columns:
            for i, col in enumerate(df_columns_lower):
                if req_col in col:
                    column_mapping[req_col] = df.columns[i]
        
        # Check if all required columns were found
        missing_columns = [col for col in required_columns if col not in column_mapping]
        if missing_columns:
            results["messages"].append(f"Missing required columns: {', '.join(missing_columns)}")
            return results
        
        # Process each student
        for _, row in df.iterrows():
            try:
                # Get values using the mapped column names
                roll_no_col = column_mapping['roll number']
                name_col = column_mapping['name']
                image_col = column_mapping['image']
                
                # Skip rows without roll number, name or image URL
                if pd.isna(row[roll_no_col]) or pd.isna(row[name_col]) or pd.isna(row[image_col]):
                    continue
                
                roll_no = str(row[roll_no_col]).strip()
                name = str(row[name_col]).strip()
                image_url = str(row[image_col]).strip()
                
                # Register the student
                result = register_student_from_image(roll_no, name, image_url)
                
                if result["success"]:
                    results["success"] += 1
                else:
                    results["failed"] += 1
                    
                results["messages"].append(result["message"])
                
            except Exception as e:
                results["failed"] += 1
                results["messages"].append(f"Error processing row: {str(e)}")
        
        return results
        
    except Exception as e:
        results["messages"].append(f"Error importing students: {str(e)}")
        return results

def start_class_session(course_code, duration_minutes=90):
    """Start a new class session"""
    global current_session
    
    current_session = {
        "active": True,
        "start_time": datetime.now(),
        "course_code": course_code,
        "duration_minutes": duration_minutes,
        "present_threshold_minutes": 10,
        "late_threshold_minutes": 30,
        "processed_faces": set()
    }
    
    # Schedule session end
    end_time = datetime.now() + timedelta(minutes=duration_minutes)
    
    return {
        "success": True,
        "message": f"Class session for {course_code} started successfully",
        "start_time": current_session["start_time"].strftime('%Y-%m-%d %H:%M:%S'),
        "end_time": end_time.strftime('%Y-%m-%d %H:%M:%S'),
        "duration_minutes": duration_minutes
    }

def end_class_session():
    """End the current class session"""
    global current_session
    
    if not current_session["active"]:
        return {
            "success": False,
            "message": "No active class session to end"
        }
    
    course_code = current_session["course_code"]
    start_time = current_session["start_time"]
    duration = current_session["duration_minutes"]
    
    # Reset session
    current_session = {
        "active": False,
        "start_time": None,
        "course_code": None,
        "duration_minutes": 90,
        "present_threshold_minutes": 10,
        "late_threshold_minutes": 30,
        "processed_faces": set()
    }
    
    return {
        "success": True,
        "message": f"Class session for {course_code} ended successfully",
        "start_time": start_time.strftime('%Y-%m-%d %H:%M:%S'),
        "duration_minutes": duration
    }

def get_session_status():
    """Get the current session status"""
    if not current_session["active"]:
        return {
            "active": False,
            "message": "No active class session"
        }
    
    now = datetime.now()
    elapsed = now - current_session["start_time"]
    elapsed_minutes = elapsed.total_seconds() / 60
    remaining_minutes = current_session["duration_minutes"] - elapsed_minutes
    
    return {
        "active": True,
        "course_code": current_session["course_code"],
        "start_time": current_session["start_time"].strftime('%Y-%m-%d %H:%M:%S'),
        "elapsed_minutes": round(elapsed_minutes, 1),
        "remaining_minutes": round(remaining_minutes, 1),
        "duration_minutes": current_session["duration_minutes"],
        "present_threshold_minutes": current_session["present_threshold_minutes"],
        "late_threshold_minutes": current_session["late_threshold_minutes"]
    }

# Initialize by loading encodings and users
init_message = load_encodings()
users_data = load_users()

# Video camera class for streaming
class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.face_detected = False
        self.current_frame = None
        self.auto_capture_mode = False
        self.last_recognition_time = time.time() - 10  # Initialize with offset to allow immediate recognition
        self.recognition_cooldown = 5  # Seconds between recognition attempts
        
    def __del__(self):
        self.video.release()
        
    def get_frame(self):
        success, frame = self.video.read()
        if not success:
            return None
            
        self.current_frame = frame
        
        # Find faces in the frame
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        
        # Draw rectangle around faces
        for (top, right, bottom, left) in face_locations:
            # Scale back up face locations
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            self.face_detected = True
            
            # If in auto capture mode and cooldown has passed, try to recognize the face
            if self.auto_capture_mode and time.time() - self.last_recognition_time > self.recognition_cooldown:
                threading.Thread(target=self.auto_recognize_face).start()
                self.last_recognition_time = time.time()  # Update immediately to prevent multiple threads
        
        if not face_locations:
            self.face_detected = False
            
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
    
    def set_auto_capture_mode(self, enabled):
        """Enable or disable automatic face capture mode"""
        self.auto_capture_mode = enabled
        return self.auto_capture_mode
    
    def auto_recognize_face(self):
        """Automatically recognize a face and mark attendance if in a session"""
        global current_session
        
        # Update the last recognition time
        self.last_recognition_time = time.time()
        
        # Only proceed if a session is active
        if not current_session["active"]:
            return None
        
        # Capture and recognize the face
        face_encoding = self.capture_face()
        if face_encoding is None:
            return None
        
        # Compare with known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            confidence = 1 - face_distances[best_match_index]
            confidence_threshold = 0.6
            
            if matches[best_match_index] and confidence > confidence_threshold:
                matched_metadata = known_face_metadata[best_match_index]
                roll_no = matched_metadata['roll_no']
                name = matched_metadata['name']
                
                # Check if this face has already been processed in this session
                if roll_no in current_session["processed_faces"]:
                    print(f"Face already processed in this session: {name} ({roll_no})")
                    return None
                
                # Mark attendance
                attendance_record = mark_attendance(roll_no, name)
                
                # Add to processed faces
                current_session["processed_faces"].add(roll_no)
                
                # Save the captured face
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                face_filename = f"{roll_no}_{timestamp}.jpg"
                face_path = os.path.join(CAPTURED_FACES_DIR, face_filename)
                
                # Extract and save the face region
                if self.current_frame is not None:
                    small_frame = cv2.resize(self.current_frame, (0, 0), fx=0.25, fy=0.25)
                    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                    face_locations = face_recognition.face_locations(rgb_small_frame)
                    
                    if face_locations:
                        top, right, bottom, left = face_locations[0]
                        top *= 4
                        right *= 4
                        bottom *= 4
                        left *= 4
                        
                        face_image = self.current_frame[top:bottom, left:right]
                        cv2.imwrite(face_path, face_image)
                
                print(f"Auto-marked attendance for {name} ({roll_no}) with status: {attendance_record['status']}")
                return attendance_record
        
        return None
        
    def capture_face(self):
        if self.current_frame is not None and self.face_detected:
            # Process the current frame to get face encoding
            small_frame = cv2.resize(self.current_frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_small_frame)
            
            if face_locations:
                # Scale back to original size for encoding
                face_location = face_locations[0]
                scaled_location = (face_location[0]*4, face_location[1]*4, 
                                 face_location[2]*4, face_location[3]*4)
                
                rgb_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
                encodings = face_recognition.face_encodings(rgb_frame, [scaled_location])
                
                if encodings:
                    return encodings[0]
        
        return None

# Global video camera instance
video_camera = None

def gen(camera):
    while True:
        frame = camera.get_frame()
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# Authentication decorator
def login_required(f):
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            flash('Please log in to access this page', 'error')
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

def teacher_required(f):
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            flash('Please log in to access this page', 'error')
            return redirect(url_for('login', next=request.url))
        if session['user']['role'] != 'teacher':
            flash('You do not have permission to access this page', 'error')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = authenticate_user(username, password)
        
        if user:
            session['user'] = user
            flash(f'Welcome, {user["name"]}!', 'success')
            next_page = request.args.get('next')
            if next_page:
                return redirect(next_page)
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password', 'error')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    flash('You have been logged out', 'success')
    return redirect(url_for('index'))

@app.route('/register_account', methods=['GET', 'POST'])
def register_account():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        role = request.form.get('role')
        name = request.form.get('name')
        roll_no = request.form.get('roll_no') if role == 'student' else None
        
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return render_template('register_account.html')
        
        success, message = register_user(username, password, role, name, roll_no)
        
        if success:
            flash(message, 'success')
            return redirect(url_for('login'))
        else:
            flash(message, 'error')
    
    return render_template('register_account.html')

@app.route('/register')
@login_required
def register():
    return render_template('register.html')

@app.route('/student_register')
@login_required
def student_register():
    # Only allow students to register themselves
    if session['user']['role'] == 'student':
        return render_template('student_register.html', student=session['user'])
    else:
        # Teachers can register any student
        return redirect(url_for('register'))

@app.route('/attendance')
@teacher_required
def attendance():
    return render_template('attendance.html')

@app.route('/reports')
@teacher_required
def reports():
    return render_template('reports.html')

@app.route('/import')
@teacher_required
def import_page():
    return render_template('import.html')

@app.route('/session')
@teacher_required
def session_page():
    return render_template('session.html')

@app.route('/video_feed')
@login_required
def video_feed():
    global video_camera
    if video_camera is None:
        video_camera = VideoCamera()
    return Response(gen(video_camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/register_student', methods=['POST'])
@login_required
def register_student_api():
    global video_camera, known_face_encodings, known_face_metadata
    
    # If student is registering themselves
    if session['user']['role'] == 'student':
        roll_no = session['user'].get('roll_no')
        name = session['user'].get('name')
        
        if not roll_no:
            return jsonify({
                "success": False,
                "message": "No roll number associated with your account"
            })
    else:
        # Teacher registering a student
        roll_no = request.form.get('roll_no')
        name = request.form.get('name')
    
    # Check if student already exists
    for metadata in known_face_metadata:
        if metadata['roll_no'] == roll_no:
            return jsonify({
                "success": False,
                "message": f"Student with roll number {roll_no} already exists"
            })
    
    if video_camera is None:
        video_camera = VideoCamera()
    
    face_encoding = video_camera.capture_face()
    
    if face_encoding is not None:
        # Save the encoding and metadata
        known_face_encodings.append(face_encoding)
        known_face_metadata.append({
            'roll_no': roll_no,
            'name': name
        })
        
        # Save to disk
        save_encodings()
        return jsonify({
            "success": True,
            "message": f"Successfully registered {name} (Roll No: {roll_no})"
        })
    else:
        return jsonify({
            "success": False,
            "message": "No face detected. Please try again."
        })

@app.route('/api/get_students', methods=['GET'])
@login_required
def get_students_api():
    students = get_registered_students()
    return jsonify({"students": students})

@app.route('/api/get_attendance', methods=['GET'])
@login_required
def get_attendance_api():
    records = get_attendance_records()
    
    # If student, only return their own records
    if session['user']['role'] == 'student' and 'roll_no' in session['user']:
        student_roll = session['user']['roll_no']
        records = [r for r in records if r['roll_no'] == student_roll]
    
    return jsonify({"records": records})

@app.route('/api/delete_student', methods=['POST'])
@teacher_required
def delete_student_api():
    roll_no = request.form.get('roll_no')
    result = delete_student(roll_no)
    return jsonify(result)

@app.route('/api/take_attendance', methods=['POST'])
@teacher_required
def take_attendance_api():
    global video_camera, known_face_encodings, known_face_metadata
    
    if not known_face_encodings:
        return jsonify({
            "success": False,
            "message": "No registered faces found. Please register at least one student first."
        })
    
    if video_camera is None:
        video_camera = VideoCamera()
    
    face_encoding = video_camera.capture_face()
    
    if face_encoding is not None:
        # Compare face with known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            confidence = 1 - face_distances[best_match_index]
            confidence_threshold = 0.6
            
            if matches[best_match_index] and confidence > confidence_threshold:
                matched_metadata = known_face_metadata[best_match_index]
                roll_no = matched_metadata['roll_no']
                name = matched_metadata['name']
                
                attendance_record = mark_attendance(roll_no, name)
                
                # Save the captured face
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                face_filename = f"{roll_no}_{timestamp}.jpg"
                face_path = os.path.join(CAPTURED_FACES_DIR, face_filename)
                
                # Extract and save the face region
                if video_camera.current_frame is not None:
                    small_frame = cv2.resize(video_camera.current_frame, (0, 0), fx=0.25, fy=0.25)
                    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                    face_locations = face_recognition.face_locations(rgb_small_frame)
                    
                    if face_locations:
                        top, right, bottom, left = face_locations[0]
                        top *= 4
                        right *= 4
                        bottom *= 4
                        left *= 4
                        
                        face_image = video_camera.current_frame[top:bottom, left:right]
                        cv2.imwrite(face_path, face_image)
                
                return jsonify({
                    "success": True,
                    "message": f"Attendance marked for {name} (Roll No: {roll_no})",
                    "student": {
                        **attendance_record,
                        "confidence": float(confidence)
                    }
                })
            else:
                return jsonify({
                    "success": False,
                    "message": "Face not recognized with sufficient confidence."
                })
        else:
            return jsonify({
                "success": False,
                "message": "No registered faces to compare with."
            })
    else:
        return jsonify({
            "success": False,
            "message": "No face detected. Please try again."
        })

@app.route('/api/import_students', methods=['POST'])
@teacher_required
def import_students_api():
    spreadsheet_url = request.form.get('spreadsheet_url')
    
    # If URL is provided, use it, otherwise use the default CSV file
    if spreadsheet_url:
        results = import_students_from_spreadsheet(spreadsheet_url)
    else:
        results = import_students_from_spreadsheet()
    
    return jsonify(results)

@app.route('/api/import_from_google', methods=['POST'])
@teacher_required
def import_from_google_api():
    # Use the Google Sheets URL from the request
    spreadsheet_url = "https://docs.google.com/spreadsheets/d/1GLy4a5mMzrOfItuE2JvZujqDjn_oNrNuKfkTgfrj4AY/edit?usp=sharing"
    results = import_students_from_spreadsheet(spreadsheet_url)
    return jsonify(results)

@app.route('/api/start_session', methods=['POST'])
@teacher_required
def start_session_api():
    course_code = request.form.get('course_code', 'Unknown Course')
    duration_minutes = int(request.form.get('duration_minutes', 90))
    present_threshold = int(request.form.get('present_threshold', 10))
    late_threshold = int(request.form.get('late_threshold', 30))
    
    # Start the session
    result = start_class_session(course_code, duration_minutes)
    
    # Update thresholds
    current_session["present_threshold_minutes"] = present_threshold
    current_session["late_threshold_minutes"] = late_threshold
    
    # Enable auto capture mode
    global video_camera
    if video_camera is None:
        video_camera = VideoCamera()
    
    video_camera.set_auto_capture_mode(True)
    
    return jsonify(result)

@app.route('/api/end_session', methods=['POST'])
@teacher_required
def end_session_api():
    # Disable auto capture mode
    global video_camera
    if video_camera is not None:
        video_camera.set_auto_capture_mode(False)
    
    # End the session
    result = end_class_session()
    return jsonify(result)

@app.route('/api/session_status', methods=['GET'])
@login_required
def session_status_api():
    status = get_session_status()
    return jsonify(status)

@app.route('/api/toggle_auto_capture', methods=['POST'])
@teacher_required
def toggle_auto_capture_api():
    enabled = request.form.get('enabled', 'true').lower() == 'true'
    
    global video_camera
    if video_camera is None:
        video_camera = VideoCamera()
    
    auto_capture_enabled = video_camera.set_auto_capture_mode(enabled)
    
    return jsonify({
        "success": True,
        "auto_capture_enabled": auto_capture_enabled
    })

if __name__ == '__main__':
    # Try to import students from the spreadsheet on startup
    print("Attempting to import students from Google Sheets...")
    results = import_students_from_spreadsheet("https://docs.google.com/spreadsheets/d/1GLy4a5mMzrOfItuE2JvZujqDjn_oNrNuKfkTgfrj4AY/edit?usp=sharing")
    print(f"Import results: {results['success']} successful, {results['failed']} failed")
    for message in results["messages"]:
        print(f"- {message}")
    
    # Open browser automatically
    webbrowser.open('http://127.0.0.1:5000')
    # Then run the app
    app.run(debug=True, port=5000)
