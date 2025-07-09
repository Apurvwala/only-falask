# main.py
import glob
import json
import os
import random
import smtplib
import threading
import time
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List
import io # For handling image data in memory

import numpy as np
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64

# Import the face_recognition library
import face_recognition
# Import Pillow for image manipulation (replaces OpenCV for general image tasks)
from PIL import Image

# Kivy imports
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.clock import Clock # For scheduling UI updates from other threads
from kivy.logger import Logger as KivyLogger # Use Kivy's logger for consistency within Kivy

# --- Configuration Constants (from original Kivy app) ---
SAMPLES_PER_USER: int = 10
FRAME_REDUCE_FACTOR: float = 0.5 # Not directly used for backend processing, but good to keep in mind for frame quality
RECOGNITION_INTERVAL: int = 3 * 60 # 3 minutes
AUDIO_FILE: str = "thank_you.mp3" # This will be handled by frontend
TICK_ICON_PATH: str = "tick.png" # This will be handled by frontend

# Define a recognition threshold for face_recognition (lower is better, 0.6 is common)
RECOGNITION_THRESHOLD: float = 0.6

GOOGLE_FORM_VIEW_URL: str = (
    "https://docs.google.com/forms/u/0/d/e/1FAIpQLScO9FVgTOXCeuw210SK6qx2fXiouDqouy7TTuoI6UD80ZpYvQ/viewform"
)
GOOGLE_FORM_POST_URL: str = (
    "https://docs.google.com/forms/u/0/d/e/1FAIpQLScO9FVgTOXCeuw210SK6qx2fXiouDqouy7TTuoI6UD80ZpYvQ/formResponse"
)
FORM_FIELDS: Dict[str, str] = {
    "name": "entry.935510406",
    "emp_id": "entry.886652582",
    "date": "entry.1160275796",
    "time": "entry.32017675",
}

# Environment variables for sensitive info
EMAIL_ADDRESS: str = os.environ.get("FACEAPP_EMAIL", "faceapp0011@gmail.com")
EMAIL_PASSWORD: str = os.environ.get("FACEAPP_PASS", "ytup bjrd pupf tuuj")
SMTP_SERVER: str = "smtp.gmail.com"
SMTP_PORT: int = 587
ADMIN_EMAIL_ADDRESS: str = os.environ.get("FACEAPP_ADMIN_EMAIL", "projects@archtechautomation.com")

# Simple logger for backend console output (now uses KivyLogger)
def Logger(message: str) -> None:
    KivyLogger.info(f"FlaskBackend: {message}")

def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)

def python_time_now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _crop_and_resize_for_passport(pil_image: Image.Image, target_size: Tuple[int, int] = (240, 320)) -> Image.Image:
    """
    Crops and resizes a PIL Image to a target aspect ratio and size,
    similar to passport photo requirements.
    """
    w, h = pil_image.size
    target_width, target_height = target_size
    target_aspect_ratio = target_width / target_height
    current_aspect_ratio = w / h

    cropped_image = pil_image
    if current_aspect_ratio > target_aspect_ratio:
        new_width = int(h * target_aspect_ratio)
        x_start = (w - new_width) // 2
        cropped_image = pil_image.crop((x_start, 0, x_start + new_width, h))
    elif current_aspect_ratio < target_aspect_ratio:
        new_height = int(w / target_aspect_ratio)
        y_start = (h - new_height) // 2
        cropped_image = pil_image.crop((0, y_start, w, y_start + new_height))

    # Use Image.LANCZOS for high-quality downsampling
    resized_image = cropped_image.resize(target_size, Image.LANCZOS)
    return resized_image

class FaceAppBackend:
    def __init__(self):
        self.known_faces_dir: str = str(Path("./known_faces")) # Store faces in a local directory
        ensure_dir(self.known_faces_dir)
        Logger(f"[INFO] Known faces directory set to: {self.known_faces_dir}")

        # Store known face encodings and their corresponding IDs
        self.known_face_encodings: List[np.ndarray] = []
        self.known_face_ids: List[Tuple[str, str]] = [] # Stores (name, emp_id) for each encoding

        self.last_seen_time: Dict[str, float] = {}
        self.otp_storage: Dict[str, str] = {}
        self.pending_names: Dict[str, Optional[str]] = {} # Stores name for capture process
        self.user_emails: Dict[str, str] = {}
        self.daily_attendance_status: Dict[str, str] = {} # Stores emp_id -> date (YYYY-MM-DD) for in/out tracking
        self.last_recognized_info: Dict[str, Any] = {} # Initialize as empty dict

        self.capture_mode: bool = False # Flag to indicate if samples are being captured
        self.capture_target_count: int = 0
        self.capture_collected_count: int = 0
        self.capture_name: Optional[str] = None
        self.capture_emp_id: Optional[str] = None
        self.capture_start_index: int = 0
        self.capture_lock = threading.Lock() # To prevent race conditions during capture

        self._load_known_faces_and_emails() # Initial loading and encoding
        self.daily_attendance_status = self._load_daily_attendance_status() # Load attendance status

    def _load_known_faces_and_emails(self):
        """
        Loads images from known_faces_dir, computes face encodings,
        and loads user emails.
        """
        self.known_face_encodings = []
        self.known_face_ids = []
        
        ensure_dir(self.known_faces_dir)
        for file in sorted(os.listdir(self.known_faces_dir)):
            if not file.lower().endswith((".jpg", ".png")):
                continue
            try:
                # Filename format: name_emp_id_XXX.jpg
                parts = file.split("_")
                if len(parts) < 3:
                    Logger(f"[WARN] Skipping unrecognised filename format: {file}")
                    continue
                # Reconstruct name if it had underscores, then convert to lowercase
                name = "_".join(parts[:-2]).lower()
                emp_id = parts[-2].upper()
            except ValueError:
                Logger(f"[WARN] Skipping unrecognised filename format: {file}")
                continue

            img_path = Path(self.known_faces_dir) / file
            try:
                # face_recognition.load_image_file uses PIL internally, so it's fine
                img = face_recognition.load_image_file(str(img_path))
                # Find all face locations and encodings in the image
                face_locations = face_recognition.face_locations(img)
                face_encodings = face_recognition.face_encodings(img, face_locations)

                if face_encodings:
                    # Assuming one primary face per training image
                    self.known_face_encodings.append(face_encodings[0])
                    self.known_face_ids.append((name, emp_id))
                    Logger(f"[INFO] Loaded encoding for {name.title()} ({emp_id}) from {file}")
                else:
                    Logger(f"[WARN] No face found in training image: {file}")
            except Exception as e:
                Logger(f"[ERROR] Could not load or process image {img_path}: {e}")
                continue
        
        Logger(f"[INFO] Loaded {len(self.known_face_encodings)} known face encodings for {len(set(self.known_face_ids))} unique identities.")
        self.user_emails = self._load_emails()

    def _load_emails(self) -> Dict[str, str]:
        """Loads user emails from a JSON file."""
        emails_file = Path(self.known_faces_dir) / "user_emails.json"
        if emails_file.is_file():
            try:
                with emails_file.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError as exc:
                Logger(f"[WARN] Invalid JSON in email storage: {exc}; starting fresh.")
            except IOError as exc:
                Logger(f"[ERROR] Could not read user_emails.json: {exc}")
        return {}

    def _save_email(self, emp_id: str, email: str) -> None:
        """Saves a user's email to the JSON file."""
        self.user_emails[emp_id] = email
        try:
            with (Path(self.known_faces_dir) / "user_emails.json").open("w", encoding="utf-8") as f:
                json.dump(self.user_emails, f, indent=2)
        except IOError as exc:
            Logger(f"[ERROR] Could not save user_emails.json: {exc}")

    def _load_daily_attendance_status(self) -> Dict[str, str]:
        """Loads daily attendance status from a JSON file."""
        attendance_file = Path(self.known_faces_dir) / "daily_attendance.json"
        if attendance_file.is_file():
            try:
                with attendance_file.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError as exc:
                Logger(f"[WARN] Invalid JSON in daily attendance status: {exc}; starting fresh.")
            except IOError as exc:
                Logger(f"[ERROR] Could not read daily_attendance.json: {exc}")
        return {}

    def _save_daily_attendance_status(self) -> None:
        """Saves daily attendance status to a JSON file."""
        try:
            with (Path(self.known_faces_dir) / "daily_attendance.json").open("w", encoding="utf-8") as f:
                json.dump(self.daily_attendance_status, f, indent=2)
        except IOError as exc:
            Logger(f"[ERROR] Could not save daily_attendance.json: {exc}")

    def _generate_otp(self) -> str:
        """Generates a 6-digit OTP."""
        return str(random.randint(100000, 999999))

    def _send_email(self, recipient_email: str, subject: str, body_html: str, image_data: Optional[bytes] = None, image_cid: Optional[str] = None) -> bool:
        """Sends a generic email with optional image attachment."""
        msg = MIMEMultipart("related") # Use 'related' to embed images
        msg["From"] = EMAIL_ADDRESS
        msg["To"] = recipient_email
        msg["Subject"] = subject

        # Attach HTML body
        msg.attach(MIMEText(body_html, "html"))

        # Attach image if provided
        if image_data and image_cid:
            image = MIMEImage(image_data, "jpeg") # Assuming JPEG format for face images
            image.add_header("Content-ID", f"<{image_cid}>")
            image.add_header("Content-Disposition", "inline", filename="face_detection.jpg")
            msg.attach(image)

        try:
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                server.starttls()
                server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
                server.send_message(msg)
            Logger(f"[INFO] Email sent to {recipient_email} with subject: '{subject}'")
            return True
        except Exception as exc:
            Logger(f"[ERROR] SMTP error when sending email to {recipient_email}: {exc}")
            return False

    def _send_otp_email(self, email: str, otp: str, name: str, emp_id: str, is_admin_email: bool = False) -> bool:
        """Sends an OTP or admin notification email (uses generic _send_email)."""
        if is_admin_email:
            subject = f"FaceApp Notification: Person Details - {name.title()} ({emp_id})"
            body_html = (
                f"<p>Details of a person for whom an OTP process was initiated:</p>"
                f"<p><b>Name:</b> {name.title()}<br>"
                f"<b>Employee ID:</b> {emp_id}</p>"
                f"<p>Generated OTP: <b>{otp}</b></p>"
            )
        else:
            subject = "Your FaceApp OTP"
            body_html = (
                f"<h2>OTP Verification for {name.title()} ({emp_id})</h2><p>Your OTP is <b>{otp}</b>. "
                "It is valid for 10 minutes.</p>"
                "<p>Please use this OTP to proceed with your photo update/registration.</p>"
            )
        return self._send_email(email, subject, body_html)

    def _send_attendance_email(self, email: str, name: str, emp_id: str, detection_time: str, email_type: str, face_image_b64: Optional[str] = None) -> bool:
        """
        Sends an attendance email (in-time or out-time) with an optional embedded face image.
        email_type can be "in" or "out".
        """
        current_date_display = datetime.now().strftime("%A, %B %d, %Y")
        image_cid = "detected_face_image" # Content-ID for the embedded image

        image_html = ""
        image_data = None
        if face_image_b64:
            try:
                image_data = base64.b64decode(face_image_b64)
                image_html = f'<p><img src="cid:{image_cid}" alt="Detected Face" style="width:240px;height:320px;border-radius:8px;"></p>'
            except Exception as e:
                Logger(f"[ERROR] Failed to decode base64 image for email: {e}")
                image_html = "" # Clear image HTML on error
                image_data = None


        if email_type == "in":
            subject = f"FaceApp Attendance: In-Time Recorded for {name.title()} ({emp_id})"
            body_html = (
                f"<h2>Attendance Recorded!</h2>"
                f"<p>Dear {name.title()},</p>"
                f"<p>Your attendance has been successfully recorded.</p>"
                f"<p><b>Date:</b> {current_date_display}<br>"
                f"<b>In-Time:</b> {detection_time}</p>"
                f"{image_html}" # Include image HTML here
                f"<p>Thank you!</p>"
            )
        elif email_type == "out":
            subject = f"FaceApp Attendance: Out-Time Recorded for {name.title()} ({emp_id})"
            body_html = (
                f"<h2>Out-Time Recorded!</h2>"
                f"<p>Dear {name.title()},</p>"
                f"<p>Your out-time has been successfully recorded.</p>"
                f"<p><b>Date:</b> {current_date_display}<br>"
                f"<b>Out-Time:</b> {detection_time}</p>"
                f"{image_html}" # Include image HTML here
                f"<p>Have a great day!</p>"
            )
        else:
            Logger(f"[ERROR] Invalid email_type '{email_type}' for attendance email.")
            return False

        return self._send_email(email, subject, body_html, image_data, image_cid)


    def _submit_to_google_form(self, name: str, emp_id: str) -> None:
        """Submits attendance data to a Google Form."""
        payload = {
            FORM_FIELDS["name"]: name.title(),
            FORM_FIELDS["emp_id"]: emp_id,
            FORM_FIELDS["date"]: datetime.now().strftime("%d/%m/%Y"),
            FORM_FIELDS["time"]: datetime.now().strftime("%H:%M:%S"),
        }
        headers = {
            "User-Agent": "Mozilla/5.0 (FaceApp Attendance Bot)",
            "Referer": GOOGLE_FORM_VIEW_URL,
        }
        Logger(f"[INFO] Attempting to submit attendance for {name} ({emp_id}) to URL: {GOOGLE_FORM_POST_URL}")
        Logger(f"[INFO] Payload: {payload}")
        try:
            with requests.Session() as session:
                resp = session.post(GOOGLE_FORM_POST_URL, data=payload, headers=headers, timeout=10, allow_redirects=False)
            if resp.status_code in (200, 302):
                Logger("[INFO] Attendance submitted successfully to Google Form.")
                # Frontend will display success message
            else:
                Logger(f"[WARN] Google Form submission returned status {resp.status_code}. Response: {resp.text[:200]}...")
                # Frontend will display warning
        except requests.exceptions.Timeout:
            Logger(f"[ERROR] Google Form submission timed out for {name} ({emp_id}).")
            # Frontend will display error
        except requests.exceptions.ConnectionError as exc:
            Logger(f"[ERROR] Google Form submission connection error for {name} ({emp_id}): {exc}")
            # Frontend will display error
        except requests.RequestException as exc:
            Logger(f"[ERROR] An unexpected error occurred during form submission for {name} ({emp_id}): {exc}")
            # Frontend will display error

    def process_frame(self, frame_data_b64: str) -> Dict[str, Any]:
        """
        Processes a single frame for face detection and recognition using face_recognition.
        If in capture mode, it saves face samples.
        Returns detection/recognition results.
        """
        try:
            # Decode base64 image using PIL
            image_bytes = base64.b64decode(frame_data_b64)
            # Use BytesIO to treat bytes as a file for Image.open
            pil_image = Image.open(io.BytesIO(image_bytes))

            if pil_image is None:
                Logger("[ERROR] Could not decode image data using PIL.")
                return {"status": "error", "message": "Invalid image data"}

            # Ensure image is in RGB format for face_recognition
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Convert PIL image to a NumPy array for face_recognition
            # face_recognition expects a NumPy array (height, width, channels)
            frame_np = np.array(pil_image)

            # Find all the faces and face encodings in the current frame
            face_locations = face_recognition.face_locations(frame_np)
            face_encodings = face_recognition.face_encodings(frame_np, face_locations)

            results = []
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # The 'box' coordinates are (top, right, bottom, left) from face_recognition.
                # Convert to (x, y, w, h) for consistency with previous output format.
                x_full, y_full, w_full, h_full = left, top, right - left, bottom - top

                name = "unknown"
                emp_id = ""
                conf = 1.0 # High distance for unknown (1.0 means no match)

                if self.known_face_encodings: # Only attempt recognition if known faces exist
                    # Compare current face with known faces
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)

                    # Check if the best match is within the recognition threshold
                    if face_distances[best_match_index] < RECOGNITION_THRESHOLD:
                        name, emp_id = self.known_face_ids[best_match_index]
                        conf = face_distances[best_match_index] # Store distance as confidence (lower is better)
                        Logger(f"[INFO] Recognized: {name.title()} ({emp_id}) with distance: {conf:.2f}")
                    else:
                        Logger(f"[INFO] Face detected, but not recognized (min distance: {face_distances[best_match_index]:.2f}).")
                else:
                    Logger("[INFO] No known faces for recognition.")


                face_info = {
                    "box": [x_full, y_full, w_full, h_full],
                    "name": name.title(),
                    "emp_id": emp_id,
                    "confidence": float(conf), # Using distance as 'confidence' (lower is better)
                    "status": "unknown"
                }

                if self.capture_mode:
                    with self.capture_lock:
                        if self.capture_collected_count < self.capture_target_count:
                            # Extract face ROI from original PIL image for saving
                            # PIL crop uses (left, upper, right, lower)
                            face_roi_pil = pil_image.crop((left, top, right, bottom))
                            
                            # Resize for consistency in saved samples
                            # Use Image.LANCZOS for high-quality downsampling
                            face_img_resized_pil = face_roi_pil.resize((200, 200), Image.LANCZOS)
                            filename = f"{self.capture_name}_{self.capture_emp_id}_{self.capture_start_index + self.capture_collected_count:03d}.jpg"
                            face_img_resized_pil.save(str(Path(self.known_faces_dir) / filename))
                            self.capture_collected_count += 1
                            Logger(f"[INFO] Captured sample {self.capture_collected_count}/{self.capture_target_count} for {self.capture_emp_id}")
                            face_info["capture_progress"] = f"{self.capture_collected_count}/{self.capture_target_count}"
                            face_info["status"] = "capturing"
                            if self.capture_collected_count >= self.capture_target_count:
                                Logger("[INFO] Capture complete – reloading known faces…")
                                self.capture_mode = False # End capture mode
                                # Reload known faces in a separate thread to avoid blocking
                                threading.Thread(target=self._reload_known_faces_after_capture, daemon=True).start()
                                face_info["status"] = "capture_complete"
                        else:
                            face_info["status"] = "capturing" # Still in capture mode until reloading is done
                else: # Normal recognition mode
                    # Use RECOGNITION_THRESHOLD for face_recognition distance
                    if conf < RECOGNITION_THRESHOLD: # Lower distance means better match
                        now = time.time()
                        last_seen = self.last_seen_time.get(emp_id, 0)
                        if now - last_seen > RECOGNITION_INTERVAL:
                            self.last_seen_time[emp_id] = now
                            face_info["status"] = "recognized_new"
                            # Extract the detected face ROI for the email/preview
                            face_roi_pil_for_email = pil_image.crop((left, top, right, bottom))
                            # Trigger attendance submission and preview update in separate threads
                            threading.Thread(
                                target=self._handle_successful_recognition,
                                args=(name, emp_id, face_roi_pil_for_email), # Pass PIL image
                                daemon=True,
                                name="AttendanceSubmitter",
                            ).start()
                        else:
                            face_info["status"] = "recognized_recent"
                    else:
                        face_info["status"] = "unknown"

                results.append(face_info)

            return {"status": "success", "faces": results}

        except Exception as e:
            Logger(f"[ERROR] Error processing frame: {e}")
            return {"status": "error", "message": str(e)}

    def _reload_known_faces_after_capture(self):
        """Reloads known faces and their encodings after new samples are captured."""
        self._load_known_faces_and_emails()
        Logger("[INFO] Known faces reloading finished.")


    def _handle_successful_recognition(self, name: str, emp_id: str, face_roi_pil: Image.Image):
        """Handles post-recognition actions like attendance submission and email sending."""
        Logger(f"[INFO] Recognised {name} ({emp_id}) – submitting attendance and checking email status…")
        
        current_date_str = datetime.now().strftime("%Y-%m-%d")
        current_time_str = datetime.now().strftime("%H:%M:%S")
        
        user_email = self.user_emails.get(emp_id)

        # Process face image for display on frontend and for email
        processed_face_image_pil = _crop_and_resize_for_passport(face_roi_pil, (240, 320))
        
        # Convert PIL image to base64
        buffered = io.BytesIO()
        processed_face_image_pil.save(buffered, format="JPEG") # Save as JPEG
        face_image_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        if user_email:
            # Check if this is the first recognition for today for this user
            if self.daily_attendance_status.get(emp_id) != current_date_str:
                Logger(f"[INFO] First recognition for {name} ({emp_id}) today. Sending in-time email.")
                self.daily_attendance_status[emp_id] = current_date_str
                self._save_daily_attendance_status()
                # Send "in-time" email with the face image
                threading.Thread(
                    target=self._send_attendance_email,
                    args=(user_email, name, emp_id, current_time_str, "in", face_image_b64),
                    daemon=True,
                    name="InTimeEmailSender",
                ).start()
            else:
                Logger(f"[INFO] {name} ({emp_id}) already recognized today. Sending out-time email.")
                # Send "out-time" email with the face image
                threading.Thread(
                    target=self._send_attendance_email,
                    args=(user_email, name, emp_id, current_time_str, "out", face_image_b64),
                    daemon=True,
                    name="OutTimeEmailSender",
                ).start()
        else:
            Logger(f"[WARN] No email found for {name} ({emp_id}). Skipping attendance email.")

        # Frontend will receive this info via a separate mechanism or poll
        # For now, just submit to Google Form
        threading.Thread(target=self._submit_to_google_form, args=(name, emp_id), daemon=True, name="GoogleFormSubmitter").start()

        # Store last recognized info, frontend can poll this or receive via WebSocket if implemented
        self.last_recognized_info = {
            "name": name.title(),
            "emp_id": emp_id,
            "time": current_time_str,
            "image": face_image_b64
        }

    def start_capture_samples(self, name: str, emp_id: str, updating: bool = False, sample_count: Optional[int] = None) -> Dict[str, Any]:
        """
        Initiates the sample capture process.
        Sets internal flags for `process_frame` to start saving images.
        """
        with self.capture_lock:
            if self.capture_mode:
                return {"status": "error", "message": "Already in capture mode."}

            # If updating, ensure the name is retrieved from existing data if not provided
            resolved_name = name
            if updating and not resolved_name:
                # Find the name from known_face_ids based on emp_id
                resolved_name = next((nm for nm, eid in self.known_face_ids if eid
