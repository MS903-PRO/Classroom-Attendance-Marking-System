import os
import json
import csv
import time
import shutil
import sqlite3
import subprocess
import threading
from datetime import date

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
from tkinterdnd2 import TkinterDnD, DND_FILES

import cv2
import numpy as np
from PIL import Image, ImageTk
from deepface import DeepFace

# Configuration
DB_PATH = "facialdb.db"
CLASSROOM_FILE = "classrooms.json"
ATTENDANCE_FILE = "attendance.json"
OUTPUT_DATA_FILE = "output_data.json"
WEB_CAM_URL_FILE = "webcam_url.json"

CAPTURED_IMAGES_FOLDER = "captured_images"
RECOGNIZED_FACES_FOLDER = "recognized_faces"
UNRECOGNIZED_FACES_FOLDER = "unrecognized_faces"

all_faces = []
pending_attendance = {}
session_started = False


def check_database():
    con = sqlite3.connect(DB_PATH)
    cursor = con.cursor()
    try:
        cursor.execute("SELECT ID, IMG_NAME, VALUE FROM EMBEDDING LIMIT 1")
    except sqlite3.OperationalError:
        messagebox.showerror("Error", "Database schema is incorrect!")
    finally:
        con.close()


def is_student_registered(roll_number):
    con = sqlite3.connect(DB_PATH)
    cursor = con.cursor()
    cursor.execute("SELECT COUNT(*) FROM IDENTITIES WHERE IMG_NAME LIKE ?", (f"%{roll_number}%",))
    exists = cursor.fetchone()[0]
    con.close()
    return "Registered" if exists > 0 else "Not Registered"


def match_student_in_db(embedding):
    con = sqlite3.connect(DB_PATH)
    cursor = con.cursor()
    cursor.execute("SELECT ID, IMG_NAME, EMBEDDING FROM IDENTITIES")
    all_students = cursor.fetchall()

    best_match = None
    lowest_distance = float("inf")
    embedding = np.array(embedding, dtype=np.float32)

    for student_id, img_name, stored_embedding in all_students:
        stored_embedding = np.frombuffer(stored_embedding, dtype=np.float32)
        if stored_embedding.shape != embedding.shape:
            continue
        distance = np.linalg.norm(embedding - stored_embedding)
        if distance < 11 and distance < lowest_distance:
            lowest_distance = distance
            best_match = img_name.split("_")[0]

    con.close()
    return best_match


def load_classrooms():
    # Clean up recognized faces folder
    if os.path.exists(RECOGNIZED_FACES_FOLDER):
        for file_name in os.listdir(RECOGNIZED_FACES_FOLDER):
            file_path = os.path.join(RECOGNIZED_FACES_FOLDER, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
    
    if os.path.exists(CLASSROOM_FILE):
        with open(CLASSROOM_FILE, "r") as file:
            classrooms = json.load(file)
            if isinstance(classrooms, list):
                return classrooms
            else:
                messagebox.showerror("File Error", "Invalid data format in classrooms.json")
                return []
    return []


def save_classrooms(classrooms):
    with open(CLASSROOM_FILE, "w") as file:
        json.dump(classrooms, file, indent=4)


def load_attendance():
    if os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "r") as file:
            attendance = json.load(file)
            if isinstance(attendance, dict):
                return attendance
            else:
                messagebox.showerror("File Error", "Invalid data format in attendance.json")
                return {}
    return {}


def save_attendance_with_date(attendance_data):
    today = str(date.today())
    
    if os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "r") as file:
            try:
                all_attendance = json.load(file)
            except json.JSONDecodeError:
                all_attendance = {}
    else:
        all_attendance = {}
    
    if not isinstance(attendance_data, dict):
        messagebox.showerror("Error", "Invalid attendance format!")
        return
    
    all_attendance[today] = attendance_data
    
    with open(ATTENDANCE_FILE, "w") as file:
        json.dump(all_attendance, file, indent=4)
    
    print(f"Attendance saved for {today}")
    messagebox.showinfo("Success", f"Attendance saved for {today}")


def load_webcam_url():
    if os.path.exists(WEB_CAM_URL_FILE):
        with open(WEB_CAM_URL_FILE, "r") as file:
            try:
                url_data = json.load(file)
                return url_data.get("webcam_url", "")
            except json.JSONDecodeError:
                return ""
    return ""


def save_webcam_url(url):
    if url:
        url_data = {"webcam_url": url}
        with open(WEB_CAM_URL_FILE, "w") as file:
            json.dump(url_data, file, indent=4)
        print(f"Saved Webcam URL: {url}")


def enhance_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    enhanced_img = cv2.merge([enhanced_gray] * 3)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced_img, -1, kernel)
    return sharpened


def start_session_and_clear_images():
    global session_started
    
    if not os.path.exists(CAPTURED_IMAGES_FOLDER):
        os.makedirs(CAPTURED_IMAGES_FOLDER)
    else:
        for file_name in os.listdir(CAPTURED_IMAGES_FOLDER):
            file_path = os.path.join(CAPTURED_IMAGES_FOLDER, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
    
    session_started = True
    print("Session started, old images cleared.")


def clear_old_images():
    if os.path.exists(CAPTURED_IMAGES_FOLDER):
        for file_name in os.listdir(CAPTURED_IMAGES_FOLDER):
            file_path = os.path.join(CAPTURED_IMAGES_FOLDER, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        os.makedirs(CAPTURED_IMAGES_FOLDER)


def run_face_recognition():
    try:
        print("Running search.py...")
        result = subprocess.run(["python", "search.py"], capture_output=True, text=True)
        print("Search.py Output:", result.stdout)
        print("Search.py Errors:", result.stderr)
        
        if not os.path.exists("faces.npy") or not os.path.exists(OUTPUT_DATA_FILE):
            print("Error: faces.npy or output_data.json NOT FOUND!")
            messagebox.showerror("Error", "No saved recognition results found!")
            return [], []
        
        all_faces = np.load("faces.npy", allow_pickle=True)
        with open(OUTPUT_DATA_FILE, "r") as f:
            output_data = json.load(f)
        
        return all_faces, output_data
    except Exception as e:
        messagebox.showerror("Error", f"Error running face recognition:\n{str(e)}")
        return [], []


def get_output_data():
    global all_faces
    
    if not all_faces:
        print("Warning: No faces found. Ensure face extraction runs first!")
        return []
    
    output_data = []
    
    if not os.path.exists(UNRECOGNIZED_FACES_FOLDER):
        os.makedirs(UNRECOGNIZED_FACES_FOLDER)
    
    for face_idx, face in enumerate(all_faces):
        face_image = np.array(face, dtype=np.uint8)
        
        if len(face_image.shape) == 3 and face_image.shape[2] == 3:
            face_image = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
        elif len(face_image.shape) == 2:
            face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2BGR)
        
        print(f"Processing face {face_idx} | Shape: {face_image.shape}")
        
        temp_debug_path = os.path.join(UNRECOGNIZED_FACES_FOLDER, f"debug_{face_idx}.jpg")
        cv2.imwrite(temp_debug_path, face_image)
        
        try:
            objs = DeepFace.represent(img_path=temp_debug_path, model_name="Facenet", enforce_detection=False)
        except Exception as e:
            print(f"DeepFace Error: {e}")
            objs = []
        
        if not objs:
            output_data.append([face_idx, "NULL", 0.0])
            unrecognized_path = os.path.join(UNRECOGNIZED_FACES_FOLDER, f"unknown_{face_idx}.jpg")
            cv2.imwrite(unrecognized_path, face_image)
            continue
        
        embedding = objs[0]["embedding"]
        matched_student = match_student_in_db(embedding)
        output_data.append([face_idx, matched_student if matched_student else "NULL", 1.0])
    
    return output_data


def load_registered_students():
    if os.path.exists(CLASSROOM_FILE):
        with open(CLASSROOM_FILE, "r") as file:
            classrooms = json.load(file)
            all_students = []
            for classroom in classrooms:
                all_students.extend(classroom.get("students", []))
            return all_students
    return []


def load_recognized_faces():
    recognized_faces = {}
    if os.path.exists(OUTPUT_DATA_FILE):
        with open(OUTPUT_DATA_FILE, "r") as f:
            output_data = json.load(f)
        for entry in output_data:
            if len(entry) >= 2:
                roll_number = str(entry[1])
                if roll_number != "NULL":
                    image_path = os.path.join(RECOGNIZED_FACES_FOLDER, f"{entry[0]}.jpg")
                    if os.path.exists(image_path):
                        recognized_faces[roll_number] = image_path
    return recognized_faces


def use_webcam(webcam_url):
    start_session_and_clear_images()
    
    if not webcam_url:
        messagebox.showerror("Error", "Webcam URL is not set. Please set the webcam URL first.")
        return
    
    cap = cv2.VideoCapture(webcam_url)
    
    if not cap.isOpened():
        messagebox.showerror("Error", "Unable to access webcam. Please check the webcam URL or connection.")
        return
    
    print("Webcam connected successfully.")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    if face_cascade.empty():
        messagebox.showerror("Error", "Haar cascade classifier failed to load.")
        cap.release()
        cv2.destroyAllWindows()
        return
    
    face_start_time = None
    capture_interval = 4
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) > 0:
            if face_start_time is None:
                face_start_time = time.time()
            
            time_since_face = time.time() - face_start_time
            time_remaining = max(0, int(capture_interval - time_since_face))
            timer_text = f"Next capture in: {time_remaining} sec"
            
            if time_since_face >= capture_interval:
                filename = f"captured_{int(time.time())}.jpg"
                full_path = os.path.join(CAPTURED_IMAGES_FOLDER, filename)
                cv2.imwrite(full_path, frame)
                print(f"Photo captured and saved as {full_path}")
                face_start_time = None
        else:
            face_start_time = None
            timer_text = "No face detected"
        
        frame_resized = cv2.resize(frame, (1200, 720))
        cv2.putText(frame_resized, timer_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame_resized, timer_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("Webcam Feed (PRESS 'q' TO EXIT)", frame_resized)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


def input_webcam_url():
    def save_url():
        url = url_entry.get().strip()
        if url:
            save_webcam_url(url)
            messagebox.showinfo("Success", "Webcam URL saved successfully!")
            url_window.destroy()
        else:
            messagebox.showwarning("Input Error", "Please enter a valid URL.")
    
    url_window = tk.Toplevel(root)
    url_window.title("Enter Webcam URL")
    url_window.geometry("400x200")
    
    tk.Label(url_window, text="Enter Webcam URL:", font=("Arial", 12)).pack(pady=10)
    saved_url = load_webcam_url()
    url_entry = tk.Entry(url_window, font=("Arial", 12))
    url_entry.insert(0, saved_url)
    url_entry.pack(pady=10)
    tk.Button(url_window, text="Save URL", font=("Arial", 12), command=save_url).pack(pady=10)


def generate_attendance_and_display():
    all_faces, output_data = run_face_recognition()
    
    if output_data:
        attendance_data = {}
        for entry in output_data:
            if len(entry) >= 2:
                roll_number = str(entry[1])
                if roll_number != "NULL":
                    attendance_data[roll_number] = "Present"
        
        global pending_attendance
        pending_attendance = attendance_data
        show_attendance_editor()


def show_processing_window():
    processing_window = tk.Toplevel()
    processing_window.title("Processing...")
    processing_window.geometry("300x100")
    tk.Label(processing_window, text="Processing images...", font=("Arial", 14)).pack(pady=20)
    processing_window.update()
    return processing_window


def show_attendance_editor():
    registered_students = load_registered_students()
    recognized_faces = load_recognized_faces()
    
    attendance_status = {}
    for student in registered_students:
        roll_number = student["roll_number"]
        registered_status = is_student_registered(roll_number)
        status = "Invalid" if registered_status == "Not Registered" else ("Present" if roll_number in recognized_faces else "Absent")
        attendance_status[roll_number] = status
    
    attendance_window = tk.Toplevel()
    attendance_window.title("Attendance Editor")
    attendance_window.geometry("700x600")
    
    tk.Label(attendance_window, text="Attendance", font=("Arial", 16, "bold")).pack(pady=10)
    
    table_frame = tk.Frame(attendance_window)
    table_frame.pack(pady=10, fill=tk.BOTH, expand=True)
    
    columns = ("Roll Number", "Name", "Status")
    tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=15)
    
    for col in columns:
        tree.heading(col, text=col)
        tree.column(col, anchor="center", width=200)
    tree.pack(fill=tk.BOTH, expand=True)
    
    sorted_students = sorted(registered_students, key=lambda student: (
        0 if attendance_status[student["roll_number"]] == "Absent" else
        1 if attendance_status[student["roll_number"]] == "Present" else 2,
        student["roll_number"]))
    
    for student in sorted_students:
        roll_number, name = student["roll_number"], student["name"]
        status = attendance_status.get(roll_number, "Invalid")
        item = tree.insert("", "end", values=(roll_number, name, status))
        tree.item(item, tags=("present" if status == "Present" else "absent" if status == "Absent" else "invalid",))
    
    tree.tag_configure("present", foreground="green", font=("Arial", 12, "bold"))
    tree.tag_configure("absent", foreground="red", font=("Arial", 12, "bold"))
    tree.tag_configure("invalid", foreground="gray", font=("Arial", 12, "bold"))
    
    def toggle_attendance():
        selected_item = tree.selection()
        if not selected_item:
            return
        
        for item in selected_item:
            values = tree.item(item, "values")
            roll_number, current_status = values[0], values[2]
            if current_status == "Invalid":
                continue
            new_status = "Present" if current_status == "Absent" else "Absent"
            attendance_status[roll_number] = new_status
            tree.item(item, values=(values[0], values[1], new_status))
            tree.item(item, tags=("present" if new_status == "Present" else "absent",))
    
    def remove_image_from_editor(roll_number, frame):
        if roll_number in attendance_status:
            attendance_status[roll_number] = "Absent"
            frame.destroy()
            for item in tree.get_children():
                values = tree.item(item, "values")
                if values[0] == roll_number:
                    tree.item(item, values=(values[0], values[1], "Absent"))
                    tree.item(item, tags="absent")
    
    def export_attendance():
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")], title="Save Attendance As")
        if not file_path:
            return
        with open(file_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Roll Number", "Name", "Status"])
            for item in tree.get_children():
                writer.writerow(tree.item(item, "values"))
        messagebox.showinfo("Success", f"Attendance exported successfully to {file_path}!")
    
    def view_images():
        img_window = tk.Toplevel()
        img_window.title("View Images")
        img_window.geometry("800x600")
        tk.Label(img_window, text="Recognized Faces", font=("Arial", 16, "bold")).pack(pady=10)
        img_frame = tk.Frame(img_window)
        img_frame.pack(fill=tk.BOTH, expand=True)
        
        recognized_faces = load_recognized_faces()
        row, col = 0, 0
        max_cols = 15
        
        for roll_number, img_path in recognized_faces.items():
            if os.path.exists(img_path):
                img = Image.open(img_path).resize((100, 100))
                img = ImageTk.PhotoImage(img)
                frame = tk.Frame(img_frame)
                frame.grid(row=row, column=col, padx=10, pady=10)
                lbl = tk.Label(frame, image=img, text=roll_number, compound="top", font=("Arial", 12))
                lbl.image = img
                lbl.pack()
                remove_btn = tk.Button(frame, text="Remove", command=lambda rn=roll_number, f=frame: remove_image_from_editor(rn, f))
                remove_btn.pack()
                col += 1
                if col >= max_cols:
                    col = 0
                    row += 1
    
    button_frame = tk.Frame(attendance_window)
    button_frame.pack(pady=10)
    tk.Button(button_frame, text="Toggle Attendance", command=toggle_attendance).pack(side=tk.LEFT, padx=10)
    tk.Button(button_frame, text="View Images", command=view_images).pack(side=tk.LEFT, padx=10)
    tk.Button(button_frame, text="Export", command=export_attendance).pack(side=tk.LEFT, padx=10)
    tk.Button(button_frame, text="Close", command=attendance_window.destroy).pack(side=tk.LEFT, padx=10)


def view_attendance():
    if not os.path.exists(ATTENDANCE_FILE):
        messagebox.showerror("Error", "No attendance records found!")
        return
    
    with open(ATTENDANCE_FILE, "r") as file:
        attendance_data = json.load(file)
    
    if not attendance_data:
        messagebox.showerror("Error", "Attendance file is empty!")
        return
    
    attendance_window = tk.Toplevel()
    attendance_window.title("View Attendance")
    attendance_window.geometry("800x600")
    tk.Label(attendance_window, text="View Attendance", font=("Arial", 16, "bold")).pack(pady=10)
    
    date_frame = tk.Frame(attendance_window)
    date_frame.pack(pady=5)
    tk.Label(date_frame, text="Select Date:", font=("Arial", 12)).pack(side=tk.LEFT, padx=5)
    
    selected_date = tk.StringVar()
    date_dropdown = ttk.Combobox(date_frame, textvariable=selected_date, state="readonly", font=("Arial", 12))
    date_dropdown["values"] = list(attendance_data.keys())
    date_dropdown.pack(side=tk.LEFT, padx=5)
    
    table_frame = tk.Frame(attendance_window)
    table_frame.pack(pady=10, fill=tk.BOTH, expand=True)
    
    columns = ("Roll Number", "Status")
    tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=15)
    for col in columns:
        tree.heading(col, text=col)
        tree.column(col, anchor="center", width=200)
    tree.pack(fill=tk.BOTH, expand=True)
    
    def load_attendance_for_date(event):
        tree.delete(*tree.get_children())
        selected_day = selected_date.get()
        if selected_day not in attendance_data:
            messagebox.showerror("Error", "No attendance found for this date!")
            return
        daily_attendance = attendance_data[selected_day]
        if isinstance(daily_attendance, str):
            try:
                daily_attendance = json.loads(daily_attendance)
            except json.JSONDecodeError:
                messagebox.showerror("Error", "Corrupted attendance data format!")
                return
        for roll_number, status in daily_attendance.items():
            tree.insert("", "end", values=(roll_number, status))
    
    date_dropdown.bind("<<ComboboxSelected>>", load_attendance_for_date)


def add_classroom():
    class_name = class_name_entry.get().strip()
    course_name = course_name_entry.get().strip()
    section = section_entry.get().strip()
    
    if class_name and course_name and section:
        classroom = {"class_name": class_name, "course_name": course_name, "section": section, "students": []}
        classrooms.append(classroom)
        save_classrooms(classrooms)
        update_classroom_list()
        add_window.destroy()
    else:
        messagebox.showwarning("Input Error", "Please fill all fields.")


def delete_classroom(index):
    del classrooms[index]
    save_classrooms(classrooms)
    update_classroom_list()


def edit_classroom(index):
    classroom = classrooms[index]
    
    def save_edits():
        classroom["class_name"] = edit_class_name_entry.get().strip()
        classroom["course_name"] = edit_course_name_entry.get().strip()
        classroom["section"] = edit_section_entry.get().strip()
        save_classrooms(classrooms)
        update_classroom_list()
        edit_window.destroy()
    
    edit_window = tk.Toplevel(root)
    edit_window.title("Edit Classroom")
    edit_window.geometry("400x300")
    
    tk.Label(edit_window, text="Course Name:", font=("Arial", 12)).pack(pady=5)
    edit_class_name_entry = tk.Entry(edit_window, font=("Arial", 12))
    edit_class_name_entry.insert(0, classroom["class_name"])
    edit_class_name_entry.pack(pady=5)
    
    tk.Label(edit_window, text="Course Code:", font=("Arial", 12)).pack(pady=5)
    edit_course_name_entry = tk.Entry(edit_window, font=("Arial", 12))
    edit_course_name_entry.insert(0, classroom["course_name"])
    edit_course_name_entry.pack(pady=5)
    
    tk.Label(edit_window, text="Section:", font=("Arial", 12)).pack(pady=5)
    edit_section_entry = tk.Entry(edit_window, font=("Arial", 12))
    edit_section_entry.insert(0, classroom["section"])
    edit_section_entry.pack(pady=5)
    
    tk.Button(edit_window, text="Save", font=("Arial", 12), command=save_edits).pack(pady=10)


def open_classroom_page(index):
    classroom = classrooms[index]
    classroom_window = tk.Toplevel(root)
    classroom_window.title(f"{classroom['class_name']} Options")
    classroom_window.geometry("400x300")
    
    tk.Label(classroom_window, text=f"Manage {classroom['class_name']}", font=("Arial", 14)).pack(pady=10)
    tk.Button(classroom_window, text="Capture Attendance", font=("Arial", 12), command=lambda: capture_attendance(index)).pack(pady=10)
    tk.Button(classroom_window, text="See Registered Students", font=("Arial", 12), command=lambda: see_registered_students(index)).pack(pady=10)


def capture_attendance(index):
    classroom = classrooms[index]
    capture_window = tk.Toplevel(root)
    capture_window.title(f"Capture Attendance - {classroom['class_name']}")
    capture_window.geometry("400x400")
    
    tk.Label(capture_window, text=f"Attendance for {classroom['class_name']}", font=("Arial", 12)).pack(pady=10)
    
    webcam_url = load_webcam_url()
    tk.Label(capture_window, text="Webcam URL (e.g., http://127.0.0.1:8080/video):", font=("Arial", 10)).pack(pady=5)
    webcam_url_entry = tk.Entry(capture_window, font=("Arial", 12))
    webcam_url_entry.insert(0, webcam_url)
    webcam_url_entry.pack(pady=10)
    
    tk.Button(capture_window, text="Use Webcam", font=("Arial", 12), command=lambda: use_webcam(webcam_url_entry.get())).pack(pady=10)
    tk.Button(capture_window, text="Drag and Drop Images", font=("Arial", 12), command=lambda: drag_and_drop_images(classroom)).pack(pady=10)
    tk.Button(capture_window, text="Save Webcam URL", font=("Arial", 12), command=lambda: save_webcam_url(webcam_url_entry.get())).pack(pady=10)
    tk.Button(capture_window, text="Generate Attendance", font=("Arial", 12), command=generate_attendance_and_display).pack(pady=10)


def see_registered_students(index):
    classroom = classrooms[index]
    students_window = tk.Toplevel(root)
    students_window.title(f"Students in {classroom['class_name']}")
    students_window.geometry("700x500")
    
    tk.Label(students_window, text=f"Students in {classroom['class_name']}", font=("Arial", 16, "bold")).pack(pady=10)
    
    search_frame = tk.Frame(students_window)
    search_frame.pack(pady=5)
    tk.Label(search_frame, text="Search:", font=("Arial", 14)).pack(side=tk.LEFT, padx=5)
    search_entry = tk.Entry(search_frame, font=("Arial", 14))
    search_entry.pack(side=tk.LEFT, padx=5)
    
    table_frame = tk.Frame(students_window)
    table_frame.pack(pady=10, fill=tk.BOTH, expand=True)
    
    scroll_y = ttk.Scrollbar(table_frame, orient=tk.VERTICAL)
    scroll_x = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL)
    
    columns = ("Roll Number", "Name", "Registered")
    student_table = ttk.Treeview(table_frame, columns=columns, show="headings", height=12, yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
    
    scroll_y.config(command=student_table.yview)
    scroll_x.config(command=student_table.xview)
    
    for col in columns:
        student_table.heading(col, text=col, anchor="center")
        student_table.column(col, width=200, anchor="center")
    
    scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
    scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
    student_table.pack(fill=tk.BOTH, expand=True)
    
    style = ttk.Style()
    style.configure("Treeview", font=("Arial", 14))
    style.configure("Treeview.Heading", font=("Arial", 14, "bold"))
    
    def refresh_table(search_query=""):
        student_table.delete(*student_table.get_children())
        for student in classroom["students"]:
            if isinstance(student, dict):
                roll_number = student.get("roll_number", "N/A")
                name = student.get("name", "N/A")
            else:
                roll_number, name = "N/A", student
            
            registered_status = is_student_registered(roll_number)
            
            if search_query.lower() in roll_number.lower() or search_query.lower() in name.lower():
                item = student_table.insert("", tk.END, values=(roll_number, name, registered_status))
                if registered_status == "Registered":
                    student_table.item(item, tags=("registered",))
                else:
                    student_table.item(item, tags=("not_registered",))
    
    student_table.tag_configure("registered", foreground="green", font=("Arial", 14, "bold"))
    student_table.tag_configure("not_registered", foreground="red", font=("Arial", 14, "bold"))
    refresh_table()
    
    def search_students():
        search_query = search_entry.get().strip()
        refresh_table(search_query)
    
    search_entry.bind("<KeyRelease>", lambda event: search_students())
    
    def import_students():
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not file_path:
            return
        try:
            with open(file_path, newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    if len(row) >= 2:
                        roll_number, name = row[0].strip(), row[1].strip()
                        if not any(s.get("roll_number") == roll_number for s in classroom["students"] if isinstance(s, dict)):
                            classroom["students"].append({"roll_number": roll_number, "name": name})
            save_classrooms(classrooms)
            refresh_table()
            messagebox.showinfo("Success", "Students imported successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to import students: {str(e)}")
    
    def add_student():
        new_roll_number = student_roll_entry.get().strip()
        new_name = student_name_entry.get().strip()
        if new_roll_number and new_name:
            if any(s.get("roll_number") == new_roll_number for s in classroom["students"] if isinstance(s, dict)):
                messagebox.showwarning("Warning", "Student with this roll number already exists.")
                return
            classroom["students"].append({"roll_number": new_roll_number, "name": new_name})
            save_classrooms(classrooms)
            refresh_table()
            student_roll_entry.delete(0, tk.END)
            student_name_entry.delete(0, tk.END)
        else:
            messagebox.showwarning("Warning", "Please enter both Roll Number and Name.")
    
    def remove_student():
        selected_item = student_table.selection()
        if selected_item:
            student_values = student_table.item(selected_item)["values"]
            roll_number = student_values[0]
            updated_students = []
            for student in classroom["students"]:
                if isinstance(student, dict):
                    if student.get("roll_number") != roll_number:
                        updated_students.append(student)
                else:
                    updated_students.append(student)
            classroom["students"] = updated_students
            save_classrooms(classrooms)
            refresh_table()
            try:
                subprocess.run(['python', 'remove_student.py', roll_number], check=True)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to run remove_student.py: {str(e)}")
        else:
            messagebox.showwarning("Warning", "Please select a student to remove.")
    
    input_frame = tk.Frame(students_window)
    input_frame.pack(pady=5)
    tk.Label(input_frame, text="Roll No:", font=("Arial", 14)).grid(row=0, column=0, padx=5)
    student_roll_entry = tk.Entry(input_frame, font=("Arial", 14))
    student_roll_entry.grid(row=0, column=1, padx=5)
    tk.Label(input_frame, text="Name:", font=("Arial", 14)).grid(row=0, column=2, padx=5)
    student_name_entry = tk.Entry(input_frame, font=("Arial", 14))
    student_name_entry.grid(row=0, column=3, padx=5)
    tk.Button(input_frame, text="Add Student", font=("Arial", 12), command=add_student).grid(row=0, column=4, padx=5)
    tk.Button(students_window, text="Remove Selected Student", font=("Arial", 12), command=remove_student).pack(pady=5)
    tk.Button(students_window, text="Import Students from CSV", font=("Arial", 12), command=import_students).pack(pady=5)
    tk.Button(students_window, text="Register Student", font=("Arial", 12), command=register_student).pack(pady=5)
    tk.Button(students_window, text="Register Multiple Students", font=("Arial", 12), command=register_multiple_students).pack(pady=5)


def update_classroom_list():
    for widget in classroom_frame.winfo_children():
        widget.destroy()
    
    for index, classroom in enumerate(classrooms):
        classroom_card = tk.Frame(classroom_frame, bg="white", bd=2, relief=tk.RAISED)
        classroom_card.pack(fill=tk.X, pady=10, padx=10)
        
        details_frame = tk.Frame(classroom_card, bg="white")
        details_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(details_frame, text=classroom["class_name"], bg="white", font=("Arial", 16, "bold")).pack(anchor=tk.W)
        tk.Label(details_frame, text=f"Course: {classroom['course_name']}", bg="white", font=("Arial", 12)).pack(anchor=tk.W)
        tk.Label(details_frame, text=f"Section: {classroom['section']}", bg="white", font=("Arial", 12)).pack(anchor=tk.W)
        
        open_button = tk.Button(classroom_card, text="Open", font=("Arial", 12), command=lambda idx=index: open_classroom_page(idx))
        open_button.pack(pady=5)
        
        menu_button = tk.Menubutton(classroom_card, text="⋮", bg="white", fg="black", font=("Arial", 12), bd=0)
        menu_button.pack(side=tk.RIGHT, padx=10)
        menu = tk.Menu(menu_button, tearoff=0)
        menu.add_command(label="Edit", command=lambda idx=index: edit_classroom(idx))
        menu.add_command(label="Delete", command=lambda idx=index: delete_classroom(idx))
        menu_button.configure(menu=menu)


def drag_and_drop_images(classroom):
    clear_old_images()
    
    def drop(event):
        file_path = event.data
        if os.path.isfile(file_path) and file_path.lower():
            if not os.path.exists(CAPTURED_IMAGES_FOLDER):
                os.makedirs(CAPTURED_IMAGES_FOLDER)
            file_name = os.path.basename(file_path)
            destination_path = os.path.join(CAPTURED_IMAGES_FOLDER, file_name)
            shutil.copy(file_path, destination_path)
            messagebox.showinfo("Success", f"Attendance image saved to {destination_path}")
        else:
            messagebox.showwarning("Invalid File", "Please drop a valid image file.")
    
    upload_window = tk.Toplevel(root)
    upload_window.title("Upload Attendance Images")
    upload_window.geometry("600x400")
    label = tk.Label(upload_window, text="Drag and Drop Image Files Here", font=("Arial", 14))
    label.pack(pady=20)
    upload_window.drop_target_register(DND_FILES)
    upload_window.dnd_bind('<<Drop>>', drop)
    upload_window.update()


def open_add_window():
    global add_window, class_name_entry, course_name_entry, section_entry
    add_window = tk.Toplevel(root)
    add_window.title("Add Classroom")
    add_window.geometry("500x400")
    
    tk.Label(add_window, text="Class Name:", font=("Arial", 12)).pack(pady=5)
    class_name_entry = tk.Entry(add_window, font=("Arial", 12))
    class_name_entry.pack(pady=5)
    
    tk.Label(add_window, text="Course Name:", font=("Arial", 12)).pack(pady=5)
    course_name_entry = tk.Entry(add_window, font=("Arial", 12))
    course_name_entry.pack(pady=5)
    
    tk.Label(add_window, text="Section:", font=("Arial", 12)).pack(pady=5)
    section_entry = tk.Entry(add_window, font=("Arial", 12))
    section_entry.pack(pady=5)
    
    tk.Button(add_window, text="Add", font=("Arial", 12), command=add_classroom).pack(pady=10)


def register_student():
    roll_number = simpledialog.askstring("Register Student", "Enter Roll Number:")
    if not roll_number:
        messagebox.showwarning("Warning", "Roll number is required!")
        return
    try:
        subprocess.run(["python", "augment_data_individual.py", roll_number], check=True)
        messagebox.showinfo("Success", f"Data augmentation completed for {roll_number}!")
        subprocess.run(["python", "add_student_individual.py", roll_number], check=True)
        messagebox.showinfo("Success", f"Student {roll_number} successfully added!")
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")


def register_multiple_students():
    try:
        subprocess.run(["python", "augment_data_bulk.py"], check=True)
        messagebox.showinfo("Success", "Bulk data augmentation completed!")
        subprocess.run(["python", "add_student_bulk.py"], check=True)
        messagebox.showinfo("Success", "Bulk student registration completed!")
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")


# Main application
root = TkinterDnD.Tk()
root.title("Classroom Manager")
root.geometry("600x500")

classrooms = load_classrooms()
attendance = load_attendance()

canvas = tk.Canvas(root, bg="#f0f0f0")
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

scrollbar = tk.Scrollbar(root, orient=tk.VERTICAL, command=canvas.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

canvas.configure(yscrollcommand=scrollbar.set)
canvas.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

classroom_frame = tk.Frame(canvas, bg="#f0f0f0")
canvas.create_window((0, 0), window=classroom_frame, anchor="nw")

update_classroom_list()

add_button = tk.Button(root, text="+", font=("Arial", 24), bg="#4CAF50", fg="white", bd=0, width=3, height=1, command=open_add_window)
add_button.place(relx=0.95, rely=0.95, anchor=tk.SE)

root.mainloop()
