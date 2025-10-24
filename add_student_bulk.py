import os
import sqlite3
import numpy as np
from deepface import DeepFace

# Connect to SQLite database
con = sqlite3.connect("facialdb.db")
cursor = con.cursor()

instances = []

# Traverse the top-level directories (each representing a student's roll number)
base_dir = "SWE_DATA_augmented"
for roll_number in os.listdir(base_dir):
    roll_path = os.path.join(base_dir, roll_number)

    # Ensure it's a directory (skip files)
    if not os.path.isdir(roll_path):
        continue

    # Check if roll number already exists in the database
    cursor.execute("SELECT COUNT(*) FROM IDENTITIES WHERE IMG_NAME LIKE ?", (f"%{roll_number}%",))
    exists = cursor.fetchone()[0]

    if exists > 0:
        print(f"Skipping {roll_number}, student already exists in the database.")
        continue  # Skip if roll number is found

    # Process images inside the student's folder
    for filename in os.listdir(roll_path):
        img_path = os.path.join(roll_path, filename)

        # Filter only image files
        if not img_path.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')):
            continue

        try:
            objs = DeepFace.represent(
                img_path=img_path,
                model_name='Facenet',
                detector_backend='retinaface',
                enforce_detection=False
            )

            for obj in objs:
                embedding = obj["embedding"]
                instances.append((img_path, roll_number, embedding))

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

print(f"New students found: {len(instances)}")

# Get the max ID from IDENTITIES
cursor.execute("SELECT MAX(ID) FROM IDENTITIES")
max_id = cursor.fetchone()[0]
if max_id is None:
    max_id = 0

# Insert new student records
for idx, (img_name, roll_number, embeddings) in enumerate(instances, start=max_id + 1):
    insert_identity_stmt = "INSERT INTO IDENTITIES(ID, IMG_NAME, EMBEDDING) VALUES (?,?,?)"
    insert_identity_args = (idx, img_name, np.array(embeddings).tobytes())
    cursor.execute(insert_identity_stmt, insert_identity_args)

    # Insert embeddings into EMBEDDING table
    for idy, embedding in enumerate(embeddings):
        insert_embedding_stmt = "INSERT INTO EMBEDDING (FACE_ID, DIM_NUM, VALUE) VALUES (?,?,?)"
        insert_embedding_args = (idx, idy, embedding)
        cursor.execute(insert_embedding_stmt, insert_embedding_args)

con.commit()
con.close()

print("Database update complete.")

