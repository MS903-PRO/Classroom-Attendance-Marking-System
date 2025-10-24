import os
from deepface import DeepFace
import sqlite3
import numpy as np
import sys

instances = []
if len(sys.argv) < 2:
    print("No roll number provided.")
    sys.exit(1)

roll_num = sys.argv[1]

# Traverse the directory tree starting from "SWE_DATA"
for dirpath, dirnames, filenames in os.walk(f"SWE_DATA_augmented/{roll_num}"):
    for filename in filenames:
        img_path = os.path.join(dirpath, filename)  # Safer path construction

        # Modify this line to include common image extensions
        if not img_path.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')):
            continue

        #print(f"Processing image: {img_path}")  # Debugging step to see the image being processed

        try:
            # DeepFace recognition
            objs = DeepFace.represent(
                img_path=img_path,
                model_name='Facenet',
                detector_backend='retinaface',
                enforce_detection=False
            )

            # Process embeddings
            for obj in objs:
                embedding = obj["embedding"]
                instances.append((img_path, embedding))

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

print(f"Total instances found: {len(instances)}")

con = sqlite3.connect("facialdb.db")
cursor = con.cursor()
cursor.execute("SELECT MAX(ID) FROM IDENTITIES")
max_id = cursor.fetchone()[0]  # This fetches the maximum ID in the table

# If no rows exist, max_id will be None. So, we set it to 0 (or 1 if IDs should start from 1).
if max_id is None:
    max_id = 0

# Step 2: Insert new records starting from max_id + 1
for idx, instance in enumerate(instances, start=max_id + 1):  # Start idx from max_id + 1
    img_name = instance[0]
    embeddings = instance[1]
    
    # Insert into IDENTITIES table
    insert_identity_stmt = "INSERT INTO IDENTITIES(ID, IMG_NAME, EMBEDDING) VALUES (?,?,?)"
    insert_identity_args = (idx, img_name, np.array(embeddings).tobytes())
    cursor.execute(insert_identity_stmt, insert_identity_args)

    # Insert each embedding into EMBEDDING table
    for idy, embedding in enumerate(embeddings):
        insert_embedding_stmt = "INSERT INTO EMBEDDING (FACE_ID, DIM_NUM, VALUE) VALUES (?,?,?)"
        insert_embedding_args = (idx, idy, embedding)
        cursor.execute(insert_embedding_stmt, insert_embedding_args)

con.commit()

print("Student added successfully\n")
