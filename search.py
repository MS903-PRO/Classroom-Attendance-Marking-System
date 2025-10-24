from deepface import DeepFace
from collections import defaultdict
import os
import cv2
import sqlite3
import numpy as np
import json

# Source folder containing captured images
source = os.path.join('captured_images')
folder = source

# Find all image files
image_files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
print(f"Found {len(image_files)} images in '{folder}'.")

# Keep track of all detected faces
global_face_idx = 0
all_faces = []
all_face_numbers = []

# Go through each image and extract faces
for file in image_files:
    img_path = os.path.join(folder, file)
    print("Processing image:", img_path)
    
    try:
        faces = DeepFace.extract_faces(img_path, detector_backend='retinaface', enforce_detection=False)
    except Exception as e:
        print(f"  Error extracting faces from {file}: {e}")
        continue
    
    if not faces or len(faces) == 0:
        print("  No faces detected in", file)
        continue
    
    # Sort faces by their position (bottom to top, right to left)
    faces_sorted = sorted(
        faces,
        key=lambda face: (face['facial_area']['y'], face['facial_area']['x']),
        reverse=True
    )
    
    # Add each face to our collection
    for face in faces_sorted:
        all_faces.append(face['face'])
        all_face_numbers.append(global_face_idx)
        global_face_idx += 1

print(f"Total faces detected: {global_face_idx}")


def group_consecutive_entries(detections):
    """Group detection results by consecutive matches of the same person"""
    if not detections:
        return []

    groups = []
    current_group = None

    for path, distance in detections:
        try:
            distance = float(distance)
            person_id = path.split('/')[-2]

            # Start a new group if person changes
            if current_group is None or person_id != current_group["person"]:
                if current_group is not None:
                    groups.append(current_group)
                current_group = {"person": person_id, "distances": [], "count": 0}

            current_group["distances"].append(distance)
            current_group["count"] += 1
        except (ValueError, TypeError, IndexError):
            continue

    if current_group is not None:
        groups.append(current_group)

    # Merge groups separated by single entries of different people
    merged_groups = []
    i = 0
    while i < len(groups):
        current_group = groups[i]
        person = current_group["person"]
        distances = current_group["distances"][:]
        count = current_group["count"]

        # Check if we can merge with a group two positions ahead
        if (i + 2 < len(groups) and 
            groups[i + 1]["count"] == 1 and 
            groups[i + 2]["person"] == person):
            distances.extend(groups[i + 1]["distances"])
            distances.extend(groups[i + 2]["distances"])
            count += groups[i + 1]["count"] + groups[i + 2]["count"]
            i += 3
        else:
            i += 1

        merged_groups.append({"person": person, "distances": distances, "count": count})

    return merged_groups


def calculate_group_score(group):
    """Calculate confidence score for a group of matches"""
    if group["count"] == 1:
        return None  # Single matches aren't reliable enough

    min_distance = min(group["distances"])
    if min_distance == 0:
        return float('inf')
    
    # Better scores = more matches with smaller distances
    return group["count"] / min_distance


def determine_final_person(groups):
    """Figure out which person is the best match"""
    if not groups:
        return "NULL", {}

    group_scores = {}
    
    for group in groups:
        score = calculate_group_score(group)
        if score is not None:
            person = group["person"]
            if person not in group_scores or score > group_scores[person]["score"]:
                group_scores[person] = {
                    "score": score,
                    "min_distance": min(group["distances"]),
                    "group_count": group_scores.get(person, {}).get("group_count", 0) + 1
                }

    if not group_scores:
        return "NULL", {}

    # Sort by score to find the best match
    sorted_candidates = sorted(
        group_scores.items(),
        key=lambda x: -x[1]["score"]
    )

    best_person = sorted_candidates[0][0]
    score_details = {pid: data for pid, data in sorted_candidates}

    return best_person, score_details


def generate_rankings(fdata):
    """Process face matching data to determine the best match"""
    if not fdata:
        return "NULL", {}

    try:
        processed_data = [(row[0], float(row[1])) for row in fdata if len(row) >= 2]

        # Count how many different people matched
        person_count = defaultdict(int)
        for path, _ in processed_data:
            person_id = path.split('/')[-2]
            person_count[person_id] += 1

        # Too many matches usually means a bad image
        if len(person_count) > 300:
            return "NULL", {}

        groups = group_consecutive_entries(processed_data)
        best_match, score_details = determine_final_person(groups)

        if best_match == "NULL":
            return "NULL", {}

        return best_match, score_details

    except Exception as e:
        print(f"Error in generate_rankings: {e}")
        return "NULL", {}


# Open database connection
con = sqlite3.connect("facialdb.db")
cursor = con.cursor()

output_data = []

# Match each detected face against the database
for face_idx, face in enumerate(all_faces):
    print(f"Matching face {face_idx + 1}/{len(all_faces)}...")
    
    try:
        # Convert face to correct format
        target_img = (face * 255).astype(np.uint8)
        
        # Get face embedding
        objs = DeepFace.represent(
            img_path=target_img,
            model_name="Facenet",
            enforce_detection=False
        )
        
        if not objs:
            output_data.append([face_idx, "NULL", 0.0])
            continue

        embedding = objs[0]["embedding"]
        target_stmts = []
        
        # Build the embedding query
        for dim_index, value in enumerate(embedding):
            target_stmt = f"SELECT {dim_index} AS dim_num, {value} AS value"
            target_stmts.append(target_stmt)
        
        target_stmt_final = " UNION ALL ".join(target_stmts)

        # Query to find similar faces in database
        # Using manual square root calculation since SQLite doesn't have SQRT
        select_stmt = f"""
        SELECT *
        FROM (
            SELECT img_name, SUM(value) AS distance_squared
            FROM (
                SELECT img_name, (source - target) * (source - target) AS value
                FROM (
                    SELECT IDENTITIES.img_name, EMBEDDING.value AS source, target.value AS target
                    FROM IDENTITIES 
                    LEFT JOIN EMBEDDING ON IDENTITIES.ID = EMBEDDING.FACE_ID
                    LEFT JOIN ({target_stmt_final}) target
                    ON EMBEDDING.dim_num = target.dim_num
                )
            )
            GROUP BY img_name
        )
        WHERE distance_squared < 132.25
        ORDER BY distance_squared ASC
        """
        
        cursor.execute(select_stmt)
        results = cursor.fetchall()

        # Calculate actual Euclidean distance from squared distance
        fdata = [[img_name, np.sqrt(distance_squared)] for img_name, distance_squared in results]

        # Figure out the best match
        final_identified_person, weighted_scores = generate_rankings(fdata)

        # Extract the confidence score
        best_score = 0.0
        if final_identified_person != "NULL" and final_identified_person in weighted_scores:
            best_score = weighted_scores[final_identified_person].get("score", 0.0)

        identified_person = final_identified_person if final_identified_person != "NULL" else "NULL"
        output_data.append([face_idx, identified_person, best_score])
        
    except Exception as e:
        print(f"Error processing face {face_idx}: {e}")
        output_data.append([face_idx, "NULL", 0.0])

con.close()


def remove_duplicates_keep_best(data):
    """If the same person is matched multiple times, keep only the best match"""
    if not data:
        return data

    best_entries = {}

    for entry in data:
        if len(entry) < 3:
            continue

        face_idx, person, score = entry

        # Skip invalid entries
        if not isinstance(person, str) or person == "NULL":
            continue

        try:
            if isinstance(score, dict):
                score = float(score.get("score", 0.0))
            else:
                score = float(score)
        except (ValueError, TypeError):
            continue

        # Keep the match with the highest score
        if person not in best_entries or score > best_entries[person][2]:
            best_entries[person] = [face_idx, person, score]

    return list(best_entries.values())


# Remove duplicate matches
output_data = remove_duplicates_keep_best(output_data)

print("\nFace matching complete...")

# Make sure we have data to save
if not all_faces:
    print("No faces detected! Cannot save results.")
    exit()

if not output_data:
    print("No matches found! Check if database has entries.")
    exit()

# Resize faces to standard size
FIXED_FACE_SIZE = (224, 224)
all_faces_resized = [cv2.resize(face, FIXED_FACE_SIZE) for face in all_faces]

# Save everything
np.save("faces.npy", np.array(all_faces_resized))
with open("output_data.json", "w") as f:
    json.dump(output_data, f, indent=4)

print("Results saved to faces.npy and output_data.json")

# Save individual face images
if not os.path.exists("recognized_faces"):
    os.makedirs("recognized_faces")

for face_idx, face in enumerate(all_faces):
    img_path = os.path.join("recognized_faces", f"{face_idx}.jpg")
    face_bgr = cv2.cvtColor((face * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(img_path, face_bgr)

identified_count = len([x for x in output_data if x[1] != 'NULL'])
print(f"Saved {len(all_faces)} faces to 'recognized_faces' folder")
print(f"Successfully identified {identified_count} out of {len(all_faces)} faces!")
