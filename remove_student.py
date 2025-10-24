import sqlite3
import os
import shutil
import sys

def remove_student_by_roll_number(db_path, roll_number, base_folder_path):
    """
    Remove all entries for a student from the database and delete their corresponding subfolder.

    Parameters:
    - db_path: Path to the SQLite database file.
    - roll_number: The roll number of the student to remove (part of IMG_NAME).
    - base_folder_path: Path to the base folder containing subfolders named by roll numbers.
    """
    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Step 1: Find all student IDs in the IDENTITIES table using the roll number
        cursor.execute("SELECT ID FROM IDENTITIES WHERE IMG_NAME LIKE ?", (f"%{roll_number}%",))
        student_ids = cursor.fetchall()  # Fetch all matching IDs

        if not student_ids:
            print(f"No students found with roll number: {roll_number}")
            return

        # Step 2: Delete all embeddings for each student ID from the EMBEDDING table
        for student_id in student_ids:
            student_id = student_id[0]  # Extract the ID from the tuple
            cursor.execute("DELETE FROM EMBEDDING WHERE FACE_ID = ?", (student_id,))

        # Step 3: Delete all records for the roll number from the IDENTITIES table
        cursor.execute("DELETE FROM IDENTITIES WHERE IMG_NAME LIKE ?", (f"%{roll_number}%",))

        # Commit the transaction
        conn.commit()
        print(f"All entries for roll number {roll_number} have been removed from the database.")

        # Step 4: Delete the subfolder with the roll number as its name
        subfolder_path = os.path.join(base_folder_path, roll_number)
        if os.path.exists(subfolder_path):
            shutil.rmtree(subfolder_path)  # Recursively delete the subfolder
            print(f"Subfolder '{roll_number}' has been deleted from '{base_folder_path}'.")
        else:
            print(f"No subfolder found with name: {roll_number}")

    except sqlite3.Error as e:
        print(f"An error occurred while accessing the database: {e}")
    except OSError as e:
        print(f"An error occurred while deleting the subfolder: {e}")
    finally:
        # Close the connection
        if conn:
            conn.close()

db_path = "facialdb.db"  # Replace with the path to your database
base_folder_path = "SWE_DATA_augmented"  # Replace with the path to the base folder containing subfolders

if len(sys.argv) < 2:
    print("No roll number provided.")
    sys.exit(1)

roll_number = sys.argv[1]

print(f"Removing student: {roll_number}")

# Remove all entries for the student from the database and delete their subfolder
remove_student_by_roll_number(db_path, roll_number, base_folder_path)
