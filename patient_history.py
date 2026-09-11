import sqlite3
from datetime import datetime

DATABASE_NAME = "patient_history.db"


def create_database():
    connection = sqlite3.connect(DATABASE_NAME)
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS patient_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_name TEXT NOT NULL,
            gender TEXT,
            age INTEGER,
            disease TEXT NOT NULL,
            prediction TEXT,
            confidence REAL,
            date_time TEXT,
            input_data TEXT
        )
    """)

    connection.commit()
    connection.close()


def save_patient_record(
    patient_name,
    gender,
    age,
    disease,
    prediction,
    confidence,
    input_data
):
    connection = sqlite3.connect(DATABASE_NAME)
    cursor = connection.cursor()

    date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute("""
        INSERT INTO patient_history
        (
            patient_name,
            gender,
            age,
            disease,
            prediction,
            confidence,
            date_time,
            input_data
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        patient_name,
        gender,
        age,
        disease,
        prediction,
        confidence,
        date_time,
        input_data
    ))

    connection.commit()
    connection.close()


def get_patient_history(patient_name=None):
    connection = sqlite3.connect(DATABASE_NAME)

    if patient_name:

        query = """
            SELECT
                id,
                patient_name,
                gender,
                age,
                disease,
                prediction,
                confidence,
                date_time,
                input_data
            FROM patient_history
            WHERE patient_name LIKE ?
            ORDER BY id DESC
        """

        data = connection.execute(
            query,
            (f"%{patient_name}%",)
        ).fetchall()

    else:

        query = """
            SELECT
                id,
                patient_name,
                gender,
                age,
                disease,
                prediction,
                confidence,
                date_time,
                input_data
            FROM patient_history
            ORDER BY id DESC
        """

        data = connection.execute(query).fetchall()

    connection.close()

    return data


# ==========================================
# 👤 GET EXISTING PATIENT NAMES
# ==========================================

def get_patient_names():

    connection = sqlite3.connect(DATABASE_NAME)

    query = """
        SELECT DISTINCT patient_name
        FROM patient_history
        WHERE patient_name IS NOT NULL
        AND patient_name != ''
        ORDER BY patient_name
    """

    data = connection.execute(query).fetchall()

    connection.close()

    return [row[0] for row in data]