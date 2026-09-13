import sqlite3
import uuid
import hashlib
import secrets
from datetime import datetime

DATABASE_NAME = "patient_history.db"


def generate_patient_id():
    return "NIR-P-" + uuid.uuid4().hex[:8].upper()


def generate_doctor_id():
    return "NIR-D-" + uuid.uuid4().hex[:8].upper()


def _hash_password(password, salt=None):
    if salt is None:
        salt = secrets.token_bytes(16)
    password_hash = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, 120000
    )
    return salt.hex(), password_hash.hex()


def _verify_password(password, salt_hex, stored_hash):
    try:
        salt = bytes.fromhex(salt_hex)
    except ValueError:
        return False
    _, password_hash = _hash_password(password, salt)
    return secrets.compare_digest(password_hash, stored_hash)


def create_database():
    connection = sqlite3.connect(DATABASE_NAME)
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS patient_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT,
            doctor_id TEXT,
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

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS doctors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doctor_id TEXT UNIQUE NOT NULL,
            doctor_name TEXT NOT NULL,
            gender TEXT,
            domain TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT UNIQUE NOT NULL,
            user_type TEXT NOT NULL,
            name TEXT NOT NULL,
            gender TEXT,
            age INTEGER,
            domain TEXT,
            password_hash TEXT NOT NULL,
            password_salt TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS patient_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT NOT NULL,
            patient_name TEXT NOT NULL,
            file_name TEXT NOT NULL,
            file_type TEXT,
            file_path TEXT NOT NULL,
            extracted_text TEXT,
            uploaded_at TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS patient_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT NOT NULL,
            patient_name TEXT NOT NULL,
            message TEXT NOT NULL,
            doctor_id TEXT,
            doctor_name TEXT,
            reply TEXT,
            created_at TEXT NOT NULL,
            replied_at TEXT
        )
    """)

    connection.commit()

    columns = [row[1] for row in cursor.execute("PRAGMA table_info(patient_history)").fetchall()]

    if "patient_id" not in columns:
        cursor.execute("ALTER TABLE patient_history ADD COLUMN patient_id TEXT")

    if "doctor_id" not in columns:
        cursor.execute("ALTER TABLE patient_history ADD COLUMN doctor_id TEXT")

    rows = cursor.execute("""
        SELECT id FROM patient_history
        WHERE patient_id IS NULL OR patient_id = ''
    """).fetchall()

    for row in rows:
        cursor.execute(
            "UPDATE patient_history SET patient_id = ? WHERE id = ?",
            (generate_patient_id(), row[0])
        )

    connection.commit()
    connection.close()


def register_patient(name, gender, age, password):
    name = str(name).strip()
    if not name or not password:
        return None, "Name and password are required."
    connection = sqlite3.connect(DATABASE_NAME)
    cursor = connection.cursor()
    user_id = generate_patient_id()
    salt, password_hash = _hash_password(password)
    try:
        cursor.execute("""
            INSERT INTO users
            (user_id, user_type, name, gender, age, domain,
             password_hash, password_salt, created_at)
            VALUES (?, 'Patient', ?, ?, ?, NULL, ?, ?, ?)
        """, (
            user_id, name, gender, age, password_hash, salt,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))
        connection.commit()
        return user_id, None
    except sqlite3.IntegrityError:
        return None, "Could not create the Patient ID. Please try again."
    finally:
        connection.close()


def register_doctor(name, gender, domain, password):
    name = str(name).strip()
    if not name or not password:
        return None, "Name and password are required."
    connection = sqlite3.connect(DATABASE_NAME)
    cursor = connection.cursor()
    doctor_id = generate_doctor_id()
    salt, password_hash = _hash_password(password)
    try:
        cursor.execute("""
            INSERT INTO users
            (user_id, user_type, name, gender, age, domain,
             password_hash, password_salt, created_at)
            VALUES (?, 'Doctor', ?, ?, NULL, ?, ?, ?, ?)
        """, (
            doctor_id, name, gender, domain, password_hash, salt,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))
        cursor.execute("""
            INSERT INTO doctors (doctor_id, doctor_name, gender, domain)
            VALUES (?, ?, ?, ?)
        """, (doctor_id, name, gender, domain))
        connection.commit()
        return doctor_id, None
    except sqlite3.IntegrityError:
        connection.rollback()
        return None, "Could not create the Doctor ID. Please try again."
    finally:
        connection.close()


def login_user(user_id, password, user_type):
    connection = sqlite3.connect(DATABASE_NAME)
    row = connection.execute("""
        SELECT user_id, user_type, name, gender, age, domain,
               password_hash, password_salt
        FROM users
        WHERE user_id = ? AND user_type = ?
    """, (user_id.strip().upper(), user_type)).fetchone()
    connection.close()

    if not row:
        return None
    if not _verify_password(password, row[7], row[6]):
        return None

    return {
        "user_id": row[0],
        "user_type": row[1],
        "name": row[2],
        "gender": row[3],
        "age": row[4],
        "domain": row[5],
    }


def get_user(user_id):
    connection = sqlite3.connect(DATABASE_NAME)
    row = connection.execute("""
        SELECT user_id, user_type, name, gender, age, domain, created_at
        FROM users WHERE user_id = ?
    """, (user_id.strip().upper(),)).fetchone()
    connection.close()
    return row


def get_or_create_patient_id(patient_name):
    connection = sqlite3.connect(DATABASE_NAME)
    row = connection.execute("""
        SELECT user_id FROM users
        WHERE user_type = 'Patient' AND name = ?
        ORDER BY id ASC LIMIT 1
    """, (patient_name.strip(),)).fetchone()
    if row:
        patient_id = row[0]
    else:
        row = connection.execute("""
            SELECT patient_id FROM patient_history
            WHERE patient_name = ? AND patient_id IS NOT NULL
            AND patient_id != '' ORDER BY id ASC LIMIT 1
        """, (patient_name,)).fetchone()
        patient_id = row[0] if row else generate_patient_id()
    connection.close()
    return patient_id


def save_patient_record(patient_name, gender, age, disease, prediction,
                        confidence, input_data, patient_id=None, doctor_id=None):
    connection = sqlite3.connect(DATABASE_NAME)
    cursor = connection.cursor()
    if patient_id is None:
        patient_id = get_or_create_patient_id(patient_name)
    date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        INSERT INTO patient_history
        (patient_id, doctor_id, patient_name, gender, age, disease,
         prediction, confidence, date_time, input_data)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (patient_id, doctor_id, patient_name, gender, age, disease,
          prediction, confidence, date_time, input_data))
    connection.commit()
    record_id = cursor.lastrowid
    connection.close()
    return record_id


def create_doctor(doctor_name, gender=None, domain=None):
    connection = sqlite3.connect(DATABASE_NAME)
    cursor = connection.cursor()
    doctor_id = generate_doctor_id()
    cursor.execute("""
        INSERT INTO doctors (doctor_id, doctor_name, gender, domain)
        VALUES (?, ?, ?, ?)
    """, (doctor_id, doctor_name, gender, domain))
    connection.commit()
    connection.close()
    return doctor_id


def get_patient_history_by_id(patient_id):
    connection = sqlite3.connect(DATABASE_NAME)
    data = connection.execute("""
        SELECT id, patient_id, doctor_id, patient_name, gender, age,
               disease, prediction, confidence, date_time, input_data
        FROM patient_history WHERE patient_id = ? ORDER BY id DESC
    """, (patient_id.strip().upper(),)).fetchall()
    connection.close()
    return data


def get_patient_history(patient_name=None):
    connection = sqlite3.connect(DATABASE_NAME)
    if patient_name:
        data = connection.execute("""
            SELECT id, patient_id, doctor_id, patient_name, gender, age,
                   disease, prediction, confidence, date_time, input_data
            FROM patient_history WHERE patient_name LIKE ? ORDER BY id DESC
        """, (f"%{patient_name}%",)).fetchall()
    else:
        data = connection.execute("""
            SELECT id, patient_id, doctor_id, patient_name, gender, age,
                   disease, prediction, confidence, date_time, input_data
            FROM patient_history ORDER BY id DESC
        """).fetchall()
    connection.close()
    return data



def get_patient_by_name(name):
    """Return the first registered patient matching a name."""
    connection = sqlite3.connect(DATABASE_NAME)
    row = connection.execute("""
        SELECT user_id, user_type, name, gender, age, domain, created_at
        FROM users
        WHERE user_type = 'Patient' AND name LIKE ?
        ORDER BY id ASC LIMIT 1
    """, (f"%{str(name).strip()}%",)).fetchone()
    connection.close()
    return row


def get_patient_by_id(patient_id):
    """Return a registered patient by NIRAMAYA Patient ID."""
    connection = sqlite3.connect(DATABASE_NAME)
    row = connection.execute("""
        SELECT user_id, user_type, name, gender, age, domain, created_at
        FROM users WHERE user_id = ? AND user_type = 'Patient'
    """, (patient_id.strip().upper(),)).fetchone()
    connection.close()
    return row


def save_patient_report(patient_id, patient_name, file_name, file_type, file_path, extracted_text=''):
    """Store metadata for a patient-uploaded medical report."""
    connection = sqlite3.connect(DATABASE_NAME)
    cursor = connection.cursor()
    cursor.execute("""
        INSERT INTO patient_reports
        (patient_id, patient_name, file_name, file_type, file_path, extracted_text, uploaded_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        patient_id.strip().upper(), patient_name, file_name, file_type,
        file_path, extracted_text, datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ))
    connection.commit()
    report_id = cursor.lastrowid
    connection.close()
    return report_id


def get_patient_reports(patient_id):
    """Return all uploaded reports for a patient, newest first."""
    connection = sqlite3.connect(DATABASE_NAME)
    data = connection.execute("""
        SELECT id, patient_id, patient_name, file_name, file_type, file_path, extracted_text, uploaded_at
        FROM patient_reports WHERE patient_id = ? ORDER BY id DESC
    """, (patient_id.strip().upper(),)).fetchall()
    connection.close()
    return data

def save_patient_message(patient_id, patient_name, message):
    """Save a message from a patient for a doctor to review."""
    message = str(message).strip()
    if not message:
        raise ValueError("Message cannot be empty.")
    connection = sqlite3.connect(DATABASE_NAME)
    cursor = connection.cursor()
    cursor.execute("""
        INSERT INTO patient_messages
        (patient_id, patient_name, message, doctor_id, doctor_name, reply, created_at, replied_at)
        VALUES (?, ?, ?, NULL, NULL, NULL, ?, NULL)
    """, (
        patient_id.strip().upper(), patient_name, message,
        datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ))
    connection.commit()
    message_id = cursor.lastrowid
    connection.close()
    return message_id


def get_patient_messages(patient_id):
    """Return all patient messages and doctor replies, newest first."""
    connection = sqlite3.connect(DATABASE_NAME)
    data = connection.execute("""
        SELECT id, patient_id, patient_name, message, doctor_id, doctor_name,
               reply, created_at, replied_at
        FROM patient_messages
        WHERE patient_id = ?
        ORDER BY id DESC
    """, (patient_id.strip().upper(),)).fetchall()
    connection.close()
    return data


def get_unanswered_patient_messages(patient_id=None):
    """Return messages awaiting a doctor's reply."""
    connection = sqlite3.connect(DATABASE_NAME)
    if patient_id:
        data = connection.execute("""
            SELECT id, patient_id, patient_name, message, doctor_id, doctor_name,
                   reply, created_at, replied_at
            FROM patient_messages
            WHERE patient_id = ? AND (reply IS NULL OR reply = '')
            ORDER BY id ASC
        """, (patient_id.strip().upper(),)).fetchall()
    else:
        data = connection.execute("""
            SELECT id, patient_id, patient_name, message, doctor_id, doctor_name,
                   reply, created_at, replied_at
            FROM patient_messages
            WHERE reply IS NULL OR reply = ''
            ORDER BY id ASC
        """).fetchall()
    connection.close()
    return data


def reply_to_patient_message(message_id, doctor_id, doctor_name, reply):
    """Store a doctor's reply to a patient message."""
    reply = str(reply).strip()
    if not reply:
        raise ValueError("Reply cannot be empty.")
    connection = sqlite3.connect(DATABASE_NAME)
    cursor = connection.cursor()
    cursor.execute("""
        UPDATE patient_messages
        SET doctor_id = ?, doctor_name = ?, reply = ?, replied_at = ?
        WHERE id = ?
    """, (
        doctor_id, doctor_name, reply,
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'), message_id
    ))
    connection.commit()
    updated = cursor.rowcount
    connection.close()
    return updated > 0


def get_patient_names():
    connection = sqlite3.connect(DATABASE_NAME)
    data = connection.execute("""
        SELECT DISTINCT patient_name FROM patient_history
        WHERE patient_name IS NOT NULL AND patient_name != ''
        ORDER BY patient_name
    """).fetchall()
    connection.close()
    return [row[0] for row in data]


def get_patient_id(patient_name):
    return get_or_create_patient_id(patient_name)


def get_doctor(doctor_id):
    connection = sqlite3.connect(DATABASE_NAME)
    row = connection.execute("""
        SELECT doctor_id, doctor_name, gender, domain
        FROM doctors WHERE doctor_id = ?
    """, (doctor_id.strip().upper(),)).fetchone()
    connection.close()
    return row


create_database()
