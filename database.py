from pymongo import MongoClient
from datetime import datetime
import gridfs
import os

# Read from environment variable (set in Render dashboard)
MONGO_URI = os.environ.get(
    "MONGO_URI",
    "mongodb+srv://vidyadasari03_db_user:oTaCPVY5JonytYI6@cluster0.ydvyefy.mongodb.net/xray_db?retryWrites=true&w=majority&appName=Cluster0"
)

client = MongoClient(MONGO_URI)
db     = client["xray_db"]

# Collections
patients_col  = db["patients"]
analyses_col  = db["analyses"]
fs            = gridfs.GridFS(db)

# PATIENT FUNCTIONS

def save_patient(name, phone, age, gender):
    doc = {
        "name":       name,
        "phone":      phone,
        "age":        age,
        "gender":     gender,
        "created_at": datetime.utcnow()
    }
    result = patients_col.insert_one(doc)
    return str(result.inserted_id)

def get_patient_by_phone(phone):
    return patients_col.find_one({"phone": phone})

# ANALYSIS FUNCTIONS

def save_analysis(patient_id, body_part, disease,
                  confidence, report_text, language="English"):
    doc = {
        "patient_id":  patient_id,
        "body_part":   body_part,
        "disease":     disease,
        "confidence":  round(float(confidence), 2),
        "report_text": report_text,
        "language":    language,
        "timestamp":   datetime.utcnow()
    }
    result = analyses_col.insert_one(doc)
    return str(result.inserted_id)

def get_patient_history(patient_id):
    return list(analyses_col.find(
        {"patient_id": patient_id}
    ).sort("timestamp", -1))

# PDF STORAGE

def save_pdf(pdf_path, patient_name, analysis_id):
    with open(pdf_path, "rb") as f:
        file_id = fs.put(
            f,
            filename=f"{patient_name}_{analysis_id}.pdf"
        )
    return str(file_id)

def get_pdf(file_id):
    from bson import ObjectId
    return fs.get(ObjectId(file_id)).read()

# TEST CONNECTION
try:
    client.admin.command('ping')
    print("MongoDB connected successfully")
    print("Database: xray_db")
except Exception as e:
    print(f"MongoDB connection failed: {e}")
