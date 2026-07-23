import sys
import os
import pickle
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
try:
    import gdown
    GDOWN_AVAILABLE = True
except ImportError:
    GDOWN_AVAILABLE = False
    print("Warning: gdown not installed. Cloud auto-download disabled.")

# Add src to sys.path to resolve internal modules
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# import modules from the project
from face_module.face_recognizer import FaceRecognizer
from deepface import DeepFace

app = FastAPI(title="Gait & Face Auth API")

# Setup CORS to allow React frontend to connect
# Reads allowed origins from ALLOWED_ORIGINS env var (comma-separated), default allows all
_allowed_origins_env = os.environ.get("ALLOWED_ORIGINS", "*")
ALLOWED_ORIGINS = [o.strip() for o in _allowed_origins_env.split(",")] if _allowed_origins_env != "*" else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
DB_PATH = os.path.join(PROJECT_ROOT, "database", "embeddings.pkl")

# Initialize FaceRecognizer
face_rec = FaceRecognizer(model_name="VGG-Face")

def download_embeddings_if_needed():
    """
    Download embeddings.pkl from Google Drive if not present.
    Normally this file is committed to git directly (424KB).
    This fallback is for environments where the file is missing.
    """
    if os.path.exists(DB_PATH):
        print(f"✅ Database found at {DB_PATH}")
        return

    gdrive_id = os.environ.get("EMBEDDINGS_GDRIVE_ID", "")
    if not gdrive_id:
        print("⚠️  Warning: Database not found. Set EMBEDDINGS_GDRIVE_ID env var to download it.")
        return

    if not GDOWN_AVAILABLE:
        print("⚠️  Warning: gdown not installed. Cannot auto-download embeddings.")
        return

    print(f"☁️  Downloading embeddings.pkl from Google Drive (id: {gdrive_id})...")
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    url = f"https://drive.google.com/uc?id={gdrive_id}"
    gdown.download(url, DB_PATH, quiet=False)
    if os.path.exists(DB_PATH):
        print("✅ embeddings.pkl downloaded successfully!")
    else:
        print("❌ Failed to download embeddings.pkl.")

# Load User DB (download from GDrive on Cloud if needed)
download_embeddings_if_needed()

user_db = {}
if os.path.exists(DB_PATH):
    with open(DB_PATH, "rb") as f:
        user_db = pickle.load(f)
    print(f"✅ Loaded {len(user_db)} users from database.")
else:
    print(f"⚠️  Warning: Database not found at {DB_PATH}. Face verification will fail.")

def cosine_similarity(v1, v2):
    if v1 is None or v2 is None:
        return 0.0
    v1 = np.asarray(v1, dtype=np.float32)
    v2 = np.asarray(v2, dtype=np.float32)
    if v1.shape != v2.shape:
        return 0.0
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))

@app.on_event("startup")
async def startup_event():
    # Preload the VGG-Face model into memory at startup to avoid warm-up delays during scans
    print("Pre-loading DeepFace VGG-Face model into memory... This speeds up scanning.")
    DeepFace.build_model("VGG-Face")
    print("DeepFace VGG-Face model pre-loaded.")

@app.get("/")
def read_root():
    return {"message": "Gait & Face Auth API is running"}

@app.post("/api/verify_face")
async def verify_face(
    user_id: str = Form(...),
    file: UploadFile = File(...)
):
    with open(os.path.join(PROJECT_ROOT, "api_debug.log"), "a", encoding="utf-8") as f_log:
        f_log.write(f"\n--- Incoming Request: user_id={user_id} ---\n")
        
        if user_id not in user_db:
            f_log.write(f"Error: User {user_id} not in user_db.\n")
            return {"match": False, "score": 0.0, "message": f"User {user_id} not enrolled in database"}
        
        enrolled_data = user_db[user_id]
        enrolled_face_mean = enrolled_data.get("face_mean")
        
        if enrolled_face_mean is None:
            f_log.write(f"Error: User {user_id} has no enrolled face data\n")
            return {"match": False, "score": 0.0, "message": f"User {user_id} does not have enrolled face data"}

        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            f_log.write("Error: Failed to decode image\n")
            return {"match": False, "score": 0.0, "message": "Failed to decode image"}
        
        # Save temp file for DeepFace
        temp_path = os.path.join(PROJECT_ROOT, f"temp_api_{user_id}.jpg")
        try:
            cv2.imwrite(temp_path, img)
            # Make a copy for debugging so it's not deleted
            cv2.imwrite(os.path.join(PROJECT_ROOT, "last_scan.jpg"), img)
            f_log.write(f"Saved temp image: {temp_path}\n")
            
            # Use enforce_detection=True with "ssd" backend for 3x FATSER speed while keeping security
            # SSD is significantly faster than RetinaFace and prevents the camera-cover bypass nicely
            try:
                test_emb = DeepFace.represent(img_path=temp_path, model_name="VGG-Face", enforce_detection=True, detector_backend="ssd")
            except ValueError as ve:
                if "could not be detected" in str(ve).lower() or "face" in str(ve).lower():
                    f_log.write("Error: No face detected in the image by ssd\n")
                    return {"match": False, "score": 0.0, "message": "No face detected in the image"}
                raise ve
            
            if test_emb and len(test_emb) > 0:
                v1 = test_emb[0]["embedding"]
                sim = cosine_similarity(v1, enrolled_face_mean)
                f_log.write(f"Similarity Score: {sim:.4f}\n")
                
                # Set threshold to 0.35 (adjusted from 0.50) 
                # to tolerate the difference between a high-res phone photo and a low-res webcam image
                is_match = sim > 0.35
                f_log.write(f"Match: {is_match}\n")
                
                return {
                    "match": is_match,
                    "score": float(sim),
                    "message": "Verification successful" if is_match else f"Face does not match (Score: {sim:.2f})"
                }
            else:
                f_log.write("Error: No face detected in the image\n")
                return {"match": False, "score": 0.0, "message": "No face detected in the image"}
                
        except Exception as e:
            f_log.write(f"Exception: {str(e)}\n")
            return {"match": False, "score": 0.0, "message": f"Error processing image: {str(e)}"}
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass


@app.post("/api/enroll")
async def enroll_face(
    user_id: str = Form(...),
    file: UploadFile = File(...)
):
    with open(os.path.join(PROJECT_ROOT, "api_debug.log"), "a", encoding="utf-8") as f_log:
        f_log.write(f"\n--- Incoming Enrollment Request: user_id={user_id} ---\n")
        
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            f_log.write("Enrollment Error: Failed to decode image\n")
            return {"success": False, "message": "Failed to decode image"}
        
        temp_path = os.path.join(PROJECT_ROOT, f"temp_enroll_{user_id}.jpg")
        try:
            cv2.imwrite(temp_path, img)
            f_log.write(f"Saved temp enroll image: {temp_path}\n")
            
            # Extract face embedding using DeepFace VGG-Face with ssd detector
            try:
                emb_res = DeepFace.represent(img_path=temp_path, model_name="VGG-Face", enforce_detection=True, detector_backend="ssd")
            except ValueError as ve:
                if "could not be detected" in str(ve).lower() or "face" in str(ve).lower():
                    f_log.write("Enrollment Error: No face detected in the image by ssd\n")
                    return {"success": False, "message": "No face detected. Please position your face clearly in front of the camera."}
                raise ve
            
            if emb_res and len(emb_res) > 0:
                embedding = emb_res[0]["embedding"]
                
                # Save to user_db dictionary
                user_db[user_id] = {
                    'face_embeddings': [embedding],
                    'face_mean': np.array(embedding, dtype=np.float32),
                    'gait_features': None
                }
                
                # Write database back to file
                with open(DB_PATH, "wb") as f:
                    pickle.dump(user_db, f)
                
                f_log.write(f"Enrollment Success: Registered face for user_id={user_id}\n")
                return {"success": True, "message": f"Successfully enrolled face for student ID {user_id}"}
            else:
                f_log.write("Enrollment Error: No face detected in the image\n")
                return {"success": False, "message": "No face detected in the image"}
                
        except Exception as e:
            f_log.write(f"Enrollment Exception: {str(e)}\n")
            return {"success": False, "message": f"Error enrolling face: {str(e)}"}
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
