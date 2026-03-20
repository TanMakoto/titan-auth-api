import sys
import os
import pickle
import numpy as np
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from face_module.face_recognizer import FaceRecognizer

def cosine_similarity(v1, v2):
    v1 = np.asarray(v1, dtype=np.float32)
    v2 = np.asarray(v2, dtype=np.float32)
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

with open("database/embeddings.pkl", "rb") as f:
    db = pickle.load(f)

enrolled_face_mean = db["6612247018"]["face_mean"]

face_rec = FaceRecognizer(model_name="VGG-Face")
try:
    test_emb = face_rec.extract_features("last_scan.jpg", enforce_detection=False)
    sim = cosine_similarity(test_emb[0]["embedding"], enrolled_face_mean)
    print(f"Similarity without detection: {sim:.4f}")
except Exception as e:
    print(e)
