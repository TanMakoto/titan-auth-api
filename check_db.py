import pickle
import os

DB_PATH = "g:/gait_face_auth-main/database/embeddings.pkl"
if os.path.exists(DB_PATH):
    with open(DB_PATH, "rb") as f:
        db = pickle.load(f)
    print("Keys in database:", list(db.keys()))
    for key, val in db.items():
        print(f"User: {key}, Has Face Embeddings: {len(val.get('face_embeddings', [])) > 0}")
else:
    print("Database not found")
