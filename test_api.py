import requests
import numpy as np
import cv2

img = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.imwrite("dummy_face.jpg", img)

with open("dummy_face.jpg", "rb") as f:
    files = {"file": ("dummy_face.jpg", f, "image/jpeg")}
    data = {"user_id": "6612247018"}
    try:
        r = requests.post("http://localhost:8000/api/verify_face", files=files, data=data)
        print("Status Code:", r.status_code)
        print("Response:", r.json())
    except Exception as e:
        print("Exception:", e)
