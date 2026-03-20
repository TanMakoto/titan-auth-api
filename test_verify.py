import sys
import os
import glob
from deepface import DeepFace

target_img = "last_scan.jpg"
enrolled_imgs = glob.glob("dataset/6612247018/data1/*.jpg")

if enrolled_imgs:
    source_img = enrolled_imgs[0]
    print(f"Comparing {target_img} with {source_img}...")
    try:
        result = DeepFace.verify(
            img1_path=source_img, 
            img2_path=target_img, 
            model_name="VGG-Face", 
            detector_backend="retinaface",
            distance_metric="cosine",
            enforce_detection=False
        )
        print("Verification with retinaface (no enforce):")
        print(result)
        
        result2 = DeepFace.verify(
            img1_path=source_img, 
            img2_path=target_img, 
            model_name="VGG-Face", 
            detector_backend="opencv",
            distance_metric="cosine",
            enforce_detection=False
        )
        print("\nVerification with opencv (no enforce):")
        print(result2)
    except Exception as e:
        print(e)
