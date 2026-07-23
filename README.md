---
title: Titan Auth API
emoji: 🎭
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
license: mit
---

# Titan Auth API — Face Recognition Service

FastAPI backend for PSRU Smart Attendance System.
Performs face verification using DeepFace (VGG-Face model).

## Endpoints
- `GET /` — Health check
- `POST /api/verify_face` — Verify face against enrolled user
- `POST /api/enroll` — Enroll a new user's face
