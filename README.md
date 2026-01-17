# Face Recognition System (Scalable • Modular • Review-Ready)

A **scalable face recognition system** built using deep-learning face embeddings and a vector database.  
The system is designed to **safely recognize up to 1,000 users**, focusing on **accuracy, edge-case handling, and clean architecture**.

---

## 🚀 Key Features

- 🔍 Robust face detection using InsightFace
- 🧠 Embedding-based recognition (512-D vectors)
- 🗄️ Vector database (ChromaDB) for fast similarity search
- ⚖️ Weighted Top-K voting to reduce false positives
- 🚦 Safe decision states: **MATCH / UNCERTAIN / UNKNOWN**
- 🧑 Guided enrollment UI with **mandatory vs optional image types**
- 🧪 Evaluation & testing utilities
- 📦 Modular, production-friendly codebase

---

## 🧠 System Flow (High Level)

Input Image
↓
Face Detection
↓
Quality Validation
↓
Embedding Generation
↓
Vector Database (ChromaDB)
↓
Top-K Similarity Search
↓
Weighted Voting
↓
Decision:

MATCH

UNCERTAIN

UNKNOWN

yaml
Copy code

---

## 📂 Project Structure

face_recognition_system/
│
├── src/ # Core ML logic
│ ├── detector.py # Face detection
│ ├── quality.py # Quality filtering
│ ├── embedder.py # Embedding extraction
│ ├── database.py # Vector DB operations
│ ├── matcher.py # Matching & voting logic
│ ├── visualizer.py # Bounding box & labels
│ └── utils.py
│
├── scripts/
│ ├── enroll_users.py # Batch enrollment
│ └── evaluate.py # Accuracy evaluation
│
├── data/
│ ├── enroll/ # Enrollment images (ignored by git)
│ └── test/ # Test images (ignored by git)
│
├── vector_db/ # Persistent embeddings (ignored by git)
│
├── ui_enroll.py # Guided enrollment UI (Streamlit)
├── ui_detect.py # Face detection test UI
├── app.py # Recognition pipeline (glue)
│
├── test_empty_db.py # Empty DB sanity test
├── requirements.txt
└── docs/
└── face_recognition_system.md

yaml
Copy code

---

## 🧪 Image Enrollment Strategy

### Mandatory Images (Minimum Required)
- **Front (neutral)**
- **Left profile**
- **Right profile**

### Optional Images (Recommended)
- Smile / expression
- With glasses
- Low light
- Bright light
- Slight blur

**Rules**
- Minimum: **3 images per user**
- Recommended: **6–8 images per user**
- One face per image

---

## ⚖️ Matching Logic

| Distance Range | Decision |
|---|---|
| `< 0.40` | MATCH |
| `0.40 – 0.50` | UNCERTAIN |
| `> 0.50` | UNKNOWN |

> The system prefers **saying UNKNOWN over making a wrong match**, reducing security risk.

---

## 🧪 How to Run & Test

### 1️⃣ Setup Environment
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
2️⃣ Detection Test (UI)
bash
Copy code
streamlit run ui_detect.py
Use this to verify whether a new input image is detectable.

3️⃣ Enrollment (UI)
bash
Copy code
streamlit run ui_enroll.py
Upload images

Select image type

Store embeddings in vector DB

4️⃣ Empty DB Safety Test
bash
Copy code
python test_empty_db.py
Expected:

makefile
Copy code
Decision: UNKNOWN
5️⃣ Batch Enrollment (Optional)
bash
Copy code
python scripts/enroll_users.py
6️⃣ Evaluation
bash
Copy code
python scripts/evaluate.py
Generates accuracy and false-match statistics.

🧰 Technology Stack
Component	Version
Python	3.10.x
InsightFace	0.7.3
ONNX Runtime	1.23.2
ChromaDB	1.4.1
OpenCV	Headless / Standard
Streamlit	Latest

⚠️ Windows Build Wheel Issue (Important)
Problem
InsightFace includes C++ extensions → wheel build fails on Windows without MSVC.

Solution

Install Microsoft Visual C++ Build Tools

Select:

C++ build tools

MSVC v14.x

Windows 10/11 SDK

Restart system

Install with:

bash
Copy code
pip install insightface==0.7.3 --prefer-binary
Alternative (Recommended)
Use Linux / Google Colab / Docker for zero build issues.

🛑 Limitations
No anti-spoofing (photo/video attacks not handled)

Masked or heavily occluded faces may fail

Extreme pose angles (>60°) reduce accuracy

Video / real-time recognition not implemented

📈 Scalability Notes
Users	Status
≤100	Excellent
500	Stable
1,000	Production-ready
>5,000	Requires FAISS / sharding

🎯 Outcomes of the Project
Clean, modular ML system

Real-world enrollment strategy

Safe recognition decisions

Strong edge-case handling

Review-ready architecture & documentation

🔮 Future Improvements
FastAPI inference service

Anti-spoofing module

Video / webcam recognition

Cloud deployment (Docker + Linux)

Audit & logging layer

