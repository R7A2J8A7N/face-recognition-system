# 🚀 Face Recognition System

**Scalable • Modular • Production-Oriented**

A production-style **face recognition intelligence system** built using
deep-learning embeddings and a vector database.

The system is engineered for:

-   ✅ Clean architecture\
-   ✅ Stateless recognition\
-   ✅ High accuracy\
-   ✅ Edge-case handling\
-   ✅ Scalable deployment

Designed to safely support **\~1,000 users** out of the box, with a
clear path toward hyperscale.

------------------------------------------------------------------------

# 🧠 Project Philosophy

This project focuses strictly on the **intelligence layer**.

It intentionally does **NOT** handle:

-   Camera hardware\
-   UI workflows\
-   Authentication\
-   User consent\
-   Business logic

👉 These belong to the host application.

The engine processes images and returns **confidence-based identity
decisions**.

------------------------------------------------------------------------

# 🔥 Key Features

-   🔍 Robust detection using InsightFace\
-   🧠 512-D deep face embeddings\
-   🗄️ Fast similarity search via ChromaDB\
-   ⚖️ Weighted Top-K voting to reduce false positives\
-   🚦 Safe decision states:
    -   MATCH\
    -   UNCERTAIN\
    -   UNKNOWN\
-   📦 Modular production-ready architecture\
-   🧪 Dataset-based batch enrollment\
-   🖼️ Debug output with bounding boxes\
-   ⚙️ Config-driven thresholds\
-   🧱 Stateless recognition pipeline

------------------------------------------------------------------------

# 🧠 System Flow

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
    MATCH | UNCERTAIN | UNKNOWN

------------------------------------------------------------------------

# 📂 Project Structure

    face_recognition_system/
    │
    ├── dataset/              # Enrollment images
    │   ├── user_1/
    │   └── user_2/
    │
    ├── test_images/          # Images used ONLY for recognition testing
    │
    ├── output/               # Auto-saved recognition results
    │
    ├── face_db/              # Persistent vector database
    │
    ├── logs/                 # Runtime logs
    │
    ├── src/
    │   ├── core/             # Intelligence layer
    │   │   detector.py
    │   │   quality.py
    │   │   embedder.py
    │   │   matcher.py
    │   │   confidence.py
    │   │   face_engine.py
    │
    │   ├── db/
    │   │   database.py
    │
    │   ├── utils/
    │   │   image_loader.py
    │   │   visualization.py   # Debug only
    │
    │   └── config/
    │       settings.py
    │
    ├── app.py                # CLI runner
    ├── requirements.txt
    └── Dockerfile (optional)

------------------------------------------------------------------------

# 🧰 Technology Stack

  Component      Version
  -------------- ---------
  Python         3.10
  InsightFace    0.7+
  ONNX Runtime   Latest
  ChromaDB       Latest
  OpenCV         Latest
  NumPy          Latest

------------------------------------------------------------------------

# ⚙️ Installation

## 1️⃣ Create Virtual Environment

``` bash
python -m venv venv
```

Activate:

### Windows

    venv\Scripts\activate

### enrollment 
python app.py --mode inspect
python app.py --mode enroll --dataset dataset
### Test 

python app.py --mode recognize --image test_images/random_person.jpg

## 2️⃣ Install Dependencies

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

# ⚠️ Windows Build Issue (InsightFace)

InsightFace may fail to build due to missing C++ tools.

## ✅ Fix

Install **Microsoft Visual C++ Build Tools** and select:

-   C++ build tools\
-   MSVC v14.x\
-   Windows SDK

Then run:

``` bash
pip install insightface --prefer-binary
```

### ⭐ Recommended Alternative

Use **Linux / WSL / Docker** for fewer build issues.

------------------------------------------------------------------------

# 🧪 How to Run the System

⚠️ Always run commands from the **project root**.

------------------------------------------------------------------------

## 🔥 Step 1 --- Prepare Dataset

Structure MUST be:

    dataset/
       amit/
            img1.jpg
            img2.jpg

       rohit/
            img1.jpg

👉 Folder name = `user_id`.

------------------------------------------------------------------------

## 🔥 Step 2 --- Batch Enrollment

``` bash
python app.py --mode enroll --dataset dataset
```

Expected:

    ✅ Stored XX embeddings.

Embeddings will be saved inside:

    vector_db/

------------------------------------------------------------------------

## 🔥 Step 3 --- Recognition Test

Use images NOT present in the dataset.

``` bash
python app.py --mode recognize --image test_images/test1.jpg
```
## for api
uvicorn src.api.main:app --reload


Example output:

    user_id    confidence    distance    decision
    -------------------------------------------
    amit       0.93          0.31        MATCH

------------------------------------------------------------------------

## 🔥 Step 4 --- Debug Output Image

When a match occurs, an annotated image is saved automatically:

    output/match_170000.jpg

Contains:

-   Bounding box\
-   Name\
-   Confidence

Useful for audits and debugging.

------------------------------------------------------------------------

# 🧪 Enrollment Strategy (CRITICAL FOR ACCURACY)

## Minimum Required

-   Front face\
-   Left profile\
-   Right profile

## Recommended

-   With glasses\
-   Smile\
-   Different lighting\
-   Slight angle

👉 **Best Practice: 5--10 images per user**

More embeddings → stronger identity cluster.

------------------------------------------------------------------------

# ⚖️ Matching Logic

  Distance        Decision
  --------------- -----------
  `< 0.35`        MATCH
  `0.35 – 0.45`   UNCERTAIN
  `> 0.45`        UNKNOWN

The system prioritizes rejecting unknown faces over false matches.

------------------------------------------------------------------------

# 🧪 Proper Testing Strategy

## ✅ Positive Test

Enroll a user → test with a NEW photo.

Expected distance:

    0.20 – 0.40

------------------------------------------------------------------------

## ❌ Negative Test

Use a person NOT in DB.

Expected:

    UNKNOWN

⚠️ Never test using enrollment images --- it creates fake accuracy.

------------------------------------------------------------------------

# 📈 Scalability

  Users     Status
  --------- -------------------------------------
  ≤100      Excellent
  \~500     Stable
  \~1,000   Production-ready
  \>5,000   Consider FAISS / distributed search

------------------------------------------------------------------------

# 🛑 Current Limitations

-   No anti-spoofing (photo attacks possible)\
-   Extreme face angles reduce accuracy\
-   Masked faces may fail\
-   Video pipeline not implemented

------------------------------------------------------------------------

# 🔮 Future Improvements

-   FastAPI inference service\
-   GPU acceleration\
-   FAISS migration\
-   Anti-spoofing / liveness detection\
-   Distributed vector search\
-   Cloud deployment\
-   Audit logging

------------------------------------------------------------------------

# 🎯 Project Outcomes

-   ✔ Clean ML architecture\
-   ✔ Stateless recognition\
-   ✔ Batch enrollment pipeline\
-   ✔ Edge-case handling\
-   ✔ Production-style codebase\
-   ✔ Review-ready documentation

------------------------------------------------------------------------


