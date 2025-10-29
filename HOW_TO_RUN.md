# 🚀 HOW TO RUN - Kumis Virtual Try-On System

**Panduan Lengkap: Dari Dataset Collection hingga Running Aplikasi**

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Step 1: Setup Environment](#step-1-setup-environment)
3. [Step 2: Collect Dataset](#step-2-collect-dataset)
4. [Step 3: Train Model](#step-3-train-model)
5. [Step 4: Test Model](#step-4-test-model)
6. [Step 5: Run UDP Server](#step-5-run-udp-server)
7. [Step 6: Run Godot Client](#step-6-run-godot-client)
8. [Troubleshooting](#troubleshooting)
9. [Tips & Best Practices](#tips--best-practices)

---

## Prerequisites

### Hardware Requirements:
- **CPU:** Intel Core i5 atau lebih tinggi
- **RAM:** Minimal 8 GB
- **Webcam:** Built-in atau USB webcam
- **Storage:** 5 GB free space

### Software Requirements:
- **Python:** 3.8 - 3.12
- **Godot Engine:** 4.x
- **Git:** (optional) untuk clone repository
- **Internet:** Untuk download dataset

### System:
- Windows 10/11 (panduan ini untuk Windows)
- Linux/Mac (adjust path dengan `/` instead of `\`)

---

## Step 1: Setup Environment

### 1.1 Install Python

**Check Python version:**
```powershell
python --version
# Output: Python 3.10.x atau 3.11.x (recommended)
```

**Jika belum install:**
1. Download dari https://www.python.org/downloads/
2. Install dengan checklist "Add Python to PATH"
3. Restart terminal

---

### 1.2 Install Dependencies

```powershell
# Navigate ke project folder
cd "Kumis_Server"

# Install all requirements
pip install -r requirements.txt
```

**Expected packages installed:**
- opencv-python (4.8.0+)
- scikit-learn (1.3.0+)
- numpy (1.24.0+)
- joblib (1.3.0+)
- matplotlib (3.7.0+)
- seaborn (0.12.0+)
- Pillow (10.0.0+)
- tqdm (4.65.0+)
- requests (2.31.0+)

**Verify installation:**
```powershell
python -c "import cv2, sklearn, numpy, joblib, matplotlib, tqdm, requests; print('✅ All packages installed!')"
```

---

### 1.3 Install Kaggle CLI (Optional - untuk download dataset)

```powershell
pip install kaggle
```

**Setup Kaggle API:**
1. Buka https://www.kaggle.com/settings
2. Scroll ke "API" section
3. Click "Create New API Token"
4. Download `kaggle.json`
5. Copy ke folder:
   ```powershell
   mkdir "$env:USERPROFILE\.kaggle"
   Copy-Item "Downloads\kaggle.json" "$env:USERPROFILE\.kaggle\kaggle.json"
   ```

---

### 1.4 Install Godot Engine

**Download:**
1. Buka https://godotengine.org/download
2. Download **Godot 4.x** (Standard version)
3. Extract ke folder (misal: `C:\Godot\`)
4. (Optional) Add ke PATH

**Verify:**
- Double-click `Godot_v4.x.exe`
- Should open Godot Project Manager

---

## Step 2: Collect Dataset

Dataset diperlukan untuk training SVM classifier:
- **Faces:** 500+ gambar wajah
- **Non-faces:** 1000+ gambar bukan wajah

### 2.1 Create Data Folders

```powershell
cd Kumis_Server

# Create directories
New-Item -ItemType Directory -Force -Path "data\faces"
New-Item -ItemType Directory -Force -Path "data\non_faces"
```

---

### 2.2 Method A: Unsplash (RECOMMENDED - Fastest!)

**Download non-faces dari Unsplash (10 menit):**

```powershell
# Run download script
python download_unsplash.py

# Script akan download:
# - 200 building images
# - 200 nature images
# - 100 car images
# - 150 food images
# - 100 animal images
# - 100 furniture images
# - 100 texture images
# - 50 plant images
# Total: 1000 images
```

**Progress output:**
```
🚀 Start download? (y/n): y

📥 Downloading 200 images: 'building' (640x480)
building    : 100%|████████████| 200/200 [2:30<00:00, 1.33img/s]
✅ 200/200 images successfully downloaded for 'building'
...

🎉 DOWNLOAD COMPLETE!
✅ Successfully downloaded: 1000/1000 images
⏱️ Total time: 600.5 seconds (10.0 minutes)
```

---

**Download faces dari Kaggle (5 menit):**

```powershell
# Download LFW dataset
kaggle datasets download -d jessicali9530/lfw-dataset

# Extract
Expand-Archive lfw-dataset.zip -DestinationPath temp_lfw

# Copy 500 random faces
Get-ChildItem -Path temp_lfw -Recurse -Filter *.jpg | Get-Random -Count 500 | ForEach-Object { Copy-Item $_.FullName -Destination "data\faces\$($_.Name)" }

# Cleanup
Remove-Item -Recurse temp_lfw
Remove-Item lfw-dataset.zip
```

---

### 2.3 Method B: Webcam Collection (No Internet)

**Collect faces:**
```powershell
python collect_dataset.py --webcam --output data/faces --count 500 --type positive --camera 0
```

**Instructions:**
- Webcam akan terbuka
- Posisikan wajah di depan camera
- Press **SPACE** untuk capture (atau auto-capture)
- Gerakkan wajah (kiri, kanan, atas, bawah)
- Ubah ekspresi (senyum, serius, dll)
- Pakai/lepas kacamata untuk variasi
- Continue sampai 500 gambar

**Collect non-faces:**
```powershell
python collect_dataset.py --webcam --output data/non_faces --count 1000 --type negative --camera 0
```

**Instructions:**
- Tunjukkan objek ke webcam (BUKAN wajah!)
- Objek: buku, mug, keyboard, dinding, lantai, tanaman, dll
- Press **SPACE** untuk capture
- Variasi objek dan angle
- Continue sampai 1000 gambar

---

### 2.4 Validate Dataset

```powershell
# Check count
Write-Host "Faces: " (Get-ChildItem data\faces | Measure-Object).Count
Write-Host "Non-faces: " (Get-ChildItem data\non_faces | Measure-Object).Count
```

**Expected output:**
```
Faces:  500
Non-faces:  1000
```

**✅ Dataset ready jika total ≥ 1500 images!**

---

## Step 3: Train Model

### 3.1 Start Training

```powershell
python app.py train --pos_dir data/faces --neg_dir data/non_faces --k 256 --svm linear --C 1.0
```

**Parameters explained:**
- `--pos_dir`: Folder berisi face images
- `--neg_dir`: Folder berisi non-face images
- `--k`: Number of visual words (clusters) untuk BoVW (default: 256)
- `--svm`: SVM kernel (`linear` atau `rbf`)
- `--C`: SVM regularization parameter (default: 1.0)

---

### 3.2 Training Process

**Output yang diharapkan:**

```
=== SVM Face Detector Training ===

📂 Loading dataset...
  Positive samples: 500 (from data/faces)
  Negative samples: 1000 (from data/non_faces)
  Total samples: 1500

📊 Splitting dataset...
  Training set: 1200 samples (80%)
  Test set: 300 samples (20%)

🎨 Extracting ORB features...
Training samples: 100%|████████████| 1200/1200 [01:30<00:00, 13.33it/s]

🔬 Building BoVW codebook (k=256)...
  K-means clustering...
  ✅ Codebook created: 256 clusters

📦 Encoding features with BoVW...
Training samples: 100%|████████████| 1200/1200 [00:45<00:00, 26.67it/s]
Test samples: 100%|████████████| 300/300 [00:11<00:00, 26.67it/s]

🤖 Training SVM (linear kernel, C=1.0)...
  ✅ SVM trained!

📊 Evaluating on test set...
  Accuracy: 87.4%
  Precision: 88.2%
  Recall: 86.5%
  F1 Score: 87.3%
  ROC AUC: 92.1%

💾 Saving models...
  ✅ models/codebook.pkl
  ✅ models/svm.pkl
  ✅ models/scaler.pkl
  ✅ models/config.json

✅ Training complete!
  Time elapsed: 12 minutes 35 seconds
```

**Training time:** 10-20 minutes (depending on CPU)

---

### 3.3 Check Trained Models

```powershell
# Verify models exist
Get-ChildItem models

# Expected output:
# codebook.pkl    (8-10 MB)
# svm.pkl         (2-3 MB)
# scaler.pkl      (~1 MB)
# config.json     (few KB)
```

---

## Step 4: Test Model

### 4.1 Evaluate Model Performance

```powershell
python app.py eval --report reports/metrics.json --cm reports/confusion_matrix.png --pr reports/pr_curve.png --roc reports/roc_curve.png
```

**Output:**
```
=== Model Evaluation ===

📦 Loading models...
  ✅ Models loaded successfully

📂 Loading test data...
  Test samples: 300

📊 Evaluating model...
  Accuracy: 87.4%
  Precision: 88.2%
  Recall: 86.5%
  F1 Score: 87.3%
  ROC AUC: 92.1%

📈 Confusion Matrix:
           Predicted
           Non-Face  Face
Actual
Non-Face     195      5
Face          33     67

📊 Generating plots...
  ✅ Confusion matrix saved: reports/confusion_matrix.png
  ✅ Precision-Recall curve: reports/pr_curve.png
  ✅ ROC curve: reports/roc_curve.png

💾 Report saved: reports/metrics.json
```

**Check reports:**
```powershell
explorer reports
```

---

### 4.2 Test with Single Image

```powershell
python app.py infer --image path/to/test_image.jpg --output output.jpg
```

**Example:**
```powershell
# Download test image atau gunakan dari webcam
python app.py infer --image test_face.jpg --output detected.jpg
```

**Output:**
```
📦 Loading models...
  ✅ Models loaded

🖼️ Loading image: test_face.jpg (640x480)

🔍 Detecting faces...
  ✅ Detected 1 face(s)
  Face 1: (150, 100, 400, 350) - Confidence: 0.92

💾 Saving result: detected.jpg
  ✅ Image saved!
```

---

### 4.3 Test with Webcam (Standalone)

```powershell
python app.py webcam
```

**What happens:**
- Webcam opens
- Real-time face detection
- Green box around detected faces
- FPS counter
- Press **Q** to quit

**Expected FPS:** 15-20 FPS @ 640x480

---

## Step 5: Run UDP Server

### 5.1 Start Server

```powershell
python udp_kumis_server.py
```

**Optional parameters:**
```powershell
python udp_kumis_server.py --camera 0 --width 640 --height 480 --fps 15 --models models
```

---

### 5.2 Server Output

```
==================================================
🥸 Kumis Try-On Server (SVM+ORB)
==================================================

📦 Loading models from models...
  ✅ Models loaded successfully!
     - Codebook: 256 clusters
     - SVM: linear kernel

🎥 Initializing camera 0...
  ✅ Camera initialized: 640x480 @ 30FPS

🚀 Starting UDP Server...
  Server: 127.0.0.1:8888
  Client: 127.0.0.1:9999

📡 Waiting for connections...
  Send 'CONNECT' from client to register
  Send 'SET_KUMIS <filename>' to change kumis

  Press Ctrl+C to stop
```

**Server is ready!** ✅

**Leave this terminal running!** Don't close it.

---

### 5.3 Verify Server is Running

**In another terminal:**
```powershell
netstat -an | findstr "8888"
# Should show: TCP  127.0.0.1:8888  LISTENING
```

---

## Step 6: Run Godot Client

### 6.1 Open Godot Project

1. **Launch Godot Engine**
   - Double-click `Godot_v4.x.exe`

2. **Import Project**
   - Click "Import" button
   - Navigate to: `KumisNusantara_App/project.godot`
   - Click "Import & Edit"

3. **Wait for import** (first time: 1-2 minutes)

---

### 6.2 Run Application

**Method 1: Press F5**
```
F5 key → Run project
```

**Method 2: Click Play Button**
```
Click ▶️ button at top-right
```

---

### 6.3 Application Flow

#### **Screen 1: Main Menu**
```
┌────────────────────────────────┐
│                                │
│     KUMIS TRY-ON SYSTEM        │
│                                │
│      [🥸 Try On Kumis]         │
│                                │
│          [Quit]                │
│                                │
└────────────────────────────────┘
```

**Action:** Click **"Try On Kumis"** button

---

#### **Screen 2: Kumis Selection**
```
┌────────────────────────────────────────┐
│  🥸 Pilih Gaya Kumis (13 Styles!)      │
├────────────────────────────────────────┤
│ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐  │
│ │ [🥸] │ │ [🥸] │ │ [🥸] │ │ [🥸] │  │
│ │Class │ │Style1│ │Style2│ │Style3│  │
│ └──────┘ └──────┘ └──────┘ └──────┘  │
│ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐  │
│ │ [🥸] │ │ [🥸] │ │ [🥸] │ │ [🥸] │  │
│ │Style4│ │Style5│ │Style6│ │Style7│  │
│ └──────┘ └──────┘ └──────┘ └──────┘  │
│           ... (13 total) ...          │
├────────────────────────────────────────┤
│ [← Kembali]      [✅ Pilih Kumis →]   │
└────────────────────────────────────────┘
```

**Actions:**
1. Click any kumis style button → Button turns **green**
2. Click **"✅ Pilih Kumis"** button

**Server console shows:**
```
  ✅ Client connected: ('127.0.0.1', 54321)
  🎭 Kumis set to: kumis_5.png
```

---

#### **Screen 3: Webcam View**
```
┌────────────────────────────────────────┐
│ Status: ✅ Connected       FPS: 15     │
├────────────────────────────────────────┤
│                                        │
│         [VIDEO STREAM]                 │
│      Face detection + kumis            │
│                                        │
├────────────────────────────────────────┤
│ [← Kembali]  [Sembunyikan Kumis]      │
└────────────────────────────────────────┘
```

**Features:**
- ✅ Real-time video stream
- ✅ Face detection (green box)
- ✅ Kumis overlay on detected face
- ✅ FPS counter
- ✅ Toggle button (hide/show kumis)
- ✅ Back button (change style)

**Actions:**
- **Sembunyikan Kumis:** Hide overlay (face detection still works)
- **Tampilkan Kumis:** Show overlay again
- **← Kembali:** Return to selection screen

**Server console shows:**
```
  📤 Broadcasting frames to 1 client(s) at 15 FPS
  🎭 Kumis overlay: OFF
  🎭 Kumis overlay: ON
```

---

### 6.4 Test Different Kumis Styles

1. Click **"← Kembali"**
2. Select different style (e.g., Style 10)
3. Click **"Pilih Kumis"**
4. Webcam shows new style
5. Repeat for all 13 styles!

---

## Troubleshooting

### Problem 1: "No module named 'cv2'"
**Solution:**
```powershell
pip install opencv-python
```

---

### Problem 2: "Models not found"
**Solution:**
```powershell
# Train model first!
python app.py train --pos_dir data/faces --neg_dir data/non_faces
```

---

### Problem 3: "Cannot open camera"
**Solutions:**
1. Check if webcam is being used by another app (close it)
2. Try different camera ID:
   ```powershell
   python udp_kumis_server.py --camera 1
   ```
3. Check webcam permissions (Windows Settings → Privacy)

---

### Problem 4: "Connection timeout" (Godot)
**Solutions:**
1. Make sure UDP server is running
2. Check firewall:
   ```powershell
   # Allow Python through firewall
   netsh advfirewall firewall add rule name="Python UDP" dir=in action=allow protocol=UDP localport=8888
   netsh advfirewall firewall add rule name="Python UDP" dir=in action=allow protocol=UDP localport=9999
   ```
3. Check ports not used by other apps:
   ```powershell
   netstat -an | findstr "8888"
   netstat -an | findstr "9999"
   ```

---

### Problem 5: Low FPS (<10 FPS)
**Solutions:**
1. Reduce resolution:
   ```powershell
   python udp_kumis_server.py --width 320 --height 240
   ```
2. Close other applications
3. Use faster SVM kernel (linear instead of rbf)
4. Reduce ORB features:
   - Edit `pipelines/features.py` → `nfeatures=300` instead of 500

---

### Problem 6: Face not detected
**Solutions:**
1. Check lighting (too dark?)
2. Face directly towards camera
3. Check model accuracy (Step 4.1)
4. If accuracy low, collect more diverse dataset
5. Adjust confidence threshold:
   - Edit `udp_kumis_server.py` → `confidence_threshold=0.3` (line ~89)

---

### Problem 7: Kumis position wrong
**Explanation:** Kumis positioned based on eye detection
**Solutions:**
1. Make sure eyes are visible (not covered)
2. Face camera directly (not extreme angle)
3. Adjust kumis scale in `pipelines/overlay.py`:
   ```python
   kumis_width = int(face_width * 0.7)  # Try 0.6 or 0.8
   ```

---

### Problem 8: Godot "Scene not found"
**Solution:**
```powershell
# Verify files exist
Test-Path "KumisNusantara_App\Scenes\KumisNusantara\KumisSelectionScene.tscn"
# Should return: True

# If False, check file path or re-create scene
```

---

### Problem 9: Dataset download slow
**Solutions:**
1. For Unsplash:
   - Check internet connection
   - Increase delay in `download_unsplash.py` (line ~76):
     ```python
     time.sleep(1.0)  # Instead of 0.5
     ```
2. For Kaggle:
   - Use smaller dataset (CelebA subset instead of full)
3. Use webcam collection instead (no internet needed)

---

### Problem 10: Training accuracy low (<80%)
**Solutions:**
1. **Collect more data:**
   - 1000+ faces (instead of 500)
   - 2000+ non-faces (instead of 1000)

2. **Improve data quality:**
   - Remove corrupted images
   - Check no faces in non_faces folder
   - More diverse data (different lighting, angles, people)

3. **Tune hyperparameters:**
   ```powershell
   # Try RBF kernel
   python app.py train --svm rbf --C 10.0
   
   # Try more clusters
   python app.py train --k 512
   ```

4. **Use hyperparameter search:**
   - Edit `pipelines/train.py` → Enable grid search
   - Will take longer but find best params

---

## Tips & Best Practices

### Dataset Collection:
1. ✅ **Quality over quantity** - 500 good faces > 1000 bad faces
2. ✅ **Diversity** - Different ages, genders, ethnicities, lighting
3. ✅ **Clean non-faces** - ABSOLUTELY no faces in non_faces folder
4. ✅ **Validation** - Check images manually before training

### Training:
1. ✅ **Start with linear SVM** - Faster, often good enough
2. ✅ **Monitor metrics** - Accuracy >85% is good for this task
3. ✅ **Save checkpoints** - Models saved automatically
4. ✅ **Compare kernels** - Try both linear and RBF

### Testing:
1. ✅ **Test standalone first** - Use `app.py webcam` before UDP
2. ✅ **Check reports** - Look at confusion matrix, PR curve
3. ✅ **Test edge cases** - Profile faces, bad lighting, occlusions

### Deployment:
1. ✅ **Run server first** - Always start UDP server before Godot
2. ✅ **Check logs** - Monitor server console for errors
3. ✅ **Test network** - Verify ports not blocked
4. ✅ **Optimize performance** - Adjust resolution/FPS as needed

### Development:
1. ✅ **Version control** - Use git for code changes
2. ✅ **Document changes** - Update README when adding features
3. ✅ **Test incrementally** - Test each step before moving to next
4. ✅ **Backup models** - Save trained models to cloud/external drive

---

## Quick Reference Commands

### Dataset:
```powershell
# Unsplash (non-faces)
python download_unsplash.py

# Kaggle (faces)
kaggle datasets download -d jessicali9530/lfw-dataset
```

### Training:
```powershell
# Basic training
python app.py train --pos_dir data/faces --neg_dir data/non_faces

# Advanced training
python app.py train --pos_dir data/faces --neg_dir data/non_faces --k 512 --svm rbf --C 10.0
```

### Testing:
```powershell
# Evaluate model
python app.py eval

# Test single image
python app.py infer --image test.jpg --output result.jpg

# Test webcam
python app.py webcam
```

### Deployment:
```powershell
# Start UDP server
python udp_kumis_server.py

# Start with custom settings
python udp_kumis_server.py --camera 0 --width 640 --height 480 --fps 20
```

### Godot:
```
F5 → Run project
Ctrl+Q → Quit
```

---

## Success Checklist

Before running full application, verify:

- [x] Python 3.8+ installed
- [x] All pip packages installed
- [x] Godot 4.x installed
- [x] Dataset collected (500+ faces, 1000+ non-faces)
- [x] Model trained (models/*.pkl exist)
- [x] Model accuracy >85%
- [x] Webcam working
- [x] UDP server starts without errors
- [x] Godot project imports successfully
- [x] All 13 kumis PNG files in both locations

---

## Estimated Time

| Step | Time |
|------|------|
| Setup environment | 10-15 min |
| Collect dataset | 15-20 min |
| Train model | 10-20 min |
| Test model | 5 min |
| Run application | Instant |
| **TOTAL** | **45-60 min** |

---

## Next Steps

After successful run:
1. 📸 Test with different kumis styles
2. 🎨 Add more kumis PNG files
3. 📊 Improve model (collect more data, tune params)
4. 🚀 Deploy on different machines
5. 📝 Customize UI (Godot scenes)

---

**🎉 Happy Mustache Try-On! 🥸**

*Last updated: October 28, 2025*
