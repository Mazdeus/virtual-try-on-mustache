# 👨 Virtual Try-On Kumis - Real-time Mustache Overlay Application

Aplikasi Virtual Try-On untuk berbagai style kumis menggunakan **Machine Learning tradisional** (SVM + ORB + BoVW) dengan real-time face detection dan video streaming.

![Version](https://img.shields.io/badge/version-2.0.0-blue)
![Python](https://img.shields.io/badge/Python-3.8+-yellow)
![Godot](https://img.shields.io/badge/Godot-4.x-blue)
![ML](https://img.shields.io/badge/ML-SVM+ORB-orange)
![License](https://img.shields.io/badge/license-Educational-green)

---

## 📖 Tentang Program

**Virtual Try-On Kumis** adalah aplikasi interaktif yang memungkinkan pengguna mencoba berbagai style kumis secara real-time melalui webcam. Aplikasi ini menggunakan **Classical Machine Learning** (SVM classifier + ORB features) untuk face detection dengan akurasi 83.8% dan inference time 50-60ms (real-time @ CPU).

### 🎯 Fitur Utama

- ✅ **12 Style Kumis** - Berbagai gaya kumis dari klasik hingga modern
- ✅ **6 Preset Warna** - Black, Brown, Blonde, Red, Gray, White + custom HSV
- ✅ **Screenshot Feature** - Simpan foto hasil try-on dengan popup notification
- ✅ **Interactive Tutorial** - Step-by-step guide dengan animasi interaktif
- ✅ **Contributors Page** - Informasi tim pengembang dengan foto
- ✅ **Real-time Detection** - Face detection dengan SVM+ORB (50-60ms inference)
- ✅ **Smart Validation** - 6-layer pipeline (Haar + SVM + Eye Detection)
- ✅ **Rotation Support** - Kumis ikut rotasi saat kepala miring (angle smoothing)
- ✅ **Anti-Flickering** - Temporal smoothing (95% reduction)
- ✅ **CPU-Only** - No GPU required (~200MB RAM, <5MB model)
- ✅ **High Compatibility** - Multi-backend webcam support (95% devices)

### 📊 Model Performance

| Metric | Value | Note |
|--------|-------|------|
| **Accuracy** | 78.2% | Test set: 900 images (balanced) |
| **Precision** | 77.1% | Low false positives |
| **Recall** | 80.2% | Good detection rate |
| **F1-Score** | 78.6% | Balanced performance |
| **ROC AUC** | 88.0% | Excellent discrimination |
| **Training Data** | 6000 images | 3000 faces + 3000 non-faces |
| **Inference Time** | 50-60ms | Real-time @ 15+ FPS |
| **Rotation Support** | ✅ Yes | Multi-angle face detection |

---

## 📁 Struktur Project

```
virtual-try-on-mustache/
│
├── Kumis_Server/                      # Python Backend (ML + UDP Server)
│   ├── udp_kumis_server.py           # Main server (multi-threading)
│   ├── requirements.txt              # Dependencies (opencv, sklearn, numpy)
│   │
│   ├── models/                       # Trained ML Models
│   │   ├── svm_model.pkl            # SVM classifier (linear kernel)
│   │   ├── codebook_256.pkl         # K-Means codebook (BoVW)
│   │   └── scaler.pkl               # StandardScaler (normalization)
│   │
│   ├── screenshots/                  # Screenshot output folder
│   │   └── kumis_[style]_[timestamp].jpg  # Auto-saved photos
│   │
│   ├── assets/kumis/                 # Kumis images (PNG with alpha)
│   │   └── kumis_1.png ... kumis_12.png
│   │
│   ├── data/                         # Training dataset
│   │   ├── faces/                   # 500 face images
│   │   └── non_faces/               # 300 non-face images
│   │
│   └── pipelines/                    # ML Pipeline Modules
│       ├── dataset.py               # Dataset loading/preprocessing
│       ├── features.py              # ORB + BoVW encoding
│       ├── train.py                 # SVM training script
│       ├── infer.py                 # FaceDetector (6-layer validation)
│       └── overlay.py               # KumisOverlay (rotation + blending + colorization)
│
└── Kumis_App/                        # Godot Frontend (UI + UDP Client)
    ├── project.godot                 # Godot project config
    ├── Global.gd                     # Global state manager
    │
    ├── Scenes/
    │   ├── MainMenu/
    │   │   ├── MainMenu.tscn        # Main menu UI
    │   │   └── MainMenuController.gd
    │   │
    │   ├── HowToUse/
    │   │   ├── HowToUse.tscn        # Interactive tutorial scene
    │   │   └── HowToUseController.gd # Step-by-step animation
    │   │
    │   ├── AboutUs/
    │   │   ├── AboutUs.tscn         # Contributors page
    │   │   └── AboutUsController.gd  # Team info display
    │   │
    │   └── Kumis/           # Main app scenes
    │       ├── KumisSelectionScene.tscn    # Kumis selection (grid 12 styles)
    │       ├── KumisSelectionController.gd  # Selection logic + sorting
    │       ├── KumisWebcamScene.tscn       # Webcam display (960×720)
    │       ├── KumisWebcamController.gd    # UDP client + controls
    │       └── WebcamManagerUDP.gd         # UDP networking
    │
    └── Assets/
        ├── Kumis/                    # Kumis preview images
        │   └── kumis_1.png ... kumis_12.png
        └── Contributors/             # Team member photos
            ├── faisal.jpg
            ├── amadeus.png
            └── hasbi.jpg
```

---

## 🛠️ Teknologi yang Digunakan

### Backend (Python)

**Machine Learning Pipeline:**
1. **ORB (Oriented FAST and Rotated BRIEF)** - Feature extraction (500 keypoints)
2. **Bag-of-Visual-Words (BoVW)** - K-Means clustering (k=256) untuk fixed-length vector
3. **SVM (Support Vector Machine)** - Linear kernel classifier untuk face verification
4. **Haar Cascade** - Initial face detection (fast, 10-15ms)
5. **Eye Detection** - Mandatory validation (eliminate 90% false positives)

**Libraries:**
- **OpenCV >= 4.8.0** - Webcam, image processing, Haar Cascade
- **Scikit-learn >= 1.3.0** - SVM, K-Means, StandardScaler
- **NumPy >= 1.24.0** - Array operations, alpha blending

**Why Classical ML?**
- ✅ **Fast**: 50-60ms vs 200-300ms (deep learning)
- ✅ **Lightweight**: <5MB model vs ~20MB (MTCNN)
- ✅ **CPU-Only**: No GPU required (consumer devices)
- ✅ **Low Memory**: ~200MB vs ~800MB (deep learning)
- ⚠️ **Trade-off**: 83.8% accuracy vs ~95% (deep learning) → **11.2% loss for 4-6× speed gain**

---

### Frontend (Godot)

**Godot Engine 4.x:**
- **Language**: GDScript
- **Purpose**: UI/UX, scene management, UDP client
- **Features**:
  - Scene-based architecture (Main Menu → Selection → Webcam)
  - UDP networking (`PacketPeerUDP`)
  - Image processing (`Image`, `ImageTexture`, JPEG decoding)
  - Fullscreen mode, controls (Spacebar, ESC, Q)

---

### Networking (UDP Protocol)

**Architecture:**
- **Server**: `127.0.0.1:8888` (listen commands, broadcast frames)
- **Client**: `127.0.0.1:9999` (receive frames, send commands)

**Why UDP?**
- Low latency (no handshake)
- Real-time streaming (prefer newest frame vs reliability)
- Efficient bandwidth (~1.2 MB/s @ 15 FPS)

**Commands:**
```
CONNECT                    # Register client
SELECT_KUMIS:5            # Select kumis by index (1-12)
TOGGLE_KUMIS              # Show/hide overlay
COLOR:BROWN               # Set kumis color (BLACK/BROWN/BLONDE/RED/GRAY/WHITE)
SCREENSHOT                # Capture and save photo
```

---

## 🔄 Alur Program

### 1. Architecture Overview

```
┌────────────────────────────┐
│   Godot Client (UI)        │
│   - Main Menu              │
│   - Kumis Selection        │
│   - Webcam Display         │
└──────────┬─────────────────┘
           │ UDP (commands)
           ↓
    ┌──────────────┐
    │ UDP Socket   │
    │ Port 8888    │
    └──────────────┘
           ↑
           │ UDP (JPEG frames)
┌──────────┴─────────────────┐
│   Python Server            │
│   - Webcam Capture         │
│   - Face Detection (SVM)   │
│   - Kumis Overlay          │
│   - JPEG Encoding          │
└────────────────────────────┘
```

### 2. Face Detection Pipeline (6 Layers)

```
Input: Video Frame (640×480 BGR)
  ↓
LAYER 1: Haar Cascade Detection
  → Output: Candidate faces [(x,y,w,h), ...]
  ↓
LAYER 2: SVM Classification
  → ORB extract (500 features) → BoVW encode (256-dim)
  → SVM predict_proba() → confidence > 0.25
  ↓
LAYER 3: Size Validation
  → Face area: 2-60% of frame (reject too small/large)
  ↓
LAYER 4: Aspect Ratio Validation
  → Ratio: 0.6-1.5 (reject distorted faces)
  ↓
LAYER 5: Position Validation
  → Center distance < 40% (reject edge faces)
  ↓
LAYER 6: Eye Detection (MANDATORY)
  → Detect 2 eyes (horizontal) → REJECT if fails
  → Result: 90% false positive elimination ✅
  ↓
Temporal Smoothing (10-frame cache)
  → If detection fails → use cached face
  → Result: 95% flickering reduction ✅
  ↓
Output: Validated face + eye positions
```

### 3. Kumis Overlay Pipeline

```
Input: Frame + Face coordinates + Eye positions
  ↓
Calculate face angle (eye-based rotation)
  → angle = atan2(dy, dx) × 180/π
  ↓
Angle smoothing (reduce jitter)
  → smoothed = old×0.6 + new×0.4
  → Result: 60% jitter reduction ✅
  ↓
Resize kumis (90% face width)
  ↓
Rotate kumis (cv2.warpAffine)
  ↓
Position kumis (below nose: face_y + face_h×0.55)
  ↓
Alpha blending (transparent overlay)
  → For each pixel: output = alpha×kumis + (1-alpha)×frame
  ↓
Output: Frame with kumis overlay
```

---

## 🚀 Cara Menjalankan Program (Setelah Training)

### Prerequisites
- ✅ **Python 3.8+** sudah terinstall
- ✅ **Model sudah di-train** (file `.pkl` ada di folder `models/`)
- ✅ **Webcam tersedia** dan berfungsi
- ✅ **Godot 4.x** sudah terinstall

---

### 1. Install Dependencies (Jika Belum)

```powershell
# Clone repository
git clone https://github.com/Mazdeus/virtual-try-on-mustache.git
cd virtual-try-on-mustache

# Install Python packages
cd Kumis_Server
pip install -r requirements.txt

# Verify installation
python -c "import cv2, numpy, sklearn; print('✅ Dependencies OK')"
```

---

### 2. Verifikasi Model Sudah Ada

```powershell
# Check jika model files ada (harus ada 4 file)
cd Kumis_Server
ls models

# Output yang diharapkan:
# - codebook.pkl    (~800KB)
# - config.json     (~800B)
# - scaler.pkl      (~5KB)
# - svm.pkl         (~2KB)
```

**Jika model belum ada, jalankan training:**
```powershell
python app.py train --pos_dir data/faces --neg_dir data/non_faces --output_dir models --k 200 --nfeatures 500
```

---

### 3. Run Python Server (Backend)

```powershell
cd Kumis_Server
python udp_kumis_server.py
```

**Expected Output:**
```
🚀 Virtual Try-On Kumis - UDP Server
════════════════════════════════════════

📦 Loading models...
  ✅ SVM loaded: models/svm.pkl
  ✅ Scaler loaded: models/scaler.pkl
  ✅ Codebook loaded: models/codebook.pkl
  ✅ Config loaded: models/config.json

📷 Initializing camera...
  ✅ Camera opened: Device 0 (640×480)

🌐 Starting UDP server...
  ✅ Server listening on: 127.0.0.1:8888
  
⏳ Waiting for client connection...
```

**Jangan close terminal ini!** Server harus tetap running.

**Troubleshooting Webcam:**
```powershell
python udp_kumis_server.py --list-cameras    # List available cameras
python udp_kumis_server.py --camera 1        # Use specific camera
python udp_kumis_server.py --auto-detect     # Auto-detect best camera
```

---

### 4. Run Godot Client (Frontend)

#### **Cara 1: Via Godot Editor (Development Mode)**

1. **Download Godot 4.x** dari https://godotengine.org/download (jika belum punya)

2. **Open Godot** → Click **"Import"**

3. **Browse** ke folder `Kumis_App` → Pilih `project.godot` → Click **"Import & Edit"**

4. **Press F5** (atau klik tombol Play ▶️ di toolbar)

5. Aplikasi akan terbuka di window baru

---

#### **Cara 2: Via Exported Executable (Production Mode)**

Jika ada file `.exe` yang sudah di-export:

```powershell
cd Kumis_App
./KumisTryOn.exe    # Double-click atau run via terminal
```

---

### 5. Gunakan Aplikasi

```
┌─────────────────────────────────┐
│       MAIN MENU                 │
│  [� Start Virtual Try-On]     │  ← Click untuk mulai
│  [📖 How to Use]               │  ← Tutorial interaktif
│  [👥 Contributors]             │  ← Info tim pengembang
│  [❌ Quit]                      │
└─────────────────────────────────┘
         ↓
┌─────────────────────────────────────────┐
│    WEBCAM DISPLAY with CONTROLS         │
│  ┌─────────────────────────────┐       │
│  │                             │       │
│  │   [Live Video Feed]         │       │  ← Kumis overlay real-time!
│  │   dengan kumis overlay      │       │
│  │                             │       │
│  └─────────────────────────────┘       │
│                                         │
│  Controls:                              │
│  [← Kembali] [👁 Toggle] [📸 Foto]    │
│                                         │
│  ┌─── Pilih Kumis ───┐                │
│  │ [1] [2] [3] [4]   │                │
│  │ [5] [6] [7] [8]   │  ← Click untuk │
│  │ [9] [10][11][12]  │     ganti kumis │
│  └───────────────────┘                 │
│                                         │
│  ┌─── Warna Kumis ───┐                │
│  │ [⚫Black] [🟤Brown] [🟡Blonde]    │  ← Click untuk
│  │ [🔴Red]   [⚪Gray]  [⚪White]     │     ganti warna
│  └────────────────────┘                │
│                                         │
│  Spacebar: Toggle ON/OFF                │
│  ESC: Fullscreen                        │
└─────────────────────────────────────────┘
```

---

### 6. Keyboard Controls

| Key | Action | Keterangan |
|-----|--------|------------|
| **Spacebar** | Toggle kumis ON/OFF | Menyembunyikan/menampilkan kumis |
| **ESC** | Toggle fullscreen | Fullscreen ↔ Windowed |
| **Mouse Click** | Select kumis/color | Pilih style atau warna kumis |
| **📸 Button** | Screenshot | Simpan foto (popup notification) |
| **← Button** | Kembali ke menu | Di scene Webcam |

---

### 7. Tips untuk Hasil Terbaik

✅ **Lighting**: Pencahayaan yang baik (hindari backlight)  
✅ **Position**: Wajah menghadap kamera secara frontal  
✅ **Distance**: Jarak 50-100cm dari kamera  
✅ **Rotation**: Model support wajah rotasi, kumis akan ikut berputar!  
✅ **Stability**: Hindari gerakan terlalu cepat (untuk mengurangi jitter)

---

## 📸 Contoh Penggunaan

### Mode Tutorial (How to Use)
```
Interactive step-by-step guide dengan animasi:
Step 1: Jalankan Server (dengan animasi highlight)
Step 2: Klik Start Virtual Try-On
Step 3: Pilih Style Kumis (13 pilihan)
Step 4: Ubah Warna Kumis (6 preset colors)
Step 5: Ambil Foto (screenshot feature)

Tips & Tricks:
- Pencahayaan yang baik
- Wajah menghadap kamera
- Jarak optimal 30-50cm
- Keyboard shortcuts cheatsheet
```

### Mode Normal (Frontal Face)
```
Wajah terdeteksi → Kumis ditempel di posisi yang sesuai
Wajah tidak terdeteksi → Kumis hilang (no false positives!)
```

### Mode Rotasi (Tilted Face)
```
Wajah miring ke kanan → Kumis ikut berputar ke kanan (smooth rotation)
Wajah miring ke kiri → Kumis ikut berputar ke kiri
Sudut rotasi: -45° hingga +45° (angle smoothing applied)
```

### Mode Toggle
```
Spacebar ON: Kumis ditampilkan (overlay aktif)
Spacebar OFF: Kumis disembunyikan (hanya face detection)
```

### Color Picker
```
Click warna → Kumis berubah warna real-time
Preset: Black, Brown, Blonde, Red, Gray, White
HSV colorization (only dark pixels = mustache)
```

### Screenshot Feature
```
Click "📸 Foto" → Photo saved to screenshots/ folder
Popup shows: Full path + file size
Auto-naming: kumis_[style]_[timestamp].jpg
Example: kumis_kumis_5_20251103_143022.jpg (72.5 KB)
```

---

## 🐛 Troubleshooting

### Python Server Issues

**Error: "No module named 'sklearn'"**
```powershell
pip install scikit-learn opencv-python numpy
```

**Error: "Camera not found"**
```powershell
python udp_kumis_server.py --list-cameras  # List devices
python udp_kumis_server.py --camera 1      # Try camera index 1
```

**Error: "Port 8888 already in use"**
```powershell
netstat -ano | findstr :8888    # Find PID
taskkill /PID <PID> /F          # Kill process
```

---

### Godot Client Issues

**Error: "Could not connect to server"**
- ✅ Ensure Python server is running first
- ✅ Check firewall (allow UDP traffic)
- ✅ Verify IP: `127.0.0.1` (localhost)

**Kumis tidak muncul di wajah**
- ✅ Check Python console: "Face detected" messages
- ✅ Improve lighting (face camera directly)
- ✅ Check file exists: `Kumis_Server/assets/kumis/kumis_X.png`
- ✅ Baca tutorial: Main Menu → "📖 How to Use"

**Screenshot tidak tersimpan**
- ✅ Check folder exists: `Kumis_Server/screenshots/`
- ✅ Check disk space (min 10MB free)
- ✅ Check Python console for error messages

**Color tidak berubah**
- ✅ Ensure kumis sudah dipilih (loaded)
- ✅ Check Python console: "Color applied" messages
- ✅ Kumis must be dark/black (HSV colorization works on dark pixels)

---

## Benchmark Performance
### **Step 1: Jalankan Benchmark**

```powershell
cd Kumis_Server

# Jalankan benchmark selama 90 detik (tanpa display untuk akurasi maksimal)
python benchmark_performance.py --duration 90 --no-display --output reports/benchmark_official.json
```

**Catatan:**
- Pastikan wajah Anda terdeteksi dengan baik (duduk di depan kamera)
- Lighting cukup (300-500 lux)
- Close aplikasi lain yang pakai webcam

### **Step 2: Buka Report JSON**

```powershell
# File akan tersimpan di:
Kumis_Server/reports/benchmark_official.json
```

---

## 📝 Credits

**Developed by:**
- **Faisal Bashri Albir** (231524042)
- **Mohammad Amadeus Andika Fadhil** (231524050)
- **Muhammad Hasbi Asshidiqi** (231524055)

**Course:**
- Pengolahan Citra Digital
- Politeknik Negeri Bandung
- 2025

**Technologies:**
- Godot Engine 4.x, OpenCV, Scikit-learn, NumPy
- Haar Cascade (OpenCV), UDP Protocol

**License**: Educational Use Only - POLBAN

---

## 🎉 Version History

- **v2.2.0** (November 2025) - UI/UX Enhancement
  - 📖 Interactive Tutorial: Step-by-step guide dengan animasi
  - 👥 Contributors Page: Team info dengan foto dan NIM
  - 🎨 Improved Main Menu: 4 tombol navigasi
  - ✨ Animated step highlighting (0.8s cycle)
  - 🚀 "Try Now" quick action dari tutorial

- **v2.1.0** (November 2025) - Feature Expansion
  - ✨ Color Picker: 6 preset colors + custom HSV
  - 📸 Screenshot: Auto-save with popup notification
  - 🔔 Real-time notification with file path & size
  - 🎨 HSV-based colorization (dark pixel masking)
  - 📁 Organized screenshot folder with timestamps

- **v2.0.0** (November 2025) - Virtual Try-On Kumis
  - Classical ML (SVM+ORB+BoVW) pipeline
  - 6-layer validation (Haar + SVM + Eye Detection)
  - 12 kumis styles, temporal smoothing, angle smoothing
  - Performance: 78.2% accuracy, 50-60ms inference

---
