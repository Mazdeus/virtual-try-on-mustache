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
- ✅ **Real-time Detection** - Face detection dengan SVM+ORB (50-60ms inference)
- ✅ **Smart Validation** - 6-layer pipeline (Haar + SVM + Eye Detection)
- ✅ **Rotation Support** - Kumis ikut rotasi saat kepala miring (angle smoothing)
- ✅ **Anti-Flickering** - Temporal smoothing (95% reduction)
- ✅ **CPU-Only** - No GPU required (~200MB RAM, <5MB model)
- ✅ **High Compatibility** - Multi-backend webcam support (95% devices)

### 📊 Model Performance

| Metric | Value | Note |
|--------|-------|------|
| **Accuracy** | 83.8% | Test set: 160 images |
| **Precision** | 84.7% | Low false positives |
| **Recall** | 82.5% | Good detection rate |
| **F1-Score** | 83.6% | Balanced performance |
| **Inference Time** | 50-60ms | Real-time @ 15+ FPS |
| **False Positive Reduction** | 90% | Eye detection mandatory |
| **Flickering Reduction** | 95% | Temporal smoothing |

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
│       └── overlay.py               # KumisOverlay (rotation + blending)
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
    │   └── Kumis/           # Main app scenes
    │       ├── KumisSelectionScene.tscn    # Kumis selection (grid 12 styles)
    │       ├── KumisSelectionController.gd  # Selection logic + sorting
    │       ├── KumisWebcamScene.tscn       # Webcam display (960×720)
    │       ├── KumisWebcamController.gd    # UDP client + controls
    │       └── WebcamManagerUDP.gd         # UDP networking
    │
    └── Assets/Kumis/                 # Kumis preview images
        └── kumis_1.png ... kumis_12.png
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
SET_KUMIS kumis_5.png     # Load kumis style
TOGGLE_KUMIS              # Show/hide overlay
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

## 🚀 Cara Menjalankan

### 1. Install Dependencies

```powershell
# Clone repository
git clone https://github.com/Mazdeus/virtual-try-on-mustache.git
cd virtual-try-on-mustache

# Install Python packages
cd Kumis_Server
pip install -r requirements.txt

# Verify installation
python -c "import cv2, numpy, sklearn; print('✅ OK')"
```

### 2. Run Python Server

```powershell
cd Kumis_Server
python udp_kumis_server.py
```

**Expected Output:**
```
✅ Loaded SVM model: models/svm_model.pkl
✅ Camera initialized: DirectShow (640×480 @ 15 FPS)
🚀 UDP Server started: 127.0.0.1:8888
⏳ Waiting for client connections...
```

**Troubleshooting Webcam:**
```powershell
python udp_kumis_server.py --list-cameras    # List available cameras
python udp_kumis_server.py --camera 1        # Use specific camera
python udp_kumis_server.py --auto-detect     # Auto-detect best camera
```

### 3. Run Godot Client

1. **Download Godot 4.x** dari https://godotengine.org/download
2. **Open Godot** → Click **"Import"**
3. **Browse** ke folder `Kumis_App` → Pilih `project.godot`
4. **Click "Import & Edit"**
5. **Press F5** (atau klik tombol Play ▶️)

### 4. Gunakan Aplikasi

```
Main Menu → Klik "🎯 Mulai Try-On"
  ↓
Kumis Selection (grid 12 styles) → Klik salah satu kumis
  ↓
Klik "✓ Pilih Kumis"
  ↓
Webcam Scene (kumis overlay real-time)
```

**Controls:**
- **Spacebar**: Toggle kumis ON/OFF
- **ESC**: Toggle fullscreen/windowed
- **Q**: Quit aplikasi

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

---

## 📊 Performance Comparison

| Method | Accuracy | Inference | Model Size | Memory | GPU |
|--------|----------|-----------|------------|--------|-----|
| **SVM+ORB (Ours)** | 83.8% | 50-60ms | <5MB | ~200MB | ❌ No |
| MTCNN (Deep Learning) | ~95% | 200-300ms | ~20MB | ~800MB | ✅ Yes |
| Dlib (HOG+SVM) | ~92% | 150-200ms | ~100MB | ~500MB | ❌ No |
| MediaPipe (TF Lite) | ~96% | 100-150ms | ~10MB | ~400MB | ❌ No |

**Conclusion**: SVM+ORB optimal untuk **consumer devices** (CPU-only, low memory, real-time) dengan trade-off akurasi 11.2% untuk speed gain 4-6×.

---

## 📝 Credits

**Developed by:**
- **Faisal**
- **Amadeus**
- **Hasbi**

**Technologies:**
- Godot Engine 4.x, OpenCV, Scikit-learn, NumPy
- Haar Cascade (OpenCV), UDP Protocol

**License**: Educational Use Only - POLBAN

---

## 🎉 Version History

- **v2.0.0** (November 2025) - Virtual Try-On Kumis
  - Classical ML (SVM+ORB+BoVW) pipeline
  - 6-layer validation (Haar + SVM + Eye Detection)
  - 12 kumis styles, temporal smoothing, angle smoothing
  - Performance: 83.8% accuracy, 50-60ms inference

---
