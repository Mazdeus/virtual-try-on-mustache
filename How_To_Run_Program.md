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