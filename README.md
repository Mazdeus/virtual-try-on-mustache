# 🎭 Topeng Nusantara - Virtual Try-On Application

Aplikasi Virtual Try-On untuk topeng tradisional Indonesia menggunakan teknologi face detection dan real-time video processing.

![Version](https://img.shields.io/badge/version-1.4.2-blue)
![Godot](https://img.shields.io/badge/Godot-4.x-blue)
![Python](https://img.shields.io/badge/Python-3.8--3.12-yellow)
![License](https://img.shields.io/badge/license-Educational-green)

---

## 📖 Tentang Program

**Topeng Nusantara** adalah aplikasi interaktif yang memungkinkan pengguna untuk:
- **Mencoba topeng tradisional Indonesia** secara virtual melalui webcam
- **Memilih dari 7 topeng preset** (Panji, Sumatra, Hudoq, Kelana, Prabu, Betawi, Bali)
- **Membuat topeng custom** dengan menggabungkan komponen Base, Mata, dan Mulut
- **Melihat hasil real-time** dengan face detection dan overlay mask

### 🎯 Tujuan Aplikasi

Aplikasi ini dikembangkan sebagai bagian dari mata kuliah **Pengolahan Citra Digital** di Politeknik Negeri Bandung untuk:
- Implementasi teknik face detection menggunakan MediaPipe
- Pengolahan citra real-time dengan OpenCV
- Networking dengan UDP protocol
- Game engine integration (Godot)

---

## 🔄 Alur Program

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      USER INTERFACE                         │
│                    (Godot Client)                           │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   Main   │→ │   Topeng     │→ │   Webcam     │         │
│  │   Menu   │  │  Selection   │  │   Scene      │         │
│  └──────────┘  └──────────────┘  └──────────────┘         │
│                        ↓ UDP                                │
│                   (Send Commands)                           │
└─────────────────────────────────────────────────────────────┘
                         ↓↑
           ┌─────────────────────────────┐
           │    UDP Socket (Port 8888)   │
           └─────────────────────────────┘
                         ↓↑
┌─────────────────────────────────────────────────────────────┐
│                 PYTHON SERVER                               │
│            (udp_webcam_server.py)                           │
│  ┌──────────────┐  ┌────────────────┐  ┌────────────────┐ │
│  │   Webcam     │→ │  Face Filter   │→ │   UDP Send     │ │
│  │   Capture    │  │  (filter_ref)  │  │   Frames       │ │
│  └──────────────┘  └────────────────┘  └────────────────┘ │
│         ↑                  ↓                                │
│    cv2.VideoCapture   MediaPipe FaceMesh                    │
└─────────────────────────────────────────────────────────────┘
```

### Detailed Flow

#### 1. **Startup & Main Menu**
```
User membuka aplikasi
  ↓
Godot menampilkan Main Menu
  ├─ Try On Mask → Ke Topeng Selection
  └─ Quit → Exit aplikasi
```

#### 2. **Topeng Selection**
```
User di Topeng Selection Scene
  ↓
Pilih jenis topeng:
  ├─ PRESET (Face 1-7): Topeng siap pakai
  │   ├─ Panji (panji3.png)
  │   ├─ Sumatra (sumatra.png)
  │   ├─ Hudoq (hudoq.png)
  │   ├─ Kelana (kelana.png)
  │   ├─ Prabu (prabu.png)
  │   ├─ Betawi (betawi.png)
  │   └─ Bali (bali.png)
  │
  └─ CUSTOM (+): Buat topeng sendiri
      ↓
      Custom Mask Scene
      ├─ Pilih Base (base1/2/3)
      ├─ Pilih Mata (mata1/2/3 atau None)
      ├─ Pilih Mulut (mulut1/2/3 atau None)
      └─ Preview Composite (real-time)
  ↓
User klik "Pilih"
  ↓
Godot save selection ke Global variable:
  - Global.selected_mask_type = "preset" / "custom"
  - Global.selected_mask_id = ID topeng
  - Global.custom_base/mata/mulut = komponen
  ↓
Change scene ke Webcam Scene
```

#### 3. **Webcam Scene & UDP Communication**
```
Webcam Scene loaded
  ↓
┌─────────────── GODOT CLIENT ───────────────┐
│ 1. Setup WebcamManagerUDP                  │
│    - Bind UDP port 9999                    │
│    - Connect ke server 127.0.0.1:8888      │
│                                             │
│ 2. Send CONNECT command                    │
│    UDP → "CONNECT"                          │
│    (Register client ke server)             │
│                                             │
│ 3. Send SET_MASK command                   │
│    Jika PRESET:                            │
│      UDP → "SET_MASK panji3.png"           │
│    Jika CUSTOM:                            │
│      UDP → "SET_CUSTOM_MASK base1 mata2 mulut3" │
│                                             │
│ 4. Receive video frames                    │
│    Loop:                                    │
│      - Receive UDP packet (JPEG bytes)     │
│      - Decode JPEG → Image                 │
│      - Display di TextureRect              │
└─────────────────────────────────────────────┘
                    ↓↑ UDP
┌─────────────── PYTHON SERVER ──────────────┐
│ 1. Camera initialization                   │
│    cv2.VideoCapture(0)                     │
│    Set resolution: 480x360 @ 15fps         │
│                                             │
│ 2. FilterEngine initialization             │
│    - Load MediaPipe FaceMesh               │
│    - Load mask images dari folder          │
│                                             │
│ 3. Listen for commands                     │
│    Thread listen UDP commands:             │
│      - CONNECT → Register client           │
│      - SET_MASK → Load mask file           │
│      - SET_CUSTOM_MASK → Composite mask    │
│                                             │
│ 4. Main loop (broadcast thread)            │
│    While running:                           │
│      ├─ Capture frame dari webcam          │
│      ├─ Detect face dengan MediaPipe       │
│      ├─ Apply mask overlay                 │
│      │   └─ filter_engine.apply_mask()     │
│      ├─ Encode frame → JPEG (quality 40)   │
│      └─ Send UDP ke semua clients          │
│         (Broadcast ke semua registered)    │
└─────────────────────────────────────────────┘
```

#### 4. **Face Detection & Mask Overlay (filter_ref.py)**
```
Input: Video frame (BGR)
  ↓
1. Convert BGR → RGB
  ↓
2. MediaPipe FaceMesh.process()
   - Detect 468 facial landmarks
   - Get face bounding box
  ↓
3. Load mask image (PNG with alpha)
  ↓
4. Resize mask to fit face
   - Calculate face dimensions
   - Resize mask proportionally
  ↓
5. Position mask on face
   - Align mask center to face center
   - Adjust vertical position
  ↓
6. Alpha blending
   For each pixel in mask:
     if alpha > threshold:
       output[y,x] = mask_color
     else:
       output[y,x] = original_frame[y,x]
  ↓
Output: Frame with mask overlay (BGR)
```

#### 5. **Custom Mask Compositing**

**Di Godot (Preview):**
```
User pilih komponen:
  ↓
create_composite_preview():
  1. Load base.png → Image
  2. Create composite canvas
  3. Blit base ke canvas
  4. Load mata.png → Resize → blend_rect()
  5. Load mulut.png → Resize → blend_rect()
  ↓
Display composite di preview
```

**Di Python Server (Real-time):**
```
Receive "SET_CUSTOM_MASK base1 mata2 mulut3"
  ↓
filter_ref.set_custom_mask():
  1. Load base1.png dari folder
  2. Load mata2.png dari folder
  3. Load mulut3.png dari folder
  4. Composite menggunakan cv2.addWeighted()
  5. Save hasil ke temp mask
  ↓
apply_mask() menggunakan temp mask
```

---

## 🚀 Cara Menjalankan Program

### Prerequisites

**1. Python 3.8 - 3.12**
```bash
python --version
# Output: Python 3.x.x
```

**2. Godot Engine 4.x**
- Download dari: https://godotengine.org/download
- Ekstrak dan jalankan `godot.exe`

### Installation Steps

#### Step 1: Install Python Dependencies

```bash
# Masuk ke folder Webcam Server
cd "Webcam Server"

# Install dependencies
pip install -r requirements.txt

# Expected packages:
# - opencv-python >= 4.8.0
# - numpy >= 1.24.0
# - mediapipe >= 0.10.0
```

**Troubleshooting Python 3.12:**
Jika error saat install, gunakan versi terbaru:
```bash
pip install opencv-python numpy mediapipe --upgrade
```

#### Step 2: Run Python Server

```bash
# Dari folder Webcam Server
python udp_webcam_server.py
```

**Expected Output:**
```
=== Optimized UDP Webcam Server (with filter integration) ===
ℹ️ Auto-detected masks folder: ...\Webcam Server\mask
🎥 Initializing optimized camera...
✅ Camera ready: 480x360 @ 15FPS
🔧 FilterEngine initialized (filter_ref.py detected).
🚀 Optimized UDP Server: 127.0.0.1:8888
📊 Settings: 480x360, 15FPS, Q40
```

#### Step 3: Run Godot Client

1. **Buka Godot Engine**
2. **Import Project**
   - Klik "Import"
   - Browse ke folder `Walking Simulator`
   - Pilih `project.godot`
   - Klik "Import & Edit"

3. **Run Project**
   - Klik **Play** (F5) atau tombol ▶️
   - Atau **Run Specific Scene** untuk test individual scene

4. **Main Menu akan muncul**
   - Klik **"Try On Mask"** untuk mulai
   - Pilih topeng → Klik **"Pilih"**
   - Webcam akan aktif dengan topeng overlay

#### Step 4: Test Application

**Test Preset Mask:**
```
Main Menu → Try On → Pilih "Panji" → Klik "✅ Pilih Topeng"
→ Webcam aktif dengan topeng Panji di wajah
```

**Test Custom Mask:**
```
Main Menu → Try On → Klik "+" (Custom)
→ Pilih Base 1
→ Pilih Mata 2
→ Pilih Mulut 3
→ Preview menampilkan composite
→ Klik "Pilih"
→ Webcam aktif dengan custom mask
```

### Running in Production

**Start Both Services:**
```bash
# Terminal 1: Python Server
cd "Webcam Server"
python udp_webcam_server.py

# Terminal 2: Godot Client
# (Run via Godot Editor atau export executable)
```

---

## 🛠️ Teknologi yang Digunakan

### Frontend (Client)

#### **Godot Engine 4.x**
- **Bahasa**: GDScript
- **Fungsi**: 
  - User Interface (UI/UX)
  - Scene management
  - UDP client untuk receive video frames
  - Image compositing (preview custom mask)

**Key Features:**
- Scene-based architecture
- Node system untuk UI components
- Built-in networking (UDP/PacketPeerUDP)
- Image processing (Image, ImageTexture)
- Signal/Slot untuk event handling

**Files:**
```
Walking Simulator/
├── Scenes/
│   ├── MainMenu/
│   │   ├── MainMenu.tscn                 # Main menu scene
│   │   └── MainMenuController.gd          # Menu logic
│   └── TopengNusantara/
│       ├── TopengSelectionScene.tscn      # Mask selection UI
│       ├── TopengSelectionController.gd   # Selection logic
│       ├── TopengCustomizationScene.tscn  # Custom mask builder
│       ├── TopengCustomizationController.gd # Composite logic
│       ├── TopengWebcamScene.tscn         # Webcam display
│       └── TopengWebcamController.gd      # UDP client & display
├── Scenes/EthnicityDetection/
│   └── WebcamClient/
│       └── WebcamManagerUDP.gd            # UDP networking
├── Global.gd                               # Global state
└── project.godot                           # Project config
```

---

### Backend (Server)

#### **Python 3.8-3.12**

**Core Libraries:**

**1. OpenCV (cv2) >= 4.8.0**
- **Fungsi**: 
  - Webcam capture (`VideoCapture`)
  - Image processing (resize, blend, color conversion)
  - JPEG encoding/decoding
- **Operasi Utama**:
  - `cv2.VideoCapture(0)` - Akses webcam
  - `cv2.resize()` - Resize images
  - `cv2.cvtColor()` - Color space conversion
  - `cv2.imencode('.jpg')` - Encode ke JPEG

**2. MediaPipe >= 0.10.0**
- **Fungsi**: 
  - Face detection
  - Facial landmark detection (468 landmarks)
- **Model**: FaceMesh
- **Output**: 
  - Face bounding box
  - 3D coordinates untuk setiap landmark
  - Face orientation

**3. NumPy >= 1.24.0**
- **Fungsi**: 
  - Array operations
  - Image manipulation
  - Alpha blending calculations

**Architecture:**

```python
udp_webcam_server.py          # Main server
├─ socket (UDP)                # Networking
├─ threading                   # Multi-threading
│   ├─ broadcast_thread        # Send frames
│   └─ listener_thread         # Receive commands
├─ cv2.VideoCapture            # Webcam
└─ filter_ref.FilterEngine     # Face filter
    ├─ MediaPipe FaceMesh      # Face detection
    ├─ cv2 image processing    # Mask overlay
    └─ Alpha blending          # Transparency
```

**Files:**
```
Webcam Server/
├── udp_webcam_server.py       # Main UDP server
├── filter_ref.py              # Face filter engine
├── mask/                      # Mask images (PNG)
│   ├── panji3.png
│   ├── sumatra.png
│   ├── base1.png, base2.png, base3.png
│   ├── mata1.png, mata2.png, mata3.png
│   └── mulut1.png, mulut2.png, mulut3.png
└── requirements.txt           # Python dependencies
```

---

### Networking

#### **UDP Protocol**

**Why UDP?**
- **Low Latency**: No handshake, cocok untuk real-time video
- **Fast**: Tidak ada retransmission overhead
- **Efficient**: Suitable untuk streaming aplikasi

**Ports:**
- **Server**: `127.0.0.1:8888` (listen & send frames)
- **Client**: `127.0.0.1:9999` (receive frames)

**Message Format:**

**Commands (Client → Server):**
```
CONNECT                                    # Register client
SET_MASK <filename>                        # Set preset mask
SET_CUSTOM_MASK <base> <mata> <mulut>     # Set custom mask
```

**Data (Server → Client):**
```
[JPEG bytes]                               # Raw image data
```

**Packet Size:**
- Max: ~65KB (UDP limit)
- Typical: 4-6KB (JPEG quality 40, 480x360)

---

### Image Processing Pipeline

#### **Face Detection (MediaPipe)**

```python
# Initialize
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Process frame
results = face_mesh.process(rgb_frame)

# Extract landmarks
if results.multi_face_landmarks:
    landmarks = results.multi_face_landmarks[0].landmark
    # 468 points: eyes, nose, mouth, face contour
```

#### **Mask Overlay (OpenCV + Alpha Blending)**

```python
# Load mask with alpha channel
mask_img = cv2.imread('mask.png', cv2.IMREAD_UNCHANGED)
# Shape: (H, W, 4) - BGRA

# Resize to fit face
mask_resized = cv2.resize(mask_img, (face_width, face_height))

# Alpha blending
alpha = mask_resized[:, :, 3] / 255.0  # Normalize alpha
for c in range(3):  # B, G, R channels
    frame[y:y+h, x:x+w, c] = (
        alpha * mask_resized[:, :, c] +
        (1 - alpha) * frame[y:y+h, x:x+w, c]
    )
```

#### **Image Compositing (Godot)**

```gdscript
# Create canvas
var composite = Image.create(width, height, false, Image.FORMAT_RGBA8)

# Copy base
composite.blit_rect(base_img, rect, position)

# Overlay mata with alpha
composite.blend_rect(mata_img, rect, position)

# Overlay mulut with alpha
composite.blend_rect(mulut_img, rect, position)

# Create texture
var texture = ImageTexture.create_from_image(composite)
```

---

## 📊 Performance & Optimization

### Server Optimization
- **Frame Rate**: 15 FPS (configurable)
- **Resolution**: 480x360 (balance quality vs bandwidth)
- **JPEG Quality**: 40 (compress untuk UDP)
- **Multi-threading**: Separate threads untuk capture, process, send

### Client Optimization
- **Frame Buffer**: Skip frames jika terlalu cepat
- **Texture Update**: Only update saat frame baru diterima
- **Scene Management**: Unload unused scenes

---

## 📁 Project Structure

```
Filter-Face-Godot-Ver-main/
│
├── README.md                          # Documentation (this file)
│
├── Webcam Server/                     # Python server
│   ├── udp_webcam_server.py          # Main server
│   ├── filter_ref.py                 # Face filter engine
│   ├── requirements.txt              # Dependencies
│   └── mask/                         # Mask images
│       ├── panji3.png
│       ├── sumatra.png
│       ├── hudoq.png
│       ├── kelana.png
│       ├── prabu.png
│       ├── betawi.png
│       ├── bali.png
│       ├── base1.png, base2.png, base3.png
│       ├── mata1.png, mata2.png, mata3.png
│       └── mulut1.png, mulut2.png, mulut3.png
│
└── Walking Simulator/                 # Godot client
    ├── project.godot                 # Godot project file
    ├── Global.gd                     # Global state
    │
    ├── Scenes/
    │   ├── MainMenu/
    │   │   ├── MainMenu.tscn
    │   │   └── MainMenuController.gd
    │   │
    │   ├── TopengNusantara/
    │   │   ├── TopengSelectionScene.tscn
    │   │   ├── TopengSelectionController.gd
    │   │   ├── TopengCustomizationScene.tscn
    │   │   ├── TopengCustomizationController.gd
    │   │   ├── TopengWebcamScene.tscn
    │   │   └── TopengWebcamController.gd
    │   │
    │   └── EthnicityDetection/
    │       └── WebcamClient/
    │           └── WebcamManagerUDP.gd
    │
    └── Assets/
        └── Masks/                    # Preview images
            ├── panji.png
            ├── sumatra.png
            ├── base1.png, base2.png, base3.png
            ├── mata1.png, mata2.png, mata3.png
            └── mulut1.png, mulut2.png, mulut3.png
```

---

## 🐛 Troubleshooting

### Python Server Issues

**Error: "No module named 'mediapipe'"**
```bash
pip install mediapipe opencv-python numpy
```

**Error: "Camera not found"**
- Pastikan webcam terhubung
- Check permission webcam di OS
- Coba ganti camera index di code (0 → 1)

**Error: "Address already in use"**
- Port 8888 sudah digunakan
- Kill process yang menggunakan port
- Atau ubah port di code

### Godot Client Issues

**Error: "Could not connect to server"**
- Pastikan Python server sudah running
- Check firewall settings
- Verify IP address (127.0.0.1)

**Preview kosong / tidak muncul**
- Restart Godot untuk re-import assets
- Check console untuk error messages
- Verify PNG files ada di Assets/Masks/

**Topeng tidak muncul di wajah**
- Check console Python: "🎭 Mask set to: ..."
- Pastikan wajah terdeteksi (lighting cukup)
- Check MediaPipe working (no warnings)

---

## 📝 Credits

**Developed by:**
- Politeknik Negeri Bandung
- Mata Kuliah: Pengolahan Citra Digital
- Semester 5 - Teknik Informatika

**Technologies:**
- Godot Engine (Juan Linietsky, Ariel Manzur, and contributors)
- MediaPipe (Google)
- OpenCV (Intel, Willow Garage, Itseez)

---

## 📄 License

Educational use only - Politeknik Negeri Bandung

---

## 🎉 Version History

- **v1.4.2** - Full composite preview dengan alpha blending
- **v1.4.1** - Hotfix: Assets actually copied
- **v1.4.0** - Custom mask preview & UDP warning fix
- **v1.3.0** - Fix mask not appearing on face
- **v1.2.0** - Mask preview & better labels
- **v1.1.0** - Main menu & bug fixes
- **v1.0.0** - Initial release

---

## 📞 Support

Untuk pertanyaan atau issues, silakan kontak:
- **Institution**: Politeknik Negeri Bandung
- **Department**: Teknik Informatika
- **Course**: Pengolahan Citra Digital

---

**Selamat mencoba! 🎭✨**

