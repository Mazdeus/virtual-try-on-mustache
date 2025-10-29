# 🚀 Quick Setup Guide - Virtual Environment

## Problem: Path Too Long Error

Jika Anda mendapat error seperti ini saat install packages:
```
ERROR: Could not install packages due to an OSError: [Errno 2] No such file or directory
HINT: This error might have occurred since this system does not have Windows Long Path support enabled.
```

Ini karena path project terlalu panjang untuk Windows (>260 karakter).

---

## ✅ Solution 1: Enable Windows Long Path (RECOMMENDED)

### Step 1: Run Script as Administrator

1. **Right-click** pada file `enable_long_paths.ps1`
2. Select **"Run with PowerShell"**
3. Jika muncul prompt, click **"Yes"** atau **"Run anyway"**

**OR manually:**

1. Right-click **PowerShell**
2. Select **"Run as Administrator"**
3. Navigate to project folder:
   ```powershell
   cd "D:\KULIAH\POLBAN 2023\TEKNIK INFORMATIKA\SEMESTER 5\Pengolahan_Citra_Digital\Praktek\ETS\Implementasi\Filter-Face-Godot-Mask\Filter-Face-Godot-Ver-main"
   ```
4. Run script:
   ```powershell
   .\enable_long_paths.ps1
   ```

### Step 2: Restart Computer (Important!)

Windows needs restart untuk activate long path support.

### Step 3: Install Packages

After restart:

```powershell
# Activate virtual environment
Kumis_Server\env\Scripts\Activate.ps1

# Install packages
pip install opencv-python numpy scikit-learn joblib matplotlib seaborn Pillow tqdm requests flake8

# Verify installation
python -c "import cv2, sklearn, numpy; print('✅ All packages installed!')"
```

---

## ✅ Solution 2: Move Project to Shorter Path (ALTERNATIVE)

Jika tidak bisa enable long path (misal: tidak punya admin access):

### Move project ke path yang lebih pendek:

```powershell
# Example: Move to C:\Projects\
Move-Item "D:\KULIAH\POLBAN 2023\TEKNIK INFORMATIKA\SEMESTER 5\Pengolahan_Citra_Digital\Praktek\ETS\Implementasi\Filter-Face-Godot-Mask\Filter-Face-Godot-Ver-main" "C:\Projects\Kumis-TryOn"
```

Then navigate and create venv:

```powershell
cd C:\Projects\Kumis-TryOn
python -m venv Kumis_Server\env
Kumis_Server\env\Scripts\Activate.ps1
pip install -r Kumis_Server\requirements.txt
```

---

## 📦 Virtual Environment Location

Virtual environment created at:
```
Kumis_Server/env/
```

**Included in .gitignore** - Won't be pushed to GitHub ✅

---

## 🔧 Activate Virtual Environment

**Every time** you work on this project:

```powershell
# Activate
Kumis_Server\env\Scripts\Activate.ps1

# Your prompt will change to show (env)
# Example: (env) PS D:\KULIAH\...>

# Deactivate when done
deactivate
```

---

## ✅ Verify Installation

After packages installed:

```powershell
# Activate environment
Kumis_Server\env\Scripts\Activate.ps1

# Check Python version
python --version
# Should show: Python 3.12.x

# Check installed packages
pip list

# Test imports
python -c "import cv2; print('OpenCV:', cv2.__version__)"
python -c "import sklearn; print('scikit-learn:', sklearn.__version__)"
python -c "import numpy; print('NumPy:', numpy.__version__)"
```

Expected output:
```
OpenCV: 4.12.0
scikit-learn: 1.7.2
NumPy: 2.2.6
```

---

## 📝 Important Notes

1. **Always activate** environment before running Python scripts:
   ```powershell
   Kumis_Server\env\Scripts\Activate.ps1
   python app.py train
   ```

2. **Deactivate** when switching projects:
   ```powershell
   deactivate
   ```

3. **Don't push** `env/` folder to GitHub (already in .gitignore)

4. **Recreate** environment on other machines:
   ```powershell
   python -m venv Kumis_Server\env
   Kumis_Server\env\Scripts\Activate.ps1
   pip install -r Kumis_Server\requirements.txt
   ```

---

## 🐛 Troubleshooting

### Error: "Activate.ps1 cannot be loaded"

Solution:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Error: "python: command not found"

Solution:
- Install Python 3.8-3.12 from https://python.org
- During installation, check **"Add Python to PATH"**

### Error: "pip: command not found"

Solution:
```powershell
python -m ensurepip --upgrade
```

### Error: Package installation fails

Solution:
```powershell
# Upgrade pip first
python -m pip install --upgrade pip

# Then install packages
pip install -r Kumis_Server\requirements.txt
```

---

## 📚 Next Steps

After successful installation, see:
- **HOW_TO_RUN.md** - Complete guide from dataset collection to running app
- **ABOUT_PROJECT.md** - Project architecture and technical details

---

**✅ You're ready to go!**
