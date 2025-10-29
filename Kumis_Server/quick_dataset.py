"""
quick_dataset.py

QUICK & EASY dataset collection tanpa internet!
Gunakan webcam untuk collect non-face images

Usage:
    python quick_dataset.py

Instructions:
    - Tunjukkan berbagai OBJEK ke webcam (JANGAN wajah!)
    - Press SPACE untuk capture
    - Press Q untuk quit
    - Target: 1000 gambar

Tips untuk good dataset:
    - Variasi objek: buku, mug, dinding, lantai, meja, kursi, dll
    - Variasi angle: depan, samping, atas, bawah
    - Variasi lighting: terang, gelap, backlight
    - Variasi jarak: dekat, jauh
"""

import cv2
import os
from datetime import datetime


def collect_non_faces(output_dir="data/non_faces", target=1000):
    """Collect non-face images using webcam"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Count existing images
    existing = len([f for f in os.listdir(output_dir) if f.endswith('.jpg')])
    count = existing
    
    print("=" * 60)
    print("📸 QUICK NON-FACE DATASET COLLECTION")
    print("=" * 60)
    print(f"📂 Output: {output_dir}")
    print(f"📊 Existing: {existing} images")
    print(f"🎯 Target: {target} images")
    print(f"📈 Remaining: {target - existing} images")
    print("=" * 60)
    print("\n📋 INSTRUCTIONS:")
    print("✅ Tunjukkan OBJEK ke webcam (JANGAN wajah!)")
    print("✅ Good objects: buku, mug, dinding, meja, kursi, laptop, dll")
    print("✅ Press SPACE untuk capture")
    print("✅ Press Q untuk quit")
    print("=" * 60)
    
    input("\nPress ENTER to start...")
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Cannot open webcam!")
        return
    
    print("\n✅ Webcam opened!")
    print("📸 Start capturing (SPACE = capture, Q = quit)\n")
    
    while count < target:
        ret, frame = cap.read()
        
        if not ret:
            print("❌ Failed to grab frame")
            break
        
        # Resize for display
        display = cv2.resize(frame, (640, 480))
        
        # Add info overlay
        progress = f"{count}/{target} ({count/target*100:.1f}%)"
        cv2.putText(display, progress, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display, "SPACE=Capture Q=Quit", (10, 470),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show instruction in corner
        remaining = target - count
        cv2.putText(display, f"Remaining: {remaining}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        cv2.imshow('Non-Face Collection - Show OBJECTS (NOT faces!)', display)
        
        key = cv2.waitKey(1) & 0xFF
        
        # SPACE = capture
        if key == ord(' '):
            filename = f"{output_dir}/obj_{count:05d}.jpg"
            cv2.imwrite(filename, frame)
            count += 1
            print(f"✅ Captured: {count}/{target} ({count/target*100:.1f}%)")
            
            # Visual feedback
            flash = display.copy()
            cv2.rectangle(flash, (0, 0), (640, 480), (255, 255, 255), -1)
            cv2.imshow('Non-Face Collection - Show OBJECTS (NOT faces!)', flash)
            cv2.waitKey(100)
        
        # Q = quit
        elif key == ord('q') or key == ord('Q'):
            print("\n⚠️ Quitting early...")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 60)
    print("🎉 COLLECTION COMPLETE!")
    print("=" * 60)
    print(f"✅ Collected: {count - existing} new images")
    print(f"📊 Total: {count} images")
    print(f"📂 Location: {output_dir}")
    print("=" * 60)
    
    if count < target:
        print(f"\n⚠️ Target not reached. Still need {target - count} images.")
        print("💡 Run this script again to continue!")


def main():
    try:
        collect_non_faces(target=1000)
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
