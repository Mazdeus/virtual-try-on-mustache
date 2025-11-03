"""
Tool untuk mengumpulkan dataset face/non-face dari webcam atau generate dari images
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
import time


def collect_from_webcam(output_dir, count=500, camera_id=0, face_type='positive'):
    """
    Collect face/non-face samples dari webcam.
    
    Args:
        output_dir: Output directory
        count: Jumlah sampel yang ingin dikumpulkan
        camera_id: Camera device ID
        face_type: 'positive' (face) or 'negative' (non-face)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load Haar Cascade
    if face_type == 'positive':
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
    
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"Cannot open camera {camera_id}")
        return
    
    print(f"\n📸 Collecting {count} {face_type} samples...")
    print(f"  Output: {output_path}")
    print(f"\n  Instructions:")
    if face_type == 'positive':
        print(f"    - Look at camera with different angles")
        print(f"    - Move closer/farther")
        print(f"    - Change lighting")
        print(f"    - Press SPACE to capture")
        print(f"    - Press 'q' to quit")
    else:
        print(f"    - Point camera at non-face objects")
        print(f"    - Walls, books, patterns, etc.")
        print(f"    - Press SPACE to capture")
        print(f"    - Press 'q' to quit")
    
    collected = 0
    last_capture = 0
    
    while collected < count:
        ret, frame = cap.read()
        if not ret:
            break
        
        display = frame.copy()
        
        # Detect faces for positive samples
        if face_type == 'positive':
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
            
            for (x, y, w, h) in faces:
                cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Show count
        cv2.putText(display, f"Collected: {collected}/{count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display, "Press SPACE to capture", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Data Collection', display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # Space to capture
            current_time = time.time()
            if current_time - last_capture > 0.3:  # Prevent double-capture
                if face_type == 'positive':
                    # Save face crops
                    for (x, y, w, h) in faces:
                        face_crop = gray[y:y+h, x:x+w]
                        face_resized = cv2.resize(face_crop, (128, 128))
                        
                        output_file = output_path / f"face_{collected:05d}.jpg"
                        cv2.imwrite(str(output_file), face_resized)
                        collected += 1
                        
                        if collected >= count:
                            break
                else:
                    # Save random crop
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    h, w = gray.shape
                    
                    crop_size = np.random.randint(80, 150)
                    x = np.random.randint(0, max(1, w - crop_size))
                    y = np.random.randint(0, max(1, h - crop_size))
                    
                    crop = gray[y:y+crop_size, x:x+crop_size]
                    crop_resized = cv2.resize(crop, (128, 128))
                    
                    output_file = output_path / f"neg_{collected:05d}.jpg"
                    cv2.imwrite(str(output_file), crop_resized)
                    collected += 1
                
                last_capture = current_time
                print(f"  ✅ Captured {collected}/{count}")
        
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n✅ Collection complete! Saved {collected} samples to {output_path}")


def generate_negatives_from_images(image_dir, output_dir, count=1000):
    """
    Generate negative samples dari folder images (random crops).
    
    Args:
        image_dir: Directory with source images
        output_dir: Output directory
        count: Jumlah negative samples
    """
    image_path = Path(image_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all images
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in extensions:
        image_files.extend(image_path.glob(ext))
        image_files.extend(image_path.glob(ext.upper()))
    
    # Remove duplicates (Windows is case-insensitive)
    image_files = list(set(image_files))
    
    if len(image_files) == 0:
        print(f"No images found in {image_dir}")
        return
    
    print(f"\n🎲 Generating {count} negative samples from {len(image_files)} images...")
    
    # Load Haar Cascade to avoid faces
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    generated = 0
    
    while generated < count:
        # Random image
        img_file = np.random.choice(image_files)
        img = cv2.imread(str(img_file))
        
        if img is None:
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Detect faces to avoid them
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
        
        # Generate random crop
        for _ in range(3):  # Try 3 times per image
            crop_size = np.random.randint(80, min(200, h, w))
            x = np.random.randint(0, max(1, w - crop_size))
            y = np.random.randint(0, max(1, h - crop_size))
            
            # Check overlap with faces
            overlaps = False
            for (fx, fy, fw, fh) in faces:
                if not (x + crop_size < fx or fx + fw < x or 
                       y + crop_size < fy or fy + fh < y):
                    overlaps = True
                    break
            
            if not overlaps:
                crop = gray[y:y+crop_size, x:x+crop_size]
                crop_resized = cv2.resize(crop, (128, 128))
                
                output_file = output_path / f"neg_{generated:05d}.jpg"
                cv2.imwrite(str(output_file), crop_resized)
                generated += 1
                
                if generated % 100 == 0:
                    print(f"  Generated {generated}/{count}")
                
                if generated >= count:
                    break
        
        if generated >= count:
            break
    
    print(f"\n✅ Generated {generated} negative samples in {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Dataset Collection Tool')
    parser.add_argument('--webcam', action='store_true', help='Collect from webcam')
    parser.add_argument('--generate-negatives', action='store_true', help='Generate negatives from images')
    parser.add_argument('--image-dir', help='Source image directory (for generate-negatives)')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--count', type=int, default=500, help='Number of samples')
    parser.add_argument('--camera', type=int, default=0, help='Camera device ID')
    parser.add_argument('--type', default='positive', choices=['positive', 'negative'], 
                       help='Sample type for webcam (positive=faces, negative=non-faces)')
    
    args = parser.parse_args()
    
    if args.webcam:
        collect_from_webcam(args.output, args.count, args.camera, args.type)
    elif args.generate_negatives:
        if not args.image_dir:
            print("Error: --image-dir required for --generate-negatives")
            return
        generate_negatives_from_images(args.image_dir, args.output, args.count)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
