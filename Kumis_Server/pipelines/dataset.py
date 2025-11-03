"""
Dataset module for loading and splitting face/non-face data
"""

import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import random


class FaceDataset:
    """
    Dataset class for face detection training.
    Handles loading, splitting, and ROI generation.
    """
    
    def __init__(self, pos_dir=None, neg_dir=None, img_size=(128, 128)):
        """
        Initialize dataset.
        
        Args:
            pos_dir: Directory with positive samples (faces)
            neg_dir: Directory with negative samples (non-faces)
            img_size: Target image size (width, height)
        """
        self.img_size = img_size
        self.pos_samples = []
        self.neg_samples = []
        
        if pos_dir:
            self.load_positive_samples(pos_dir)
        if neg_dir:
            self.load_negative_samples(neg_dir)
    
    def load_positive_samples(self, pos_dir):
        """Load positive face samples."""
        pos_path = Path(pos_dir)
        if not pos_path.exists():
            print(f"Warning: Positive directory not found: {pos_dir}")
            return
        
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        for ext in extensions:
            image_files.extend(pos_path.glob(ext))
            image_files.extend(pos_path.glob(ext.upper()))
        
        # Remove duplicates (Windows is case-insensitive)
        image_files = list(set(image_files))
        
        print(f"Loading positive samples from {pos_dir}...")
        for img_file in image_files:
            img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img_resized = cv2.resize(img, self.img_size)
                self.pos_samples.append((img_resized, 1))  # Label 1 = face
        
        print(f"  ✅ Loaded {len(self.pos_samples)} positive samples")
    
    def load_negative_samples(self, neg_dir):
        """Load negative non-face samples."""
        neg_path = Path(neg_dir)
        if not neg_path.exists():
            print(f"Warning: Negative directory not found: {neg_dir}")
            return
        
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        for ext in extensions:
            image_files.extend(neg_path.glob(ext))
            image_files.extend(neg_path.glob(ext.upper()))
        
        # Remove duplicates (Windows is case-insensitive)
        image_files = list(set(image_files))
        
        print(f"Loading negative samples from {neg_dir}...")
        for img_file in image_files:
            img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img_resized = cv2.resize(img, self.img_size)
                self.neg_samples.append((img_resized, 0))  # Label 0 = non-face
        
        print(f"  ✅ Loaded {len(self.neg_samples)} negative samples")
    
    def get_all_samples(self):
        """Get all samples (positive + negative)."""
        all_samples = self.pos_samples + self.neg_samples
        random.shuffle(all_samples)
        
        images = [sample[0] for sample in all_samples]
        labels = [sample[1] for sample in all_samples]
        
        return images, labels
    
    def split(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
        """
        Split dataset into train/val/test sets (stratified).
        
        Args:
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            random_state: Random seed
        
        Returns:
            (X_train, y_train), (X_val, y_val), (X_test, y_test)
        """
        images, labels = self.get_all_samples()
        
        # First split: train + (val+test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            images, labels,
            test_size=(val_ratio + test_ratio),
            random_state=random_state,
            stratify=labels
        )
        
        # Second split: val and test
        val_size = val_ratio / (val_ratio + test_ratio)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=(1 - val_size),
            random_state=random_state,
            stratify=y_temp
        )
        
        print(f"\n📊 Dataset split:")
        print(f"  Train: {len(X_train)} samples ({sum(y_train)} faces, {len(y_train)-sum(y_train)} non-faces)")
        print(f"  Val:   {len(X_val)} samples ({sum(y_val)} faces, {len(y_val)-sum(y_val)} non-faces)")
        print(f"  Test:  {len(X_test)} samples ({sum(y_test)} faces, {len(y_test)-sum(y_test)} non-faces)")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def generate_rois_from_images(self, image_dir, output_pos_dir, output_neg_dir, 
                                   haar_cascade_path=None):
        """
        Generate ROIs (face and non-face) from full images using Haar Cascade.
        
        Args:
            image_dir: Directory with full images
            output_pos_dir: Output directory for face crops
            output_neg_dir: Output directory for non-face crops
            haar_cascade_path: Path to Haar Cascade XML (optional)
        """
        if haar_cascade_path is None:
            haar_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        
        face_cascade = cv2.CascadeClassifier(haar_cascade_path)
        
        image_path = Path(image_dir)
        output_pos = Path(output_pos_dir)
        output_neg = Path(output_neg_dir)
        
        output_pos.mkdir(parents=True, exist_ok=True)
        output_neg.mkdir(parents=True, exist_ok=True)
        
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        for ext in extensions:
            image_files.extend(image_path.glob(ext))
        
        print(f"Generating ROIs from {len(image_files)} images...")
        
        face_count = 0
        neg_count = 0
        
        for img_file in image_files:
            img = cv2.imread(str(img_file))
            if img is None:
                continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
            )
            
            # Save face crops
            for (x, y, w, h) in faces:
                face_crop = gray[y:y+h, x:x+w]
                face_resized = cv2.resize(face_crop, self.img_size)
                
                output_file = output_pos / f"face_{face_count:05d}.jpg"
                cv2.imwrite(str(output_file), face_resized)
                face_count += 1
            
            # Generate negative samples (random crops outside face regions)
            for _ in range(min(5, len(faces) + 2)):  # 2-5 negatives per image
                neg_crop = self._random_crop_outside_faces(gray, faces, self.img_size)
                if neg_crop is not None:
                    output_file = output_neg / f"neg_{neg_count:05d}.jpg"
                    cv2.imwrite(str(output_file), neg_crop)
                    neg_count += 1
        
        print(f"  ✅ Generated {face_count} face samples")
        print(f"  ✅ Generated {neg_count} negative samples")
    
    def _random_crop_outside_faces(self, img, faces, crop_size):
        """Generate random crop that doesn't overlap with face regions."""
        h, w = img.shape[:2]
        crop_w, crop_h = crop_size
        
        max_attempts = 50
        for _ in range(max_attempts):
            # Random position
            x = random.randint(0, max(1, w - crop_w))
            y = random.randint(0, max(1, h - crop_h))
            
            # Check overlap with faces
            crop_box = (x, y, crop_w, crop_h)
            overlaps = False
            
            for (fx, fy, fw, fh) in faces:
                face_box = (fx, fy, fw, fh)
                if self._boxes_overlap(crop_box, face_box):
                    overlaps = True
                    break
            
            if not overlaps:
                crop = img[y:y+crop_h, x:x+crop_w]
                if crop.shape[:2] == crop_size:
                    return crop
        
        return None
    
    def _boxes_overlap(self, box1, box2):
        """Check if two boxes overlap."""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        return not (x1 + w1 < x2 or x2 + w2 < x1 or 
                   y1 + h1 < y2 or y2 + h2 < y1)
