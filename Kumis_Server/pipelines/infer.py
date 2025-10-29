"""
Inference module for face detection using SVM+ORB
"""

import cv2
import numpy as np
from .utils import non_max_suppression


class FaceDetector:
    """
    Face detector using Haar Cascade + SVM+ORB pipeline.
    """
    
    def __init__(self, svm, scaler, bovw_encoder, orb_extractor, 
                 haar_cascade_path=None, confidence_threshold=0.5):
        """
        Initialize face detector.
        
        Args:
            svm: Trained SVM classifier
            scaler: Feature scaler
            bovw_encoder: BoVW encoder with codebook
            orb_extractor: ORB feature extractor
            haar_cascade_path: Path to Haar Cascade XML
            confidence_threshold: Minimum confidence for face detection
        """
        self.svm = svm
        self.scaler = scaler
        self.bovw_encoder = bovw_encoder
        self.orb_extractor = orb_extractor
        self.confidence_threshold = confidence_threshold
        
        # Load Haar Cascade for face proposal
        if haar_cascade_path is None:
            haar_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(haar_cascade_path)
        
        # Load Haar Cascade for eye detection
        eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
        self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
    
    def detect_faces(self, frame, nms_threshold=0.3):
        """
        Detect faces in frame using full pipeline.
        
        Args:
            frame: Input image (BGR)
            nms_threshold: IoU threshold for NMS
        
        Returns:
            List of face dictionaries with 'box', 'confidence', 'eyes'
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Step 1: Haar Cascade face proposals
        face_rois = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50)
        )
        
        if len(face_rois) == 0:
            return []
        
        # Step 2: Classify each ROI with SVM
        valid_faces = []
        confidences = []
        
        for (x, y, w, h) in face_rois:
            roi = gray[y:y+h, x:x+w]
            roi_resized = cv2.resize(roi, (128, 128))
            
            # Extract ORB features
            _, descriptors = self.orb_extractor.detect_and_compute(roi_resized)
            
            # Encode as BoVW
            bovw_features = self.bovw_encoder.encode(descriptors)
            bovw_features = self.scaler.transform([bovw_features])
            
            # SVM prediction
            pred = self.svm.predict(bovw_features)[0]
            
            # Get confidence score
            if hasattr(self.svm, 'decision_function'):
                confidence = self.svm.decision_function(bovw_features)[0]
            elif hasattr(self.svm, 'predict_proba'):
                confidence = self.svm.predict_proba(bovw_features)[0][1]
            else:
                confidence = 1.0 if pred == 1 else 0.0
            
            # Only keep faces above threshold
            if pred == 1 and confidence >= self.confidence_threshold:
                valid_faces.append((x, y, w, h))
                confidences.append(confidence)
        
        if len(valid_faces) == 0:
            return []
        
        # Step 3: Non-Maximum Suppression
        final_faces = non_max_suppression(valid_faces, confidences, nms_threshold)
        
        # Step 4: Detect eyes for each face
        results = []
        for (x, y, w, h) in final_faces:
            face_roi = gray[y:y+h, x:x+w]
            eyes = self.detect_eyes(face_roi, (x, y))
            
            results.append({
                'box': (x, y, w, h),
                'confidence': 1.0,  # Could store actual confidence
                'eyes': eyes
            })
        
        return results
    
    def detect_eyes(self, face_roi, face_offset=(0, 0)):
        """
        Detect eyes in face ROI.
        
        Args:
            face_roi: Grayscale face image
            face_offset: (x, y) offset of face in original image
        
        Returns:
            List of eye positions [(x, y), ...] or None
        """
        # Crop to upper half of face where eyes are located
        h, w = face_roi.shape[:2]
        eye_region = face_roi[0:int(h*0.6), :]  # Upper 60% of face
        
        eyes = self.eye_cascade.detectMultiScale(
            eye_region,
            scaleFactor=1.05,  # More sensitive
            minNeighbors=3,    # Less strict
            minSize=(int(w*0.08), int(h*0.08)),  # Adaptive min size
            maxSize=(int(w*0.35), int(h*0.35))   # Adaptive max size
        )
        
        if len(eyes) < 2:
            return None
        
        # Convert to absolute coordinates
        fx, fy = face_offset
        eye_centers = []
        
        # Sort eyes by x position (left to right)
        eyes_sorted = sorted(eyes[:3], key=lambda e: e[0])
        
        for (ex, ey, ew, eh) in eyes_sorted[:2]:  # Only first 2 eyes
            eye_center_x = fx + ex + ew // 2
            eye_center_y = fy + ey + eh // 2
            eye_centers.append((eye_center_x, eye_center_y))
        
        # Validate eye positions (should be horizontally aligned)
        if len(eye_centers) == 2:
            dy = abs(eye_centers[1][1] - eye_centers[0][1])
            dx = abs(eye_centers[1][0] - eye_centers[0][0])
            
            # If eyes are too vertically misaligned or too close, reject
            if dy > dx * 0.3 or dx < w * 0.15:
                return None
        
        return eye_centers


def estimate_mouth_position(face_box, eyes=None):
    """
    Estimate mouth center position based on face box and eyes.
    
    Args:
        face_box: (x, y, w, h)
        eyes: List of eye positions [(x1, y1), (x2, y2)] or None
    
    Returns:
        (mouth_x, mouth_y)
    """
    x, y, w, h = face_box
    
    # Default: mouth at 70% from top of face
    mouth_x = x + w // 2
    mouth_y = y + int(h * 0.70)
    
    # Refinement using eyes
    if eyes is not None and len(eyes) >= 2:
        eye1, eye2 = eyes[0], eyes[1]
        
        # Mouth center aligned with eyes center horizontally
        eye_center_x = (eye1[0] + eye2[0]) // 2
        mouth_x = eye_center_x
        
        # Vertical: below eyes by eye-distance * 0.8
        eye_center_y = (eye1[1] + eye2[1]) // 2
        eye_distance = abs(eye2[0] - eye1[0])
        mouth_y = eye_center_y + int(eye_distance * 0.8)
    
    return (mouth_x, mouth_y)


def calculate_face_angle(eyes):
    """
    Calculate face rotation angle from eye positions.
    
    Args:
        eyes: [(x1, y1), (x2, y2)]
    
    Returns:
        Angle in degrees
    """
    if eyes is None or len(eyes) < 2:
        return 0.0
    
    eye1, eye2 = eyes[0], eyes[1]
    
    dx = eye2[0] - eye1[0]
    dy = eye2[1] - eye1[1]
    
    angle = np.degrees(np.arctan2(dy, dx))
    return angle
