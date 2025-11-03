"""
Kumis overlay module for alpha blending and positioning
"""

import cv2
import numpy as np


class KumisOverlay:
    """
    Kumis overlay engine with scaling, rotation, and alpha blending.
    """
    
    def __init__(self, kumis_path=None):
        """
        Initialize kumis overlay.
        
        Args:
            kumis_path: Path to kumis PNG file (with alpha channel)
        """
        self.kumis_img = None
        self.last_angle = 0.0  # For angle smoothing
        self.angle_smooth_factor = 0.4  # 40% new angle, 60% old angle
        if kumis_path:
            self.load_kumis(kumis_path)
    
    def load_kumis(self, kumis_path):
        """
        Load kumis image with alpha channel.
        
        Args:
            kumis_path: Path to PNG file
        """
        self.kumis_img = cv2.imread(kumis_path, cv2.IMREAD_UNCHANGED)
        
        if self.kumis_img is None:
            raise FileNotFoundError(f"Kumis image not found: {kumis_path}")
        
        # Ensure RGBA format
        if self.kumis_img.shape[2] != 4:
            print(f"Warning: Kumis image doesn't have alpha channel, adding default alpha")
            # Add alpha channel
            h, w = self.kumis_img.shape[:2]
            alpha = np.ones((h, w, 1), dtype=self.kumis_img.dtype) * 255
            self.kumis_img = np.concatenate([self.kumis_img, alpha], axis=2)
        
        print(f"  🎨 Kumis loaded: {kumis_path}")
    
    def overlay(self, frame, face_box, eyes=None, scale_factor=0.6, y_offset_factor=0.6):
        """
        Overlay kumis on frame at face position.
        
        Args:
            frame: Input frame (BGR)
            face_box: (x, y, w, h) face bounding box
            eyes: Eye positions [(x1, y1), (x2, y2)] for rotation
            scale_factor: Kumis width as fraction of face width
            y_offset_factor: Vertical position (0.6 = 60% of kumis above mouth)
        
        Returns:
            Frame with kumis overlay
        """
        if self.kumis_img is None:
            return frame
        
        x, y, w, h = face_box
        
        # Step 1: Scale kumis to fit face
        kumis_scaled = self._scale_kumis(w, scale_factor)
        
        # Step 2: Rotate kumis based on eyes with smoothing
        if eyes is not None and len(eyes) >= 2:
            angle = self._calculate_face_angle(eyes)
            # Smooth angle to reduce jitter
            smoothed_angle = self.last_angle * (1 - self.angle_smooth_factor) + angle * self.angle_smooth_factor
            self.last_angle = smoothed_angle
            kumis_rotated = self._rotate_kumis(kumis_scaled, smoothed_angle)
        else:
            # No eyes detected, gradually return to 0 rotation
            self.last_angle = self.last_angle * 0.8  # Decay to 0
            kumis_rotated = self._rotate_kumis(kumis_scaled, self.last_angle) if abs(self.last_angle) > 0.5 else kumis_scaled
        
        # Step 3: Calculate position (mouth position)
        mouth_pos = self._estimate_mouth_position(face_box, eyes)
        
        # Step 4: Alpha blend kumis onto frame
        frame = self._alpha_blend(frame, kumis_rotated, mouth_pos, y_offset_factor)
        
        return frame
    
    def _scale_kumis(self, face_width, scale_factor=0.6):
        """
        Scale kumis to fit face width.
        
        Args:
            face_width: Width of detected face
            scale_factor: Kumis width as fraction of face width
        
        Returns:
            Scaled kumis image (RGBA)
        """
        target_width = int(face_width * scale_factor)
        
        h, w = self.kumis_img.shape[:2]
        aspect_ratio = w / h
        target_height = int(target_width / aspect_ratio)
        
        kumis_scaled = cv2.resize(
            self.kumis_img,
            (target_width, target_height),
            interpolation=cv2.INTER_AREA
        )
        
        return kumis_scaled
    
    def _rotate_kumis(self, kumis_img, angle):
        """
        Rotate kumis image to match face orientation.
        
        Args:
            kumis_img: Kumis image (RGBA)
            angle: Rotation angle in degrees
        
        Returns:
            Rotated kumis image (RGBA)
        """
        if abs(angle) < 2:  # Skip rotation for small angles
            return kumis_img
        
        h, w = kumis_img.shape[:2]
        center = (w // 2, h // 2)
        
        # Rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new bounding box
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        # Adjust rotation matrix for new center
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        
        # Rotate with transparent border
        kumis_rotated = cv2.warpAffine(
            kumis_img, M, (new_w, new_h),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )
        
        return kumis_rotated
    
    def _alpha_blend(self, frame, kumis_img, mouth_pos, y_offset_factor=0.6):
        """
        Alpha blend kumis onto frame.
        
        Args:
            frame: Background frame (BGR)
            kumis_img: Kumis image (RGBA)
            mouth_pos: (x, y) mouth center position
            y_offset_factor: Vertical offset (0.6 = 60% above mouth)
        
        Returns:
            Frame with kumis blended
        """
        mouth_x, mouth_y = mouth_pos
        kumis_h, kumis_w = kumis_img.shape[:2]
        
        # Position: center horizontally, offset vertically
        x1 = mouth_x - kumis_w // 2
        y1 = mouth_y - int(kumis_h * y_offset_factor)
        x2 = x1 + kumis_w
        y2 = y1 + kumis_h
        
        # Boundary check
        frame_h, frame_w = frame.shape[:2]
        
        # Clip kumis if out of bounds
        kumis_x1 = 0
        kumis_y1 = 0
        kumis_x2 = kumis_w
        kumis_y2 = kumis_h
        
        if x1 < 0:
            kumis_x1 = -x1
            x1 = 0
        if y1 < 0:
            kumis_y1 = -y1
            y1 = 0
        if x2 > frame_w:
            kumis_x2 = kumis_w - (x2 - frame_w)
            x2 = frame_w
        if y2 > frame_h:
            kumis_y2 = kumis_h - (y2 - frame_h)
            y2 = frame_h
        
        # Check if kumis is completely out of frame
        if x1 >= x2 or y1 >= y2:
            return frame
        
        # Extract ROI from frame
        roi = frame[y1:y2, x1:x2]
        
        # Extract kumis region
        kumis_region = kumis_img[kumis_y1:kumis_y2, kumis_x1:kumis_x2]
        
        if kumis_region.shape[0] == 0 or kumis_region.shape[1] == 0:
            return frame
        
        # Ensure dimensions match
        if roi.shape[:2] != kumis_region.shape[:2]:
            # Resize kumis to match ROI
            kumis_region = cv2.resize(kumis_region, (roi.shape[1], roi.shape[0]))
        
        # Split channels
        kumis_bgr = kumis_region[:, :, :3]
        alpha = kumis_region[:, :, 3] / 255.0
        
        # Alpha blending
        alpha_3ch = np.stack([alpha, alpha, alpha], axis=2)
        blended = (alpha_3ch * kumis_bgr + (1 - alpha_3ch) * roi).astype(np.uint8)
        
        # Update frame
        frame[y1:y2, x1:x2] = blended
        
        return frame
    
    def _estimate_mouth_position(self, face_box, eyes=None):
        """
        Estimate mouth center position.
        
        Args:
            face_box: (x, y, w, h)
            eyes: Eye positions or None
        
        Returns:
            (mouth_x, mouth_y)
        """
        x, y, w, h = face_box
        
        # Default: mouth at 72% from top (slightly lower for better positioning)
        mouth_x = x + w // 2
        mouth_y = y + int(h * 0.72)
        
        # Refinement using eyes
        if eyes is not None and len(eyes) >= 2:
            eye1, eye2 = eyes[0], eyes[1]
            
            # Sort eyes left to right
            if eye1[0] > eye2[0]:
                eye1, eye2 = eye2, eye1
            
            # Horizontal: eyes center
            eye_center_x = (eye1[0] + eye2[0]) // 2
            mouth_x = eye_center_x
            
            # Vertical: below eyes with better calculation
            eye_center_y = (eye1[1] + eye2[1]) // 2
            eye_distance = abs(eye2[0] - eye1[0])
            
            # Use face height if eye distance seems wrong
            if eye_distance < w * 0.15 or eye_distance > w * 0.6:
                # Eyes detection might be wrong, use face-based estimate
                mouth_y = y + int(h * 0.72)
            else:
                # Good eye detection, use eye-based estimate
                mouth_y = eye_center_y + int(eye_distance * 0.9)
        
        return (mouth_x, mouth_y)
    
    def _calculate_face_angle(self, eyes):
        """Calculate face rotation angle from eyes."""
        if eyes is None or len(eyes) < 2:
            return 0.0
        
        eye1, eye2 = eyes[0], eyes[1]
        dx = eye2[0] - eye1[0]
        dy = eye2[1] - eye1[1]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Negate angle to fix inverted rotation
        # When right eye is higher (dy < 0), head tilts right, kumis should tilt right too
        return -angle

