"""
Utility functions for the Kumis Try-On system
"""

import cv2
import numpy as np
import time
from pathlib import Path


def compute_iou(box1, box2):
    """
    Compute IoU (Intersection over Union) between two bounding boxes.
    
    Args:
        box1, box2: (x, y, w, h) format
    
    Returns:
        float: IoU score (0.0 to 1.0)
    """
    x1, y1, w1, h1 = box1[:4]
    x2, y2, w2, h2 = box2[:4]
    
    # Calculate intersection
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - intersection_area
    
    iou = intersection_area / union_area if union_area > 0 else 0.0
    return iou


def non_max_suppression(boxes, scores=None, iou_threshold=0.3):
    """
    Non-Maximum Suppression to remove overlapping boxes.
    
    Args:
        boxes: List of (x, y, w, h) tuples
        scores: List of confidence scores (optional)
        iou_threshold: IoU threshold for suppression
    
    Returns:
        List of kept boxes
    """
    if len(boxes) == 0:
        return []
    
    # If no scores provided, use dummy scores
    if scores is None:
        scores = [1.0] * len(boxes)
    
    # Combine boxes with scores
    boxes_with_scores = [(box, score) for box, score in zip(boxes, scores)]
    
    # Sort by score (descending)
    boxes_with_scores.sort(key=lambda x: x[1], reverse=True)
    
    keep = []
    
    while len(boxes_with_scores) > 0:
        # Keep highest score box
        best_box, best_score = boxes_with_scores.pop(0)
        keep.append(best_box)
        
        # Remove overlapping boxes
        boxes_with_scores = [
            (box, score) for box, score in boxes_with_scores
            if compute_iou(best_box, box) < iou_threshold
        ]
    
    return keep


def draw_face_box(img, box, label="", color=(0, 255, 0), thickness=2):
    """
    Draw bounding box with label on image.
    
    Args:
        img: Input image (BGR)
        box: (x, y, w, h) tuple
        label: Text label
        color: Box color (BGR)
        thickness: Line thickness
    
    Returns:
        Image with box drawn
    """
    x, y, w, h = box
    cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
    
    if label:
        # Draw label background
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x, y - text_h - 5), (x + text_w, y), color, -1)
        cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (255, 255, 255), 1)
    
    return img


def visualize_keypoints(img, keypoints, color=(0, 255, 0)):
    """
    Draw ORB keypoints on image.
    
    Args:
        img: Input image
        keypoints: ORB keypoints
        color: Keypoint color
    
    Returns:
        Image with keypoints drawn
    """
    img_with_kp = cv2.drawKeypoints(
        img, keypoints, None, 
        color=color, 
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    return img_with_kp


class FPSCounter:
    """Simple FPS counter for performance monitoring."""
    
    def __init__(self, avg_over=30):
        self.avg_over = avg_over
        self.frame_times = []
        self.last_time = time.time()
    
    def update(self):
        """Update with new frame."""
        current_time = time.time()
        delta = current_time - self.last_time
        self.last_time = current_time
        
        self.frame_times.append(delta)
        if len(self.frame_times) > self.avg_over:
            self.frame_times.pop(0)
    
    def get_fps(self):
        """Get current FPS."""
        if not self.frame_times:
            return 0.0
        avg_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_time if avg_time > 0 else 0.0


def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    Path(directory).mkdir(parents=True, exist_ok=True)


def load_image(path, grayscale=False):
    """
    Load image from path.
    
    Args:
        path: Image file path
        grayscale: Load as grayscale if True
    
    Returns:
        numpy array or None if error
    """
    try:
        if grayscale:
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(str(path))
        return img
    except Exception as e:
        print(f"Error loading image {path}: {e}")
        return None


def save_image(img, path):
    """
    Save image to path.
    
    Args:
        img: Image array
        path: Output path
    
    Returns:
        bool: Success status
    """
    try:
        ensure_dir(Path(path).parent)
        cv2.imwrite(str(path), img)
        return True
    except Exception as e:
        print(f"Error saving image {path}: {e}")
        return False


def resize_keep_aspect(img, target_width=None, target_height=None):
    """
    Resize image keeping aspect ratio.
    
    Args:
        img: Input image
        target_width: Target width (optional)
        target_height: Target height (optional)
    
    Returns:
        Resized image
    """
    h, w = img.shape[:2]
    
    if target_width is not None:
        ratio = target_width / w
        new_w = target_width
        new_h = int(h * ratio)
    elif target_height is not None:
        ratio = target_height / h
        new_h = target_height
        new_w = int(w * ratio)
    else:
        return img
    
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized
