"""
Feature extraction module using ORB and Bag of Visual Words
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans
import joblib
from tqdm import tqdm


class ORBFeatureExtractor:
    """
    ORB (Oriented FAST and Rotated BRIEF) feature extractor.
    """
    
    def __init__(self, nfeatures=500, scaleFactor=1.2, nlevels=8):
        """
        Initialize ORB detector.
        
        Args:
            nfeatures: Maximum number of keypoints to detect
            scaleFactor: Pyramid decimation ratio
            nlevels: Number of pyramid levels
        """
        self.orb = cv2.ORB_create(
            nfeatures=nfeatures,
            scaleFactor=scaleFactor,
            nlevels=nlevels,
            edgeThreshold=31,
            patchSize=31
        )
        self.nfeatures = nfeatures
    
    def detect_and_compute(self, img):
        """
        Detect keypoints and compute descriptors.
        
        Args:
            img: Grayscale image
        
        Returns:
            keypoints, descriptors
        """
        keypoints, descriptors = self.orb.detectAndCompute(img, None)
        return keypoints, descriptors
    
    def extract_descriptors_batch(self, images, verbose=True):
        """
        Extract descriptors from batch of images.
        
        Args:
            images: List of grayscale images
            verbose: Show progress bar
        
        Returns:
            List of descriptors (each is Nx32 array or None)
        """
        all_descriptors = []
        
        iterator = tqdm(images, desc="Extracting ORB features") if verbose else images
        
        for img in iterator:
            _, descriptors = self.detect_and_compute(img)
            all_descriptors.append(descriptors)
        
        return all_descriptors


class BoVWEncoder:
    """
    Bag of Visual Words encoder using k-means clustering.
    """
    
    def __init__(self, k=256, random_state=42):
        """
        Initialize BoVW encoder.
        
        Args:
            k: Number of visual words (codebook size)
            random_state: Random seed for k-means
        """
        self.k = k
        self.kmeans = None
        self.random_state = random_state
    
    def build_codebook(self, descriptors_list, max_descriptors=200000, verbose=True):
        """
        Build visual word codebook using k-means clustering.
        
        Args:
            descriptors_list: List of descriptor arrays
            max_descriptors: Maximum descriptors to use (for efficiency)
            verbose: Show progress
        
        Returns:
            Codebook (cluster centers)
        """
        print(f"\n🔧 Building codebook with k={self.k}...")
        
        # Collect all descriptors
        all_desc = []
        for desc in descriptors_list:
            if desc is not None and len(desc) > 0:
                all_desc.append(desc)
        
        if len(all_desc) == 0:
            raise ValueError("No descriptors found in dataset!")
        
        all_desc = np.vstack(all_desc)
        print(f"  Total descriptors: {len(all_desc)}")
        
        # Subsample if too many descriptors
        if len(all_desc) > max_descriptors:
            print(f"  Subsampling to {max_descriptors} descriptors...")
            indices = np.random.choice(len(all_desc), max_descriptors, replace=False)
            all_desc = all_desc[indices]
        
        # Convert to float for k-means
        all_desc = all_desc.astype(np.float32)
        
        # Run k-means
        print(f"  Running k-means clustering...")
        self.kmeans = KMeans(
            n_clusters=self.k,
            random_state=self.random_state,
            verbose=1 if verbose else 0,
            max_iter=100,
            n_init=10
        )
        self.kmeans.fit(all_desc)
        
        print(f"  ✅ Codebook built with {self.k} visual words")
        
        return self.kmeans.cluster_centers_
    
    def encode(self, descriptors):
        """
        Encode image descriptors as BoVW histogram.
        
        Args:
            descriptors: ORB descriptors (Nx32 array) or None
        
        Returns:
            Histogram of visual words (k-dimensional vector)
        """
        if self.kmeans is None:
            raise ValueError("Codebook not built yet! Call build_codebook() first.")
        
        # Handle case with no descriptors
        if descriptors is None or len(descriptors) == 0:
            # Return uniform histogram
            return np.ones(self.k, dtype=np.float32) / self.k
        
        # Convert to float
        descriptors = descriptors.astype(np.float32)
        
        # Assign descriptors to nearest clusters
        labels = self.kmeans.predict(descriptors)
        
        # Build histogram
        histogram, _ = np.histogram(labels, bins=np.arange(self.k + 1))
        
        # Normalize (L1 norm)
        histogram = histogram.astype(np.float32)
        norm = histogram.sum()
        if norm > 0:
            histogram = histogram / norm
        
        return histogram
    
    def encode_batch(self, descriptors_list, verbose=True):
        """
        Encode batch of descriptors as BoVW histograms.
        
        Args:
            descriptors_list: List of descriptor arrays
            verbose: Show progress
        
        Returns:
            Array of BoVW histograms (NxK)
        """
        histograms = []
        
        iterator = tqdm(descriptors_list, desc="Encoding BoVW") if verbose else descriptors_list
        
        for desc in iterator:
            hist = self.encode(desc)
            histograms.append(hist)
        
        return np.array(histograms)
    
    def save(self, path):
        """Save codebook to file."""
        if self.kmeans is None:
            raise ValueError("Codebook not built yet!")
        
        joblib.dump(self.kmeans, path)
        print(f"  💾 Codebook saved to {path}")
    
    def load(self, path):
        """Load codebook from file."""
        self.kmeans = joblib.load(path)
        self.k = self.kmeans.n_clusters
        print(f"  📂 Codebook loaded from {path} (k={self.k})")


def extract_bovw_features(images, labels, k=256, max_descriptors=200000, 
                          nfeatures=500, verbose=True):
    """
    Complete pipeline: Extract ORB features and encode as BoVW.
    
    Args:
        images: List of grayscale images
        labels: List of labels
        k: Codebook size
        max_descriptors: Max descriptors for k-means
        nfeatures: ORB keypoints per image
        verbose: Show progress
    
    Returns:
        X (BoVW features), y (labels), orb_extractor, bovw_encoder
    """
    # Step 1: Extract ORB descriptors
    print("\n🔍 Step 1: Extracting ORB descriptors...")
    orb_extractor = ORBFeatureExtractor(nfeatures=nfeatures)
    descriptors_list = orb_extractor.extract_descriptors_batch(images, verbose=verbose)
    
    # Step 2: Build codebook
    print("\n📖 Step 2: Building BoVW codebook...")
    bovw_encoder = BoVWEncoder(k=k)
    bovw_encoder.build_codebook(descriptors_list, max_descriptors=max_descriptors, verbose=verbose)
    
    # Step 3: Encode as BoVW
    print("\n🔢 Step 3: Encoding as BoVW histograms...")
    X = bovw_encoder.encode_batch(descriptors_list, verbose=verbose)
    y = np.array(labels)
    
    print(f"\n✅ Feature extraction complete!")
    print(f"  Feature shape: {X.shape}")
    print(f"  Labels shape: {y.shape}")
    
    return X, y, orb_extractor, bovw_encoder
