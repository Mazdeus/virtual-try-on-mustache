"""
Performance Benchmark Script untuk Kumis Try-On System
Mengukur latency tiap stage dan resource usage secara real-time

Usage:
    python benchmark_performance.py --duration 60 --output reports/benchmark.json
"""

import cv2
import numpy as np
import time
import json
import psutil
import os
import sys
from pathlib import Path
from collections import defaultdict
import argparse

# Import pipeline modules
from pipelines.infer import FaceDetector
from pipelines.overlay import KumisOverlay
from pipelines.features import ORBFeatureExtractor, BoVWEncoder
from pipelines.train import load_models


class PerformanceBenchmark:
    """
    Benchmark tool untuk mengukur performance sistem real-time.
    Mengukur:
    - Latency per stage (Haar, ORB, SVM, Overlay, dll)
    - FPS actual
    - CPU usage
    - RAM usage
    - Network bandwidth (JPEG encoding size)
    """
    
    def __init__(self, camera_id=0, width=640, height=480, model_dir='models'):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.model_dir = model_dir
        
        # Models
        self.face_detector = None
        self.kumis_overlay = None
        
        # Benchmark data storage
        self.timings = defaultdict(list)
        self.fps_samples = []
        self.cpu_samples = []
        self.ram_samples = []
        self.jpeg_sizes = []
        
        # Process info for resource monitoring
        self.process = psutil.Process(os.getpid())
        
        print("=" * 60)
        print("🔬 Performance Benchmark Tool")
        print("=" * 60)
    
    def load_models(self):
        """Load trained models."""
        print(f"\n📦 Loading models from {self.model_dir}...")
        
        try:
            # Load SVM, scaler, codebook
            svm, scaler, codebook, config = load_models(self.model_dir)
            
            # Initialize ORB extractor
            orb_extractor = ORBFeatureExtractor(nfeatures=500)
            
            # Initialize BoVW encoder
            bovw_encoder = BoVWEncoder(k=config.get('k', 256))
            bovw_encoder.kmeans = codebook
            
            # Initialize face detector
            self.face_detector = FaceDetector(
                svm=svm,
                scaler=scaler,
                bovw_encoder=bovw_encoder,
                orb_extractor=orb_extractor,
                confidence_threshold=0.25
            )
            
            print(f"  ✅ Models loaded successfully!")
            
        except Exception as e:
            print(f"  ❌ Error loading models: {e}")
            print(f"     Run: python app.py train")
            sys.exit(1)
    
    def load_kumis(self, kumis_file='chevron.png'):
        """Load default kumis for overlay testing."""
        kumis_path = Path('assets/kumis') / kumis_file
        
        if not kumis_path.exists():
            print(f"  ⚠️  Kumis not found: {kumis_path}")
            print(f"     Skipping overlay benchmark...")
            return False
        
        try:
            self.kumis_overlay = KumisOverlay(str(kumis_path))
            print(f"  ✅ Kumis loaded: {kumis_file}")
            return True
        except Exception as e:
            print(f"  ❌ Error loading kumis: {e}")
            return False
    
    def initialize_camera(self):
        """Initialize webcam."""
        print(f"\n🎥 Initializing camera {self.camera_id}...")
        
        # Try DirectShow first (Windows)
        self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
        
        if not self.cap.isOpened():
            print(f"  ⚠️  Trying default backend...")
            self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            print(f"  ❌ Cannot open camera {self.camera_id}")
            sys.exit(1)
        
        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        # Warm up
        for _ in range(5):
            self.cap.read()
        
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"  ✅ Camera ready: {actual_width}x{actual_height}")
    
    def measure_stage(self, stage_name):
        """Context manager untuk mengukur waktu eksekusi."""
        class TimerContext:
            def __init__(self, benchmark, name):
                self.benchmark = benchmark
                self.name = name
                self.start_time = None
            
            def __enter__(self):
                self.start_time = time.perf_counter()
                return self
            
            def __exit__(self, *args):
                elapsed = (time.perf_counter() - self.start_time) * 1000  # Convert to ms
                self.benchmark.timings[self.name].append(elapsed)
        
        return TimerContext(self, stage_name)
    
    def process_frame_with_benchmark(self, frame):
        """
        Process single frame dengan detailed timing per stage.
        Returns: processed frame, face_detected (bool)
        """
        face_detected = False
        processed_frame = frame.copy()
        
        # Stage 1: Webcam Capture (already done, just for consistency)
        # Measured in run_benchmark()
        
        # Stage 2: Haar Cascade Detection
        with self.measure_stage('haar_cascade'):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_rois = self.face_detector.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(40, 40)
            )
        
        if len(face_rois) == 0:
            return processed_frame, False
        
        # Take first detected face for benchmark
        x, y, w, h = face_rois[0]
        face_roi = gray[y:y+h, x:x+w]
        
        # Stage 3: ORB Feature Extraction
        with self.measure_stage('orb_extraction'):
            keypoints, descriptors = self.face_detector.orb_extractor.detect_and_compute(face_roi)
        
        if descriptors is None or len(descriptors) == 0:
            return processed_frame, False
        
        # Stage 4: BoVW Encoding
        with self.measure_stage('bovw_encoding'):
            feature_vector = self.face_detector.bovw_encoder.encode(descriptors)
        
        # Stage 5: SVM Prediction
        with self.measure_stage('svm_prediction'):
            feature_scaled = self.face_detector.scaler.transform([feature_vector])
            decision = self.face_detector.svm.decision_function(feature_scaled)[0]
            confidence = 1 / (1 + np.exp(-decision))  # Sigmoid
        
        if confidence < self.face_detector.confidence_threshold:
            return processed_frame, False
        
        # Stage 6: Eye Detection
        with self.measure_stage('eye_detection'):
            eye_roi = face_roi[0:int(h*0.5), :]  # Top half of face
            eyes = self.face_detector.eye_cascade.detectMultiScale(
                eye_roi,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(20, 20)
            )
        
        if eyes is None or len(eyes) < 2:
            return processed_frame, False
        
        face_detected = True
        
        # Stage 7: Kumis Overlay (if available)
        if self.kumis_overlay is not None:
            with self.measure_stage('kumis_overlay'):
                # Convert eyes back to frame coordinates
                eyes_frame = [(x + ex, y + ey, ew, eh) for (ex, ey, ew, eh) in eyes]
                processed_frame = self.kumis_overlay.overlay(
                    processed_frame,
                    face_box=(x, y, w, h),
                    eyes=eyes_frame[:2],  # Use first 2 eyes
                    scale_factor=0.65,
                    y_offset_factor=0.55
                )
        
        return processed_frame, face_detected
    
    def sample_resources(self):
        """Sample CPU and RAM usage."""
        try:
            # CPU usage (percentage)
            cpu_percent = self.process.cpu_percent(interval=0.01)
            self.cpu_samples.append(cpu_percent)
            
            # RAM usage (MB)
            ram_mb = self.process.memory_info().rss / (1024 * 1024)
            self.ram_samples.append(ram_mb)
        except:
            pass
    
    def run_benchmark(self, duration=60, display=True):
        """
        Run benchmark for specified duration.
        
        Args:
            duration: Test duration in seconds
            display: Show live preview window
        """
        print(f"\n🚀 Starting benchmark...")
        print(f"  Duration: {duration} seconds")
        print(f"  Display: {'Enabled' if display else 'Disabled'}")
        print(f"\n  Press 'q' to stop early\n")
        
        start_time = time.time()
        frame_count = 0
        faces_detected_count = 0
        last_fps_time = start_time
        fps_frame_count = 0
        
        try:
            while True:
                loop_start = time.perf_counter()
                
                # Stage: Webcam Capture
                with self.measure_stage('webcam_capture'):
                    ret, frame = self.cap.read()
                
                if not ret:
                    print("  ⚠️  Failed to capture frame")
                    break
                
                # Process frame with detailed timing
                frame_process_start = time.perf_counter()
                processed_frame, face_detected = self.process_frame_with_benchmark(frame)
                
                if face_detected:
                    faces_detected_count += 1
                
                # Stage: JPEG Encoding (untuk network bandwidth simulation)
                with self.measure_stage('jpeg_encoding'):
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
                    _, buffer = cv2.imencode('.jpg', processed_frame, encode_param)
                    jpeg_size = len(buffer.tobytes())
                    self.jpeg_sizes.append(jpeg_size)
                
                # Total frame time
                frame_time = (time.perf_counter() - loop_start) * 1000
                self.timings['total_frame'].append(frame_time)
                
                # Calculate FPS
                fps_frame_count += 1
                current_time = time.time()
                fps_elapsed = current_time - last_fps_time
                
                if fps_elapsed >= 1.0:  # Update FPS every second
                    fps = fps_frame_count / fps_elapsed
                    self.fps_samples.append(fps)
                    fps_frame_count = 0
                    last_fps_time = current_time
                    
                    # Sample resources
                    self.sample_resources()
                    
                    # Progress update
                    elapsed = current_time - start_time
                    print(f"  ⏱️  {elapsed:.0f}s | FPS: {fps:.1f} | Faces: {faces_detected_count}/{frame_count+1}")
                
                # Display (optional)
                if display:
                    # Add info overlay
                    info_frame = processed_frame.copy()
                    cv2.putText(info_frame, f"FPS: {self.fps_samples[-1] if self.fps_samples else 0:.1f}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(info_frame, f"Frame: {frame_count+1}", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    cv2.imshow('Benchmark (Press Q to stop)', info_frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\n  ⏹️  Stopped by user")
                        break
                
                frame_count += 1
                
                # Check duration
                if time.time() - start_time >= duration:
                    print(f"\n  ✅ Benchmark completed ({duration}s)")
                    break
                
        except KeyboardInterrupt:
            print("\n  ⏹️  Interrupted by user")
        finally:
            if display:
                cv2.destroyAllWindows()
        
        print(f"\n📊 Total frames processed: {frame_count}")
        print(f"   Faces detected: {faces_detected_count} ({faces_detected_count/frame_count*100:.1f}%)")
    
    def generate_report(self):
        """Generate benchmark report with statistics."""
        report = {
            'metadata': {
                'camera_id': self.camera_id,
                'resolution': f"{self.width}x{self.height}",
                'total_frames': len(self.timings['total_frame']),
                'model_config': self.model_dir
            },
            'latency_breakdown': {},
            'fps_stats': {},
            'resource_usage': {},
            'network_bandwidth': {}
        }
        
        # Latency breakdown (per stage)
        print("\n" + "=" * 60)
        print("📊 LATENCY BREAKDOWN (per frame)")
        print("=" * 60)
        
        stage_order = [
            'webcam_capture',
            'haar_cascade',
            'orb_extraction',
            'bovw_encoding',
            'svm_prediction',
            'eye_detection',
            'kumis_overlay',
            'jpeg_encoding',
            'total_frame'
        ]
        
        for stage in stage_order:
            if stage not in self.timings or len(self.timings[stage]) == 0:
                continue
            
            times = self.timings[stage]
            avg_time = np.mean(times)
            min_time = np.min(times)
            max_time = np.max(times)
            std_time = np.std(times)
            
            # Calculate percentage of total
            if stage != 'total_frame':
                total_avg = np.mean(self.timings['total_frame'])
                percentage = (avg_time / total_avg) * 100 if total_avg > 0 else 0
            else:
                percentage = 100.0
            
            report['latency_breakdown'][stage] = {
                'avg_ms': round(avg_time, 2),
                'min_ms': round(min_time, 2),
                'max_ms': round(max_time, 2),
                'std_ms': round(std_time, 2),
                'percentage': round(percentage, 1)
            }
            
            print(f"{stage:20s} | Avg: {avg_time:5.1f}ms | Min: {min_time:5.1f}ms | Max: {max_time:5.1f}ms | {percentage:5.1f}%")
        
        # FPS Statistics
        print("\n" + "=" * 60)
        print("📈 FPS STATISTICS")
        print("=" * 60)
        
        if len(self.fps_samples) > 0:
            fps_avg = np.mean(self.fps_samples)
            fps_min = np.min(self.fps_samples)
            fps_max = np.max(self.fps_samples)
            fps_std = np.std(self.fps_samples)
            
            report['fps_stats'] = {
                'average': round(fps_avg, 2),
                'min': round(fps_min, 2),
                'max': round(fps_max, 2),
                'std': round(fps_std, 2)
            }
            
            print(f"Average FPS:    {fps_avg:.2f}")
            print(f"Min FPS:        {fps_min:.2f}")
            print(f"Max FPS:        {fps_max:.2f}")
            print(f"Std Dev:        {fps_std:.2f}")
        
        # Resource Usage
        print("\n" + "=" * 60)
        print("💻 RESOURCE USAGE")
        print("=" * 60)
        
        if len(self.cpu_samples) > 0:
            cpu_avg = np.mean(self.cpu_samples)
            cpu_max = np.max(self.cpu_samples)
            
            report['resource_usage']['cpu'] = {
                'average_percent': round(cpu_avg, 1),
                'peak_percent': round(cpu_max, 1)
            }
            
            print(f"Average CPU:    {cpu_avg:.1f}%")
            print(f"Peak CPU:       {cpu_max:.1f}%")
        
        if len(self.ram_samples) > 0:
            ram_avg = np.mean(self.ram_samples)
            ram_peak = np.max(self.ram_samples)
            
            report['resource_usage']['ram'] = {
                'average_mb': round(ram_avg, 1),
                'peak_mb': round(ram_peak, 1)
            }
            
            print(f"Average RAM:    {ram_avg:.1f} MB")
            print(f"Peak RAM:       {ram_peak:.1f} MB")
        
        # Network Bandwidth (JPEG sizes)
        print("\n" + "=" * 60)
        print("🌐 NETWORK BANDWIDTH")
        print("=" * 60)
        
        if len(self.jpeg_sizes) > 0:
            jpeg_avg = np.mean(self.jpeg_sizes)
            jpeg_min = np.min(self.jpeg_sizes)
            jpeg_max = np.max(self.jpeg_sizes)
            
            # Calculate bandwidth (KB/s) assuming FPS
            fps_avg = report['fps_stats'].get('average', 15)
            bandwidth_kbps = (jpeg_avg * fps_avg) / 1024
            
            report['network_bandwidth'] = {
                'avg_frame_size_kb': round(jpeg_avg / 1024, 2),
                'min_frame_size_kb': round(jpeg_min / 1024, 2),
                'max_frame_size_kb': round(jpeg_max / 1024, 2),
                'estimated_bandwidth_kbps': round(bandwidth_kbps, 2)
            }
            
            print(f"Avg Frame Size:     {jpeg_avg/1024:.2f} KB")
            print(f"Min Frame Size:     {jpeg_min/1024:.2f} KB")
            print(f"Max Frame Size:     {jpeg_max/1024:.2f} KB")
            print(f"Est. Bandwidth:     {bandwidth_kbps:.2f} KB/s ({bandwidth_kbps*8:.1f} Kbps)")
        
        return report
    
    def save_report(self, report, output_path):
        """Save report to JSON file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n💾 Report saved: {output_file}")
    
    def cleanup(self):
        """Release resources."""
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description='Performance Benchmark untuk Kumis Try-On System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run 60-second benchmark with live preview
  python benchmark_performance.py --duration 60 --display
  
  # Run 30-second benchmark without preview (faster)
  python benchmark_performance.py --duration 30 --no-display
  
  # Custom output location
  python benchmark_performance.py --output reports/my_benchmark.json
  
  # Use external webcam
  python benchmark_performance.py --camera 1
        """
    )
    
    parser.add_argument('--camera', type=int, default=0, 
                       help='Camera device ID (default: 0)')
    parser.add_argument('--width', type=int, default=640,
                       help='Frame width (default: 640)')
    parser.add_argument('--height', type=int, default=480,
                       help='Frame height (default: 480)')
    parser.add_argument('--duration', type=int, default=60,
                       help='Benchmark duration in seconds (default: 60)')
    parser.add_argument('--models', default='models',
                       help='Models directory (default: models)')
    parser.add_argument('--kumis', default='chevron.png',
                       help='Kumis file for overlay test (default: chevron.png)')
    parser.add_argument('--output', default='reports/benchmark.json',
                       help='Output JSON file (default: reports/benchmark.json)')
    parser.add_argument('--display', dest='display', action='store_true',
                       help='Show live preview window')
    parser.add_argument('--no-display', dest='display', action='store_false',
                       help='Run without preview (faster)')
    parser.set_defaults(display=True)
    
    args = parser.parse_args()
    
    # Create benchmark instance
    benchmark = PerformanceBenchmark(
        camera_id=args.camera,
        width=args.width,
        height=args.height,
        model_dir=args.models
    )
    
    try:
        # Load models
        benchmark.load_models()
        
        # Load kumis
        benchmark.load_kumis(args.kumis)
        
        # Initialize camera
        benchmark.initialize_camera()
        
        # Run benchmark
        benchmark.run_benchmark(duration=args.duration, display=args.display)
        
        # Generate and save report
        report = benchmark.generate_report()
        benchmark.save_report(report, args.output)
        
        print("\n" + "=" * 60)
        print("✅ Benchmark selesai!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        benchmark.cleanup()


if __name__ == '__main__':
    main()
