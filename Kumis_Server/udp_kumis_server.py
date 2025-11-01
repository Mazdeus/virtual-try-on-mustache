"""
UDP Webcam Server untuk Kumis Try-On dengan integrasi Godot
Menggunakan SVM+ORB pipeline untuk face detection
"""

import cv2
import numpy as np
import socket
import threading
import time
import sys
from pathlib import Path

# Import pipeline modules
from pipelines.infer import FaceDetector
from pipelines.overlay import KumisOverlay
from pipelines.features import ORBFeatureExtractor, BoVWEncoder
from pipelines.train import load_models


class UDPKumisServer:
    """
    UDP Server untuk streaming webcam dengan kumis overlay.
    Compatible dengan Godot client.
    """
    
    def __init__(self, camera_id=0, width=640, height=480, fps=15, 
                 server_ip='127.0.0.1', server_port=8888, client_port=9999):
        """
        Initialize UDP Kumis Server.
        
        Args:
            camera_id: Webcam device ID
            width: Frame width
            height: Frame height
            fps: Target FPS
            server_ip: Server IP address
            server_port: Port untuk menerima commands dari client
            client_port: Port untuk broadcast frames ke client
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.server_ip = server_ip
        self.server_port = server_port
        self.client_port = client_port
        
        # UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((server_ip, server_port))
        
        # Registered clients
        self.clients = set()
        
        # Camera
        self.cap = None
        
        # Models
        self.face_detector = None
        self.kumis_overlay = None
        
        # State
        self.running = False
        self.current_kumis = None
        self.show_kumis = True
        
        print("=" * 50)
        print("🥸 Kumis Try-On Server (SVM+ORB)")
        print("=" * 50)
    
    def load_models(self, model_dir='models'):
        """Load trained SVM models and initialize detector."""
        print(f"\n📦 Loading models from {model_dir}...")
        
        try:
            # Load SVM, scaler, codebook
            svm, scaler, codebook, config = load_models(model_dir)
            
            # Initialize ORB extractor
            orb_extractor = ORBFeatureExtractor(nfeatures=500)
            
            # Initialize BoVW encoder with loaded codebook
            bovw_encoder = BoVWEncoder(k=config.get('k', 256))
            bovw_encoder.kmeans = codebook
            
            # Initialize face detector
            self.face_detector = FaceDetector(
                svm=svm,
                scaler=scaler,
                bovw_encoder=bovw_encoder,
                orb_extractor=orb_extractor,
                confidence_threshold=0.25  # Balance: tidak terlalu rendah (false positives) atau tinggi (missed detections)
            )
            
            print(f"  ✅ Models loaded successfully!")
            print(f"     - Codebook: {config.get('k', 256)} clusters")
            print(f"     - SVM: {config.get('svm_kernel', 'unknown')} kernel")
            
        except Exception as e:
            print(f"  ❌ Error loading models: {e}")
            print(f"     Please train models first: python app.py train")
            sys.exit(1)
    
    def detect_available_cameras(self, max_cameras=5):
        """Detect available cameras and let user choose."""
        print(f"\n🔍 Scanning for available cameras...")
        available_cameras = []
        
        # Test multiple backends for each camera ID
        backends = [
            (cv2.CAP_DSHOW, "DirectShow"),
            (cv2.CAP_MSMF, "Media Foundation"),
            (cv2.CAP_ANY, "Default")
        ]
        
        for cam_id in range(max_cameras):
            for backend_id, backend_name in backends:
                try:
                    cap = cv2.VideoCapture(cam_id, backend_id)
                    if cap.isOpened():
                        # Try to read a frame to verify it works
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            fps = int(cap.get(cv2.CAP_PROP_FPS))
                            available_cameras.append({
                                'id': cam_id,
                                'backend': backend_id,
                                'backend_name': backend_name,
                                'width': width,
                                'height': height,
                                'fps': fps
                            })
                            cap.release()
                            print(f"  ✅ Camera {cam_id} ({backend_name}): {width}x{height} @ {fps}FPS")
                            break  # Found working backend for this camera
                    cap.release()
                except:
                    pass
        
        return available_cameras
    
    def initialize_camera(self, auto_detect=False):
        """Initialize webcam with multi-backend support."""
        print(f"\n🎥 Initializing camera {self.camera_id}...")
        
        # Auto-detect mode
        if auto_detect:
            available_cameras = self.detect_available_cameras()
            if not available_cameras:
                print(f"  ❌ No cameras detected!")
                print(f"  💡 Tips:")
                print(f"     - Check if webcam is plugged in")
                print(f"     - Close other apps using webcam (Zoom, Teams, etc)")
                print(f"     - Check webcam permissions in Windows Settings")
                sys.exit(1)
            
            # Use first available camera
            cam_info = available_cameras[0]
            self.camera_id = cam_info['id']
            backend = cam_info['backend']
            print(f"\n  📹 Auto-selected: Camera {self.camera_id} ({cam_info['backend_name']})")
        else:
            backend = None
        
        # Try multiple backends in order
        backends_to_try = [
            (cv2.CAP_DSHOW, "DirectShow"),
            (cv2.CAP_MSMF, "Media Foundation"),
            (cv2.CAP_ANY, "Default")
        ]
        
        # If specific backend detected, try it first
        if backend is not None:
            backends_to_try = [(backend, "Auto-detected")] + backends_to_try
        
        self.cap = None
        for backend_id, backend_name in backends_to_try:
            try:
                print(f"  🔄 Trying {backend_name} backend...")
                test_cap = cv2.VideoCapture(self.camera_id, backend_id)
                
                if test_cap.isOpened():
                    # Test read frame
                    ret, test_frame = test_cap.read()
                    if ret and test_frame is not None:
                        self.cap = test_cap
                        print(f"  ✅ Success with {backend_name}!")
                        break
                    else:
                        test_cap.release()
                else:
                    test_cap.release()
            except Exception as e:
                print(f"  ⚠️ {backend_name} failed: {e}")
                continue
        
        if self.cap is None or not self.cap.isOpened():
            print(f"  ❌ Cannot open camera {self.camera_id}")
            print(f"  💡 Tips:")
            print(f"     - Close other apps using webcam (Zoom, Teams, etc)")
            print(f"     - Try different camera ID: --camera 1 or --camera 2")
            print(f"     - Use --auto-detect to scan all cameras")
            print(f"     - Check webcam permissions in Windows Settings")
            print(f"     - Try unplugging and replugging the webcam")
            sys.exit(1)
        
        # Set properties
        print(f"  ⏳ Configuring camera settings...")
        
        try:
            # Set resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            
            # Set FPS (optional, might not be supported)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Enable auto-exposure for better brightness
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
            
            # Increase brightness slightly
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)
            
            # Disable autofocus if causing issues
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            
            # Set buffer size to 1 (reduce latency)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
        except Exception as e:
            print(f"  ⚠️ Warning during configuration: {e}")
            print(f"  ℹ️ Continuing with default settings...")
        
        # Verify actual settings
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        print(f"  ✅ Camera initialized: {actual_width}x{actual_height} @ {actual_fps}FPS")
        
        # Update server settings to match actual camera
        self.width = actual_width
        self.height = actual_height
        
        # Warm up camera (discard first few frames)
        print(f"  🔥 Warming up camera...")
        for _ in range(5):
            self.cap.read()
        print(f"  ✅ Camera ready!")
    
    def start(self):
        """Start server threads."""
        print(f"\n🚀 Starting UDP Server...")
        print(f"  Server: {self.server_ip}:{self.server_port}")
        print(f"  Client: {self.server_ip}:{self.client_port}")
        print(f"\n📡 Waiting for connections...")
        print(f"  Send 'CONNECT' from client to register")
        print(f"  Send 'SET_KUMIS <filename>' to change kumis")
        print(f"\n  Press Ctrl+C to stop\n")
        
        self.running = True
        
        try:
            # Start listener thread (receive commands)
            print("  🔧 Starting command listener thread...")
            listener_thread = threading.Thread(target=self._listen_commands, daemon=True)
            listener_thread.start()
            
            # Start broadcast thread (send frames)
            print("  🔧 Starting frame broadcast thread...")
            broadcast_thread = threading.Thread(target=self._broadcast_frames, daemon=True)
            broadcast_thread.start()
            
            # Verify threads started
            time.sleep(0.5)
            if not listener_thread.is_alive():
                print("  ❌ ERROR: Listener thread failed to start!")
                return
            if not broadcast_thread.is_alive():
                print("  ❌ ERROR: Broadcast thread failed to start!")
                return
            
            print("  ✅ All threads running!\n")
            
            # Keep main thread alive
            loop_count = 0
            while self.running:
                loop_count += 1
                if loop_count % 10 == 0:  # Print every 10 seconds
                    print(f"  💓 Server heartbeat (loop {loop_count}): Listener={listener_thread.is_alive()}, Broadcast={broadcast_thread.is_alive()}")
                time.sleep(1.0)
                
        except KeyboardInterrupt:
            print("\n\n🛑 Shutting down server...")
        finally:
            self.stop()
    
    def stop(self):
        """Stop server."""
        self.running = False
        if self.cap:
            self.cap.release()
        self.sock.close()
        print("  ✅ Server stopped")
    
    def _listen_commands(self):
        """Thread: Listen for UDP commands from clients."""
        print("  🎧 Command listener thread started")
        
        # Set timeout once (not in loop)
        self.sock.settimeout(1.0)
        
        while self.running:
            try:
                try:
                    data, addr = self.sock.recvfrom(1024)
                except socket.timeout:
                    # Timeout is normal, just continue
                    continue
                except OSError as e:
                    # Handle Windows-specific socket errors
                    if e.winerror == 10054:
                        # Connection reset by peer - client disconnected abruptly
                        # This is normal, just continue
                        continue
                    elif self.running:
                        print(f"  ⚠️ Socket error: {e}")
                    continue
                except Exception as e:
                    if self.running:
                        print(f"  ⚠️ Unexpected error: {e}")
                    continue
                
                message = data.decode('utf-8').strip()
                
                # Parse command
                parts = message.split(' ', 1)
                cmd = parts[0].upper()
                
                if cmd == 'CONNECT':
                    # Register client
                    self.clients.add(addr)
                    print(f"  ✅ Client connected: {addr}")
                
                elif cmd == 'SET_KUMIS':
                    # Set kumis style
                    if len(parts) > 1:
                        kumis_file = parts[1]
                        self._set_kumis(kumis_file)
                
                elif cmd == 'TOGGLE_KUMIS':
                    # Toggle kumis on/off
                    self.show_kumis = not self.show_kumis
                    status = "ON" if self.show_kumis else "OFF"
                    print(f"  🎭 Kumis overlay: {status}")
                
                elif cmd == 'DISCONNECT':
                    # Remove client
                    if addr in self.clients:
                        self.clients.remove(addr)
                        print(f"  👋 Client disconnected: {addr}")
                
            except Exception as e:
                if self.running:
                    print(f"  ⚠️ Command error: {e}")
    
    def _broadcast_frames(self):
        """Thread: Capture and broadcast frames to clients."""
        print("  📹 Frame broadcast thread started")
        
        frame_delay = 1.0 / self.fps
        sequence_number = 0
        max_packet_size = 60000  # Max UDP packet size (minus header)
        
        try:
            while self.running:
                start_time = time.time()
                
                # Capture frame
                try:
                    ret, frame = self.cap.read()
                    if not ret or frame is None:
                        print(f"  ⚠️ Failed to read frame from camera")
                        time.sleep(0.1)
                        continue
                except Exception as e:
                    print(f"  ❌ Camera read error: {e}")
                    time.sleep(0.1)
                    continue
                
                # Resize if needed
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))
                
                # Detect faces and overlay kumis
                if self.show_kumis and self.kumis_overlay is not None:
                    frame = self._process_frame(frame)
                
                # Encode as JPEG with higher quality (85 = good balance)
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
                _, buffer = cv2.imencode('.jpg', frame, encode_param)
                frame_bytes = buffer.tobytes()
                
                # Split into packets with header
                total_size = len(frame_bytes)
                total_packets = (total_size + max_packet_size - 1) // max_packet_size
                
                # Broadcast to all clients
                for client_addr in list(self.clients):
                    try:
                        client_ip = client_addr[0]
                        
                        # Send each packet with header
                        for packet_index in range(total_packets):
                            start_idx = packet_index * max_packet_size
                            end_idx = min(start_idx + max_packet_size, total_size)
                            packet_data = frame_bytes[start_idx:end_idx]
                            
                            # Create header: sequence (4 bytes) + total_packets (4 bytes) + packet_index (4 bytes)
                            header = bytearray()
                            header.extend(sequence_number.to_bytes(4, byteorder='big'))
                            header.extend(total_packets.to_bytes(4, byteorder='big'))
                            header.extend(packet_index.to_bytes(4, byteorder='big'))
                            
                            # Combine header + data
                            packet = bytes(header) + packet_data
                            
                            # Send packet
                            self.sock.sendto(packet, (client_ip, self.client_port))
                            
                    except OSError as e:
                        # Handle Windows socket errors gracefully
                        if e.winerror == 10054:
                            # Client disconnected - remove from list
                            if client_addr in self.clients:
                                self.clients.discard(client_addr)
                                print(f"  👋 Client {client_addr} disconnected (connection lost)")
                        else:
                            print(f"  ⚠️ Broadcast error to {client_addr}: {e}")
                    except Exception as e:
                        print(f"  ⚠️ Broadcast error to {client_addr}: {e}")
                
                # Increment sequence number
                sequence_number = (sequence_number + 1) % 65536  # 16-bit sequence
                
                # Maintain FPS
                elapsed = time.time() - start_time
                if elapsed < frame_delay:
                    time.sleep(frame_delay - elapsed)
                    
        except Exception as e:
            print(f"  ❌ FATAL ERROR in broadcast thread: {e}")
            import traceback
            traceback.print_exc()
            self.running = False
    
    def _process_frame(self, frame):
        """Process frame: detect faces and overlay kumis."""
        try:
            # Detect faces
            faces = self.face_detector.detect_faces(frame, nms_threshold=0.3)
            
            # Debug logging (setiap 30 frames)
            if hasattr(self, '_frame_count'):
                self._frame_count += 1
            else:
                self._frame_count = 0
                
            if self._frame_count % 30 == 0:
                if len(faces) > 0:
                    # Log face info for debugging
                    for i, face_data in enumerate(faces):
                        x, y, w, h = face_data['box']
                        area_ratio = (w * h) / (frame.shape[0] * frame.shape[1])
                        center_x = x + w // 2
                        center_y = y + h // 2
                        has_eyes = "✅ EYES" if face_data['eyes'] is not None else "❌ NO EYES"
                        print(f"  🔍 Face {i+1}: {w}x{h} ({area_ratio:.1%}), center=({center_x},{center_y}), {has_eyes}")
                else:
                    print(f"  🔍 No faces detected")
            
            # Overlay kumis on each face
            for face_data in faces:
                frame = self.kumis_overlay.overlay(
                    frame,
                    face_box=face_data['box'],
                    eyes=face_data['eyes'],
                    scale_factor=0.65,      # Slightly larger kumis
                    y_offset_factor=0.55    # Lower position (more towards mouth)
                )
        
        except Exception as e:
            # Don't print error every frame to avoid spam
            pass
        
        return frame
    
    def _set_kumis(self, kumis_file):
        """Set kumis style."""
        kumis_path = Path('assets/kumis') / kumis_file
        
        if not kumis_path.exists():
            print(f"  ❌ Kumis not found: {kumis_path}")
            return
        
        try:
            self.kumis_overlay = KumisOverlay(str(kumis_path))
            self.current_kumis = kumis_file
            print(f"  � Kumis loaded: {kumis_path}")
            print(f"  �🎭 Kumis set to: {kumis_file}")
        except Exception as e:
            print(f"  ❌ Error loading kumis: {e}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='UDP Kumis Server with Multi-Backend Webcam Support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect and use first available camera
  python udp_kumis_server.py --auto-detect
  
  # Use specific camera ID
  python udp_kumis_server.py --camera 0
  python udp_kumis_server.py --camera 1  # External webcam
  
  # List all available cameras
  python udp_kumis_server.py --list-cameras
        """
    )
    parser.add_argument('--camera', type=int, default=0, help='Camera device ID (0=default, 1=external)')
    parser.add_argument('--auto-detect', action='store_true', help='Auto-detect and use first available camera')
    parser.add_argument('--list-cameras', action='store_true', help='List all available cameras and exit')
    parser.add_argument('--width', type=int, default=640, help='Frame width')
    parser.add_argument('--height', type=int, default=480, help='Frame height')
    parser.add_argument('--fps', type=int, default=15, help='Target FPS')
    parser.add_argument('--models', default='models', help='Models directory')
    
    args = parser.parse_args()
    
    # List cameras mode
    if args.list_cameras:
        print("\n" + "=" * 60)
        print("🎥 Available Cameras")
        print("=" * 60)
        temp_server = UDPKumisServer()
        cameras = temp_server.detect_available_cameras()
        if not cameras:
            print("\n❌ No cameras detected!")
        else:
            print(f"\n✅ Found {len(cameras)} camera(s):\n")
            for cam in cameras:
                print(f"  Camera {cam['id']}: {cam['backend_name']}")
                print(f"    Resolution: {cam['width']}x{cam['height']}")
                print(f"    FPS: {cam['fps']}")
                print()
        sys.exit(0)
    
    # Create server
    server = UDPKumisServer(
        camera_id=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps
    )
    
    # Load models
    server.load_models(args.models)
    
    # Initialize camera
    server.initialize_camera(auto_detect=args.auto_detect)
    
    # Start server
    server.start()


if __name__ == '__main__':
    main()
