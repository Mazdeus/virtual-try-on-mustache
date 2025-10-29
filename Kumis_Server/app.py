"""
CLI Tool untuk training dan evaluasi Kumis Try-On System
"""

import argparse
import sys
from pathlib import Path

from pipelines.dataset import FaceDataset
from pipelines.features import extract_bovw_features
from pipelines.train import (
    train_svm, hyperparameter_search, evaluate_model,
    save_models, plot_pr_curve, plot_roc_curve, plot_confusion_matrix
)
from pipelines.infer import FaceDetector
from pipelines.overlay import KumisOverlay
from pipelines.features import ORBFeatureExtractor, BoVWEncoder
from pipelines.utils import ensure_dir, save_image, load_image
import joblib
import json
import cv2


def command_train(args):
    """Training command."""
    print("\n" + "=" * 60)
    print("🎓 TRAINING KUMIS TRY-ON MODEL (SVM+ORB)")
    print("=" * 60)
    
    # Step 1: Load dataset
    print("\n📁 Step 1: Loading dataset...")
    dataset = FaceDataset(
        pos_dir=args.pos_dir,
        neg_dir=args.neg_dir,
        img_size=(128, 128)
    )
    
    if len(dataset.pos_samples) == 0 or len(dataset.neg_samples) == 0:
        print("❌ Error: Dataset empty! Please prepare face and non-face samples.")
        print(f"   Positive samples: {len(dataset.pos_samples)}")
        print(f"   Negative samples: {len(dataset.neg_samples)}")
        sys.exit(1)
    
    # Split dataset
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = dataset.split(
        train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
    )
    
    # Step 2: Extract features
    print("\n🔍 Step 2: Feature extraction (ORB + BoVW)...")
    X_train_bovw, y_train, orb_extractor, bovw_encoder = extract_bovw_features(
        X_train, y_train,
        k=args.k,
        max_descriptors=args.max_desc,
        nfeatures=args.nfeatures,
        verbose=True
    )
    
    # Save codebook
    ensure_dir(args.output_dir)
    codebook_path = Path(args.output_dir) / 'codebook.pkl'
    bovw_encoder.save(str(codebook_path))
    
    # Extract validation and test features
    from pipelines.features import ORBFeatureExtractor
    orb = ORBFeatureExtractor(nfeatures=args.nfeatures)
    
    print("\n  Extracting validation features...")
    X_val_desc = orb.extract_descriptors_batch(X_val, verbose=False)
    X_val_bovw = bovw_encoder.encode_batch(X_val_desc, verbose=False)
    
    print("  Extracting test features...")
    X_test_desc = orb.extract_descriptors_batch(X_test, verbose=False)
    X_test_bovw = bovw_encoder.encode_batch(X_test_desc, verbose=False)
    
    # Step 3: Train SVM
    print("\n🎓 Step 3: Training SVM classifier...")
    
    if args.search:
        # Hyperparameter search
        svm, scaler, best_params = hyperparameter_search(
            X_train_bovw, y_train, cv=args.cv, verbose=True
        )
    else:
        # Train with specified parameters
        svm, scaler = train_svm(
            X_train_bovw, y_train,
            kernel=args.svm,
            C=args.C,
            gamma=args.gamma,
            verbose=True
        )
        best_params = {'kernel': args.svm, 'C': args.C, 'gamma': args.gamma}
    
    # Step 4: Evaluate
    print("\n📊 Step 4: Evaluating model...")
    
    # Validation set
    print("\n  Validation Set:")
    val_metrics = evaluate_model(svm, scaler, X_val_bovw, y_val, verbose=True)
    
    # Test set
    print("\n  Test Set:")
    test_metrics = evaluate_model(svm, scaler, X_test_bovw, y_test, verbose=True)
    
    # Step 5: Save models and results
    print("\n💾 Step 5: Saving models...")
    
    config = {
        'k': args.k,
        'max_descriptors': args.max_desc,
        'nfeatures': args.nfeatures,
        'svm_kernel': best_params.get('kernel', args.svm),
        'C': best_params.get('C', args.C),
        'gamma': best_params.get('gamma', args.gamma),
        'val_metrics': val_metrics,
        'test_metrics': test_metrics
    }
    
    save_models(svm, scaler, str(codebook_path), config, args.output_dir)
    
    # Save metrics report
    ensure_dir('reports')
    with open('reports/training_results.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✅ Training complete!")
    print(f"  Models saved to: {args.output_dir}")
    print(f"  Test F1 Score: {test_metrics['f1_score']:.3f}")
    if 'roc_auc' in test_metrics:
        print(f"  Test ROC AUC: {test_metrics['roc_auc']:.3f}")


def command_eval(args):
    """Evaluation command."""
    print("\n" + "=" * 60)
    print("📊 EVALUATING MODEL")
    print("=" * 60)
    
    # Load models
    print("\n📦 Loading models...")
    svm = joblib.load(Path(args.model_dir) / 'svm.pkl')
    scaler = joblib.load(Path(args.model_dir) / 'scaler.pkl')
    codebook = joblib.load(Path(args.model_dir) / 'codebook.pkl')
    
    with open(Path(args.model_dir) / 'config.json', 'r') as f:
        config = json.load(f)
    
    print("  ✅ Models loaded")
    
    # Load test data (you need to have saved test split)
    # For now, just show config metrics
    print("\n📊 Model Configuration:")
    print(f"  Codebook size: {config['k']}")
    print(f"  SVM kernel: {config.get('svm_kernel', 'unknown')}")
    print(f"  C: {config.get('C', 'unknown')}")
    
    if 'test_metrics' in config:
        print("\n📈 Test Set Performance:")
        test_metrics = config['test_metrics']
        print(f"  Accuracy:  {test_metrics['accuracy']:.3f}")
        print(f"  Precision: {test_metrics['precision']:.3f}")
        print(f"  Recall:    {test_metrics['recall']:.3f}")
        print(f"  F1 Score:  {test_metrics['f1_score']:.3f}")
        if 'roc_auc' in test_metrics:
            print(f"  ROC AUC:   {test_metrics['roc_auc']:.3f}")
        
        # Plot confusion matrix
        if args.cm and 'confusion_matrix' in test_metrics:
            import numpy as np
            cm = np.array(test_metrics['confusion_matrix'])
            plot_confusion_matrix(cm, args.cm)
    
    # Save report
    if args.report:
        ensure_dir(Path(args.report).parent)
        with open(args.report, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"\n💾 Report saved to: {args.report}")


def command_infer(args):
    """Inference command."""
    print("\n" + "=" * 60)
    print("🔍 INFERENCE MODE")
    print("=" * 60)
    
    # Load models
    print("\n📦 Loading models...")
    from pipelines.train import load_models
    svm, scaler, codebook, config = load_models(args.model_dir)
    
    # Initialize detector
    orb_extractor = ORBFeatureExtractor(nfeatures=config.get('nfeatures', 500))
    bovw_encoder = BoVWEncoder(k=config['k'])
    bovw_encoder.kmeans = codebook
    
    face_detector = FaceDetector(
        svm=svm,
        scaler=scaler,
        bovw_encoder=bovw_encoder,
        orb_extractor=orb_extractor,
        confidence_threshold=0.5
    )
    
    # Load kumis overlay
    kumis_overlay = None
    if args.kumis:
        kumis_overlay = KumisOverlay(args.kumis)
    
    # Process image
    print(f"\n🖼️ Processing image: {args.image}")
    frame = load_image(args.image)
    
    if frame is None:
        print(f"  ❌ Cannot load image: {args.image}")
        sys.exit(1)
    
    # Detect faces
    print("  🔍 Detecting faces...")
    faces = face_detector.detect_faces(frame, nms_threshold=0.3)
    print(f"  ✅ Found {len(faces)} face(s)")
    
    # Overlay kumis
    if kumis_overlay and len(faces) > 0:
        print("  🎨 Overlaying kumis...")
        for face_data in faces:
            frame = kumis_overlay.overlay(
                frame,
                face_box=face_data['box'],
                eyes=face_data['eyes'],
                scale_factor=0.6,
                y_offset_factor=0.6
            )
    
    # Draw face boxes
    for face_data in faces:
        x, y, w, h = face_data['box']
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Save output
    ensure_dir(Path(args.out).parent)
    save_image(frame, args.out)
    print(f"\n💾 Output saved to: {args.out}")


def command_webcam(args):
    """Webcam mode (standalone, not UDP)."""
    print("\n" + "=" * 60)
    print("🎥 WEBCAM MODE (Press 'q' to quit)")
    print("=" * 60)
    
    # Load models
    print("\n📦 Loading models...")
    from pipelines.train import load_models
    svm, scaler, codebook, config = load_models(args.model_dir)
    
    # Initialize detector
    orb_extractor = ORBFeatureExtractor(nfeatures=config.get('nfeatures', 500))
    bovw_encoder = BoVWEncoder(k=config['k'])
    bovw_encoder.kmeans = codebook
    
    face_detector = FaceDetector(
        svm=svm,
        scaler=scaler,
        bovw_encoder=bovw_encoder,
        orb_extractor=orb_extractor,
        confidence_threshold=0.5
    )
    
    # Load kumis
    kumis_overlay = None
    if args.kumis:
        kumis_overlay = KumisOverlay(args.kumis)
    
    # Open webcam
    print(f"\n🎥 Opening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    
    if not cap.isOpened():
        print(f"  ❌ Cannot open camera {args.camera}")
        sys.exit(1)
    
    print("  ✅ Camera opened. Processing...")
    print("\n  Controls:")
    print("    'q' - Quit")
    print("    'h' - Toggle kumis on/off")
    
    show_kumis = True
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect faces
        faces = face_detector.detect_faces(frame, nms_threshold=0.3)
        
        # Overlay kumis
        if show_kumis and kumis_overlay and len(faces) > 0:
            for face_data in faces:
                frame = kumis_overlay.overlay(
                    frame,
                    face_box=face_data['box'],
                    eyes=face_data['eyes']
                )
        
        # Draw face boxes
        for face_data in faces:
            x, y, w, h = face_data['box']
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Show FPS
        cv2.putText(frame, f"Faces: {len(faces)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Kumis Try-On', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('h'):
            show_kumis = not show_kumis
            print(f"  Kumis: {'ON' if show_kumis else 'OFF'}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Webcam mode ended")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Kumis Try-On System - SVM+ORB Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train SVM model')
    train_parser.add_argument('--pos_dir', required=True, help='Directory with face samples')
    train_parser.add_argument('--neg_dir', required=True, help='Directory with non-face samples')
    train_parser.add_argument('--k', type=int, default=256, help='Codebook size (default: 256)')
    train_parser.add_argument('--max_desc', type=int, default=200000, help='Max descriptors for k-means')
    train_parser.add_argument('--nfeatures', type=int, default=500, help='ORB keypoints per image')
    train_parser.add_argument('--svm', default='linear', choices=['linear', 'rbf'], help='SVM kernel')
    train_parser.add_argument('--C', type=float, default=1.0, help='SVM C parameter')
    train_parser.add_argument('--gamma', default='scale', help='SVM gamma parameter')
    train_parser.add_argument('--search', action='store_true', help='Perform hyperparameter search')
    train_parser.add_argument('--cv', type=int, default=5, help='Cross-validation folds')
    train_parser.add_argument('--output_dir', default='models', help='Output directory for models')
    
    # Eval command
    eval_parser = subparsers.add_parser('eval', help='Evaluate trained model')
    eval_parser.add_argument('--model_dir', default='models', help='Models directory')
    eval_parser.add_argument('--report', help='Path to save metrics JSON')
    eval_parser.add_argument('--pr', help='Path to save PR curve')
    eval_parser.add_argument('--roc', help='Path to save ROC curve')
    eval_parser.add_argument('--cm', help='Path to save confusion matrix')
    
    # Infer command
    infer_parser = subparsers.add_parser('infer', help='Run inference on image')
    infer_parser.add_argument('--image', required=True, help='Input image path')
    infer_parser.add_argument('--out', required=True, help='Output image path')
    infer_parser.add_argument('--kumis', help='Kumis PNG file to overlay')
    infer_parser.add_argument('--model_dir', default='models', help='Models directory')
    
    # Webcam command
    webcam_parser = subparsers.add_parser('webcam', help='Run webcam mode')
    webcam_parser.add_argument('--camera', type=int, default=0, help='Camera device ID')
    webcam_parser.add_argument('--kumis', help='Kumis PNG file to overlay')
    webcam_parser.add_argument('--model_dir', default='models', help='Models directory')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        command_train(args)
    elif args.command == 'eval':
        command_eval(args)
    elif args.command == 'infer':
        command_infer(args)
    elif args.command == 'webcam':
        command_webcam(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
