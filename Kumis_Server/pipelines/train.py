"""
SVM training module with hyperparameter optimization
"""

import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve
)
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns


def train_svm(X_train, y_train, kernel='linear', C=1.0, gamma='scale', verbose=True):
    """
    Train SVM classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        kernel: 'linear' or 'rbf'
        C: Regularization parameter
        gamma: Kernel coefficient (for RBF)
        verbose: Show training info
    
    Returns:
        trained_svm, scaler
    """
    print(f"\n🎓 Training SVM (kernel={kernel}, C={C})...")
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train SVM
    if kernel == 'linear':
        svm = LinearSVC(C=C, max_iter=10000, random_state=42)
    else:
        svm = SVC(C=C, kernel=kernel, gamma=gamma, probability=True, random_state=42)
    
    svm.fit(X_train_scaled, y_train)
    
    # Training accuracy
    train_pred = svm.predict(X_train_scaled)
    train_acc = accuracy_score(y_train, train_pred)
    
    if verbose:
        print(f"  ✅ Training complete!")
        print(f"  Train accuracy: {train_acc:.3f}")
    
    return svm, scaler


def hyperparameter_search(X_train, y_train, cv=5, verbose=True):
    """
    Grid search for optimal SVM hyperparameters.
    
    Args:
        X_train: Training features
        y_train: Training labels
        cv: Cross-validation folds
        verbose: Show progress
    
    Returns:
        best_svm, scaler, best_params
    """
    print(f"\n🔍 Hyperparameter search with {cv}-fold CV...")
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Parameter grid
    param_grid = {
        'C': [0.1, 1.0, 10.0],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }
    
    # Grid search
    svm = SVC(probability=True, random_state=42)
    grid_search = GridSearchCV(
        svm, param_grid,
        cv=cv,
        scoring='f1',
        verbose=2 if verbose else 0,
        n_jobs=-1
    )
    grid_search.fit(X_train_scaled, y_train)
    
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    print(f"\n  ✅ Best parameters: {best_params}")
    print(f"  Best CV F1 score: {best_score:.3f}")
    
    return grid_search.best_estimator_, scaler, best_params


def evaluate_model(svm, scaler, X_test, y_test, verbose=True):
    """
    Evaluate trained SVM on test set.
    
    Args:
        svm: Trained SVM model
        scaler: Feature scaler
        X_test: Test features
        y_test: Test labels
        verbose: Show detailed metrics
    
    Returns:
        Dictionary of metrics
    """
    print(f"\n📊 Evaluating model on test set...")
    
    # Scale features
    X_test_scaled = scaler.transform(X_test)
    
    # Predictions
    y_pred = svm.predict(X_test_scaled)
    
    # Probability scores (if available)
    if hasattr(svm, 'predict_proba'):
        y_proba = svm.predict_proba(X_test_scaled)[:, 1]
    elif hasattr(svm, 'decision_function'):
        y_proba = svm.decision_function(X_test_scaled)
    else:
        y_proba = None
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1)
    }
    
    # ROC AUC (if probability available)
    if y_proba is not None:
        auc = roc_auc_score(y_test, y_proba)
        metrics['roc_auc'] = float(auc)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    if verbose:
        print(f"\n  Test Results:")
        print(f"  ─────────────────────────────")
        print(f"  Accuracy:  {accuracy:.3f}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall:    {recall:.3f}")
        print(f"  F1 Score:  {f1:.3f}")
        if 'roc_auc' in metrics:
            print(f"  ROC AUC:   {auc:.3f}")
        print(f"\n  Confusion Matrix:")
        print(f"  {cm}")
        print(f"\n  Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Non-Face', 'Face']))
    
    return metrics


def save_models(svm, scaler, codebook_path, config, output_dir='models'):
    """
    Save trained models and configuration.
    
    Args:
        svm: Trained SVM
        scaler: Feature scaler
        codebook_path: Path to codebook
        config: Training configuration dict
        output_dir: Output directory
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Save SVM
    svm_path = os.path.join(output_dir, 'svm.pkl')
    joblib.dump(svm, svm_path)
    print(f"  💾 SVM saved to {svm_path}")
    
    # Save scaler
    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"  💾 Scaler saved to {scaler_path}")
    
    # Save config
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  💾 Config saved to {config_path}")


def load_models(model_dir='models'):
    """
    Load trained models.
    
    Args:
        model_dir: Directory with models
    
    Returns:
        svm, scaler, codebook, config
    """
    import os
    
    svm_path = os.path.join(model_dir, 'svm.pkl')
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    codebook_path = os.path.join(model_dir, 'codebook.pkl')
    config_path = os.path.join(model_dir, 'config.json')
    
    svm = joblib.load(svm_path)
    scaler = joblib.load(scaler_path)
    codebook = joblib.load(codebook_path)
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"✅ Models loaded from {model_dir}")
    
    return svm, scaler, codebook, config


def plot_pr_curve(y_test, y_proba, save_path=None):
    """Plot Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  📊 PR curve saved to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_roc_curve(y_test, y_proba, save_path=None):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  📊 ROC curve saved to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_confusion_matrix(cm, save_path=None):
    """Plot confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Face', 'Face'],
                yticklabels=['Non-Face', 'Face'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  📊 Confusion matrix saved to {save_path}")
    else:
        plt.show()
    plt.close()
