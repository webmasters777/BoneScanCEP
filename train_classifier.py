#!/usr/bin/env python3
"""
Improved classification using Random Forest.
Trains a supervised learning model on extracted features.
"""

import os
import json
import numpy as np
from datetime import datetime
from analysis import (
    iter_image_files, load_image_bgr, extract_metrics, 
    load_dataset_labels, evaluate_classification_performance
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import joblib

def extract_dataset_features():
    """
    Extract features from all images in dataset.
    
    Returns:
        tuple: (X, y, image_info) where X is features, y is labels
    """
    dataset_path = os.path.join(os.path.dirname(__file__), "dataset project")
    image_dir = os.path.join(dataset_path, "chestRANT")
    labels = load_dataset_labels(view_type="RANT")
    
    X = []  # Features
    y = []  # Labels
    image_info = []  # File info for reference
    
    print("Extracting features from dataset...")
    count = 0
    for filename in sorted(os.listdir(image_dir)):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        
        if filename not in labels:
            continue
        
        image_path = os.path.join(image_dir, filename)
        try:
            img_bgr = load_image_bgr(image_path)
            processed_data = extract_metrics(img_bgr)
            metrics = processed_data["metrics"]
            
            # Create feature vector
            feature_vector = [
                metrics['contrast'],
                metrics['energy'],
                metrics['homogeneity'],
                metrics['mean'],
                metrics['std'],
                metrics['median']
            ]
            
            X.append(feature_vector)
            y.append(labels[filename])
            image_info.append(filename)
            
            count += 1
            if count % 500 == 0:
                print(f"  Processed {count} images...")
        
        except Exception as e:
            print(f"  Skipped {filename}: {e}")
            continue
    
    print(f"✅ Extracted features from {count} images")
    return np.array(X), np.array(y), image_info


def train_classifier(X, y):
    """
    Train Random Forest classifier on extracted features.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Labels (n_samples,)
        
    Returns:
        tuple: (trained_model, scaler, train_results)
    """
    print("\n🔧 Training Random Forest Classifier...")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"  Training set: {len(X_train)} samples")
    print(f"  Test set: {len(X_test)} samples")
    
    # Train classifier
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'  # Handle class imbalance
    )
    
    clf.fit(X_train, y_train)
    print("✅ Model trained successfully")
    
    # Cross-validation
    cv_scores = cross_val_score(clf, X_scaled, y, cv=5, scoring='balanced_accuracy')
    print(f"  Cross-validation scores: {cv_scores}")
    print(f"  Mean CV accuracy: {cv_scores.mean():.1%} (+/- {cv_scores.std():.1%})")
    
    # Evaluate on test set
    train_accuracy = clf.score(X_train, y_train)
    test_accuracy = clf.score(X_test, y_test)
    
    print(f"\n📊 Model Performance:")
    print(f"  Training accuracy: {train_accuracy:.1%}")
    print(f"  Test accuracy: {test_accuracy:.1%}")
    
    # Feature importance
    feature_names = ['Contrast', 'Energy', 'Homogeneity', 'Mean', 'Std Dev', 'Median']
    importance = clf.feature_importances_
    
    print(f"\n📈 Feature Importance:")
    for name, imp in sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True):
        print(f"  {name:<15} {imp:.4f} ({imp*100:.1f}%)")
    
    # Detailed evaluation
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\n📋 Confusion Matrix (Test Set):")
    print(f"  True Negatives:  {cm[0,0]:4d} | False Positives: {cm[0,1]:4d}")
    print(f"  False Negatives: {cm[1,0]:4d} | True Positives:  {cm[1,1]:4d}")
    
    performance = evaluate_classification_performance(y_pred, y_test)
    print(f"\n✨ Test Set Performance Metrics:")
    print(f"  Sensitivity (Recall): {performance['sensitivity']:.1%}")
    print(f"  Specificity:          {performance['specificity']:.1%}")
    print(f"  Precision:            {performance['precision']:.1%}")
    print(f"  F1-Score:             {performance['f1_score']:.4f}")
    print(f"  Overall Accuracy:     {performance['accuracy']:.1%}")
    
    train_results = {
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "cv_scores": cv_scores.tolist(),
        "performance_test": performance,
        "feature_importance": dict(zip(feature_names, importance.tolist())),
        "confusion_matrix": cm.tolist()
    }
    
    return clf, scaler, train_results


def evaluate_full_dataset(clf, scaler, X, y, image_info):
    """
    Evaluate classifier on full dataset.
    """
    print("\n🔍 Evaluating on Full Dataset...")
    
    X_scaled = scaler.transform(X)
    y_pred = clf.predict(X_scaled)
    y_pred_proba = clf.predict_proba(X_scaled)
    
    performance = evaluate_classification_performance(y_pred, y)
    
    print(f"\n📊 Full Dataset Performance Metrics:")
    print(f"  Sensitivity: {performance['sensitivity']:.1%}")
    print(f"  Specificity: {performance['specificity']:.1%}")
    print(f"  Accuracy:    {performance['accuracy']:.1%}")
    print(f"  Precision:   {performance['precision']:.1%}")
    print(f"  F1-Score:    {performance['f1_score']:.4f}")
    
    # Save predictions
    results = []
    for i, (filename, true_label, pred, proba) in enumerate(
        zip(image_info, y, y_pred, y_pred_proba)
    ):
        results.append({
            "filename": filename,
            "true_label": int(true_label),
            "predicted": int(pred),
            "confidence_normal": float(proba[0]),
            "confidence_metastasis": float(proba[1])
        })
    
    return performance, results


def main():
    """Main execution."""
    print("\n" + "="*70)
    print("IMPROVED BONE METASTASIS CLASSIFICATION - RANDOM FOREST")
    print("="*70)
    
    # Extract features
    X, y, image_info = extract_dataset_features()
    
    print(f"\n📈 Dataset Summary:")
    print(f"  Total samples: {len(X)}")
    print(f"  Normal (0): {np.sum(y==0)} ({np.sum(y==0)/len(y)*100:.1f}%)")
    print(f"  Metastasis (1): {np.sum(y==1)} ({np.sum(y==1)/len(y)*100:.1f}%)")
    
    # Train classifier
    clf, scaler, train_results = train_classifier(X, y)
    
    # Evaluate on full dataset
    performance, predictions = evaluate_full_dataset(clf, scaler, X, y, image_info)
    
    # Save model
    model_path = os.path.join(os.path.dirname(__file__), "bone_metastasis_classifier.pkl")
    joblib.dump((clf, scaler), model_path)
    print(f"\n💾 Model saved to: {model_path}")
    
    # Save results
    results_file = os.path.join(os.path.dirname(__file__), "random_forest_results.json")
    results_data = {
        "timestamp": datetime.now().isoformat(),
        "training": train_results,
        "evaluation": {
            "performance": performance,
            "sample_predictions": predictions[:100]  # Save first 100
        }
    }
    
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"📄 Results saved to: {results_file}")
    
    print("\n" + "="*70)
    print("✅ Classification training complete!")
    print("="*70)


if __name__ == "__main__":
    main()
