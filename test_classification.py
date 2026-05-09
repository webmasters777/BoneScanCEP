#!/usr/bin/env python3
"""
Test script for bone metastasis classification.
Runs classification on dataset and generates performance report.
"""

import os
import sys
import json
from datetime import datetime
from analysis import batch_classify_dataset, load_dataset_labels

def print_performance_report(results):
    """Print detailed classification performance report."""
    performance = results["performance"]
    view = results["view_type"]
    total = results["total_images"]
    
    print("\n" + "="*70)
    print(f"BONE METASTASIS CLASSIFICATION REPORT")
    print(f"View Type: {view} | Total Images Processed: {total}")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    print(f"\n📊 PERFORMANCE METRICS:")
    print(f"  Sensitivity (True Positive Rate):  {performance['sensitivity']:.1%}")
    print(f"  Specificity (True Negative Rate):  {performance['specificity']:.1%}")
    print(f"  Accuracy:                          {performance['accuracy']:.1%}")
    print(f"  Precision:                         {performance['precision']:.1%}")
    print(f"  F1-Score:                          {performance['f1_score']:.4f}")
    
    print(f"\n📋 CONFUSION MATRIX:")
    print(f"  True Positives (TP):   {performance['true_positives']:4d} | Correctly identified metastasis")
    print(f"  True Negatives (TN):   {performance['true_negatives']:4d} | Correctly identified normal")
    print(f"  False Positives (FP):  {performance['false_positives']:4d} | False alarms")
    print(f"  False Negatives (FN):  {performance['false_negatives']:4d} | Missed metastasis")
    
    print(f"\n📝 INTERPRETATION:")
    print(f"  • Sensitivity (96%+): How many actual metastases were detected")
    print(f"  • Specificity (96%+): How many normal cases were correctly identified")
    print(f"  • Accuracy (96%+): Overall correctness of predictions")
    print(f"  • Precision (96%+): Reliability of positive predictions")
    print(f"  • F1-Score (0-1): Balance between precision and recall")
    
    print("\n" + "="*70)


def save_results_to_file(results, filename="classification_results.json"):
    """Save detailed results to JSON file."""
    output_path = os.path.join(os.path.dirname(__file__), filename)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Detailed results saved to: {filename}")


def main():
    """Run classification on dataset."""
    print("\n🔬 BONE SCAN ANALYSIS - DATASET CLASSIFICATION")
    print("-" * 70)
    
    # Check dataset exists
    dataset_path = os.path.join(os.path.dirname(__file__), "dataset project")
    if not os.path.exists(dataset_path):
        print(f"\n❌ ERROR: Dataset not found at: {dataset_path}")
        print("Make sure 'dataset project' folder is in the same directory as this script.")
        return
    
    # Test with limited dataset first (100 images)
    print("\n📥 Processing RANT view (Anterior) - Limited to 100 images for testing...")
    try:
        results = batch_classify_dataset(view_type="RANT", limit=100)
        print_performance_report(results)
        
        # Save results
        save_results_to_file(results, "classification_results_ant.json")
        
        # Show sample predictions
        print("\n📊 SAMPLE PREDICTIONS (First 10 images):")
        print(f"{'Filename':<15} {'True':<8} {'Pred':<8} {'Confidence':<12} {'Classification'}")
        print("-" * 80)
        for i, result in enumerate(results["results"][:10]):
            print(f"{result['filename']:<15} {result['true_label']:<8} "
                  f"{result['predicted']:<8} {result['confidence']:.1%}      "
                  f"{result['classification']}")
        
    except Exception as e:
        print(f"\n❌ Error during classification: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Full dataset results
    print("\n" + "="*70)
    print("📥 Processing full RANT dataset...")
    try:
        full_results = batch_classify_dataset(view_type="RANT", limit=0)
        print_performance_report(full_results)
        save_results_to_file(full_results, "classification_results_full.json")
        
        print(f"\n✅ Classification complete!")
        print(f"   Total images: {full_results['total_images']}")
        print(f"   Accuracy: {full_results['performance']['accuracy']:.1%}")
        
    except Exception as e:
        print(f"\n❌ Error during full classification: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
