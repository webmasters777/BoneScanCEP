#!/usr/bin/env python3
"""
Test ResNet50 model on both chestRANT and chestRPOST datasets
"""

from analysis import batch_classify_dataset

print('Testing ResNet50 model on both datasets...')
print('=' * 50)

# Test on RANT dataset
print('Testing on chestRANT dataset:')
rant_results = batch_classify_dataset('RANT', 100)
if rant_results:
    perf = rant_results['performance']
    print(f'  Accuracy: {perf["accuracy"]:.1%}')
    print(f'  Sensitivity: {perf["sensitivity"]:.1%}')
    print(f'  Specificity: {perf["specificity"]:.1%}')
    print(f'  Precision: {perf["precision"]:.1%}')
    print(f'  F1-Score: {perf["f1_score"]:.4f}')
    print(f'  Total Images: {rant_results["total_images"]}')
else:
    print('  Model not found!')

print()

# Test on RPOST dataset
print('Testing on chestRPOST dataset:')
rpost_results = batch_classify_dataset('RPOST', 100)
if rpost_results:
    perf = rpost_results['performance']
    print(f'  Accuracy: {perf["accuracy"]:.1%}')
    print(f'  Sensitivity: {perf["sensitivity"]:.1%}')
    print(f'  Specificity: {perf["specificity"]:.1%}')
    print(f'  Precision: {perf["precision"]:.1%}')
    print(f'  F1-Score: {perf["f1_score"]:.4f}')
    print(f'  Total Images: {rpost_results["total_images"]}')
else:
    print('  Model not found!')

print()
print('Summary:')
print('- Model was trained ONLY on chestRANT (anterior view) dataset')
print('- Model can be tested on both RANT and RPOST datasets')
print('- Both datasets have similar structure and size')