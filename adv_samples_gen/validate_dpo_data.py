#!/usr/bin/env python3
"""
DPO Data Validation Script
验证生成的DPO数据质量，检测潜在问题
"""

import json
import sys
import os
from collections import defaultdict

def validate_dpo_jsonl(filepath):
    """验证DPO数据的质量"""
    print(f"\n{'='*70}")
    print(f"Validating: {filepath}")
    print(f"{'='*70}")

    if not os.path.exists(filepath):
        print(f"ERROR: File not found: {filepath}")
        return False

    issues = defaultdict(int)
    stats = {
        'total': 0,
        'valid': 0,
        'errors': []
    }

    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            stats['total'] += 1
            try:
                data = json.loads(line)

                # Check required fields
                required_fields = ['adv_truth', 'adv_prediction']
                missing_fields = [f for f in required_fields if f not in data]
                if missing_fields:
                    issues['missing_fields'] += 1
                    if len(stats['errors']) < 5:
                        stats['errors'].append(f"Line {line_num}: Missing fields {missing_fields}")
                    continue

                adv_truth = str(data['adv_truth']).strip()
                adv_prediction = str(data['adv_prediction']).strip()

                # Check 1: Empty values
                if not adv_truth or adv_truth == 'nan':
                    issues['empty_truth'] += 1
                    if len(stats['errors']) < 5:
                        stats['errors'].append(f"Line {line_num}: Empty adv_truth")
                    continue

                if not adv_prediction or adv_prediction == 'nan':
                    issues['empty_prediction'] += 1
                    if len(stats['errors']) < 5:
                        stats['errors'].append(f"Line {line_num}: Empty adv_prediction")
                    continue

                # Check 2: Identical chosen and rejected (CRITICAL BUG)
                if adv_truth == adv_prediction:
                    issues['same_truth_pred'] += 1
                    if len(stats['errors']) < 5:
                        stats['errors'].append(
                            f"Line {line_num}: adv_truth == adv_prediction "
                            f"(length={len(adv_truth)})"
                        )
                    continue

                # Check 3: Very short code (likely invalid)
                if len(adv_truth) < 20:
                    issues['too_short_truth'] += 1
                    if len(stats['errors']) < 5:
                        stats['errors'].append(f"Line {line_num}: adv_truth too short ({len(adv_truth)} chars)")

                if len(adv_prediction) < 20:
                    issues['too_short_pred'] += 1
                    if len(stats['errors']) < 5:
                        stats['errors'].append(f"Line {line_num}: adv_prediction too short ({len(adv_prediction)} chars)")

                # All checks passed
                stats['valid'] += 1

            except json.JSONDecodeError as e:
                issues['json_error'] += 1
                if len(stats['errors']) < 5:
                    stats['errors'].append(f"Line {line_num}: JSON decode error - {e}")
            except Exception as e:
                issues['other_error'] += 1
                if len(stats['errors']) < 5:
                    stats['errors'].append(f"Line {line_num}: Unexpected error - {e}")

    # Print results
    print(f"\nTotal samples: {stats['total']}")
    print(f"Valid samples: {stats['valid']} ({100*stats['valid']/stats['total']:.2f}%)")
    print(f"\nIssues found:")

    if not issues:
        print("  ✓ No issues detected!")
    else:
        for issue_type, count in sorted(issues.items(), key=lambda x: -x[1]):
            print(f"  ✗ {issue_type}: {count} ({100*count/stats['total']:.2f}%)")

    if stats['errors']:
        print(f"\nFirst {len(stats['errors'])} errors:")
        for error in stats['errors']:
            print(f"  - {error}")

    # Final verdict
    critical_issues = issues['same_truth_pred'] + issues['empty_truth'] + issues['empty_prediction']
    print(f"\n{'='*70}")
    if critical_issues == 0:
        print("✓ VALIDATION PASSED - Data quality is good!")
        return True
    else:
        print(f"✗ VALIDATION FAILED - {critical_issues} critical issues found")
        print("  Please regenerate DPO data with fixed scripts")
        return False


def compare_before_after(old_file, new_file):
    """比较修复前后的数据质量"""
    print(f"\n{'='*70}")
    print("BEFORE vs AFTER COMPARISON")
    print(f"{'='*70}")

    for label, filepath in [("BEFORE (old)", old_file), ("AFTER (new)", new_file)]:
        if os.path.exists(filepath):
            print(f"\n{label}: {filepath}")
            validate_dpo_jsonl(filepath)
        else:
            print(f"\n{label}: {filepath} - NOT FOUND")


if __name__ == '__main__':
    # Validate current DPO data
    dpo_files = [
        'train_dpo.jsonl',
        'valid_dpo.jsonl',
        'results/Qwen2.5-Coder-0.5B-Instruct/train_dpo.jsonl',
        'results/Qwen2.5-Coder-0.5B-Instruct/valid_dpo.jsonl'
    ]

    found_any = False
    for filepath in dpo_files:
        if os.path.exists(filepath):
            found_any = True
            validate_dpo_jsonl(filepath)

    if not found_any:
        print("ERROR: No DPO data files found. Please run read_dpo.py first.")
        sys.exit(1)
