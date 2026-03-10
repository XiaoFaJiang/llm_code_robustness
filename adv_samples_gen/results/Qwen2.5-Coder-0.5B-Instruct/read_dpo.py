import os
import pandas as pd
import json
import random

def read_adv_samples():
    perturbations = ["code_expression_exchange","code_stmt_exchange","code_style","insert","rename"]
    langs = ["cpp","python","java","javascript"]
    res = []

    # Statistics for filtering
    stats = {
        'total_rows': 0,
        'success_attacks': 0,
        'empty_adv_code': 0,
        'same_truth_pred': 0,
        'same_orig_adv': 0,
        'valid_samples': 0
    }

    for lang in langs:
        for p in perturbations:
            print(f"Processing {lang}/{p}.csv")
            try:
                df = pd.read_csv(os.path.join(lang,f"{p}.csv"))
            except FileNotFoundError:
                print(f"  Warning: File not found, skipping...")
                continue

            stats['total_rows'] += len(df)

            for index,row in df.iterrows():
                # Check if attack was successful
                if row["Is Success"] != 1:
                    continue
                stats['success_attacks'] += 1

                # Check if Adversarial Code is not empty
                adv_code = str(row.get("Adversarial Code", "")).strip()
                if not adv_code or adv_code == 'nan':
                    stats['empty_adv_code'] += 1
                    continue

                # Check if adv_truth != adv_prediction (Bug 2 fix)
                adv_truth = str(row.get("Adversarial truth", "")).strip()
                adv_prediction = str(row.get("Adv Prediction", "")).strip()
                if adv_truth == adv_prediction:
                    stats['same_truth_pred'] += 1
                    print(f"  Skipping row {index}: adv_truth == adv_prediction")
                    continue

                # Check if adversarial code is different from original (Bug 3 check)
                original_code = str(row.get("Original Code", "")).strip()
                if adv_code == original_code:
                    stats['same_orig_adv'] += 1
                    print(f"  Warning: row {index} has identical original and adversarial code")
                    # Still include it, as it might be edge case

                # All checks passed
                stats['valid_samples'] += 1
                res.append({'task_id':index,\
                    'original_code':row['Original Code'],\
                    'adv_code':adv_code,\
                    'adv_truth':adv_truth,\
                    'adv_prediction':adv_prediction,\
                    'lang':lang,\
                    'perturbation':p})
    
    # Print statistics
    print("\n" + "="*60)
    print("DATA FILTERING STATISTICS")
    print("="*60)
    print(f"Total CSV rows processed:     {stats['total_rows']}")
    print(f"Successful attacks (is_success=1): {stats['success_attacks']}")
    print(f"Filtered out - Empty adv_code:     {stats['empty_adv_code']}")
    print(f"Filtered out - Same truth/pred:    {stats['same_truth_pred']}")
    print(f"Warning - Same original/adv:       {stats['same_orig_adv']}")
    print(f"Valid samples for DPO:             {stats['valid_samples']}")
    print(f"Filtering rate: {100*(1-stats['valid_samples']/stats['total_rows']):.2f}%")
    print("="*60)

    if len(res) == 0:
        print("\nERROR: No valid samples found! Please check your CSV files.")
        return

    # Shuffle and split
    random.shuffle(res)
    n = len(res)
    train = res[:int(0.8*n)]
    valid = res[int(0.8*n):]

    print(f"\nTrain samples: {len(train)}")
    print(f"Valid samples: {len(valid)}")

    # Save to files
    with open("train_dpo.jsonl","w") as f:
        for oneline in train:
            f.write(json.dumps(oneline) + "\n")

    with open("valid_dpo.jsonl","w") as f:
        for oneline in valid:
            f.write(json.dumps(oneline) + "\n")

    print(f"\nSuccessfully generated:")
    print(f"  - train_dpo.jsonl ({len(train)} samples)")
    print(f"  - valid_dpo.jsonl ({len(valid)} samples)")

if __name__ == '__main__':
    read_adv_samples()