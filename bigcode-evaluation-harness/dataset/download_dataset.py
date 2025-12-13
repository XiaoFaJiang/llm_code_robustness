from datasets import load_dataset
languages = ["python","cpp","js","java"]

for lang in languages:

    ds = load_dataset("THUDM/humaneval-x", lang)
    ds["test"].to_json(f"humaneval_{lang}_tested.jsonl")
