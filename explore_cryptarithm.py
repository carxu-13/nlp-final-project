"""Explore the theblackcat102/cryptarithm dataset from HuggingFace."""
from datasets import load_dataset

ds = load_dataset("theblackcat102/cryptarithm")

# Print available splits
print("=== Available splits ===")
print(list(ds.keys()))
print()

# Use the first available split
split_name = list(ds.keys())[0]
split = ds[split_name]

# Print column names and features
print("=== Column names ===")
print(split.column_names)
print()
print("=== Features ===")
print(split.features)
print()
print(f"=== Number of rows in '{split_name}' split ===")
print(len(split))
print()

# Print 10 sample rows
print("=== Sample rows (up to 10) ===")
for i in range(min(10, len(split))):
    print(f"\n--- Row {i} ---")
    for col in split.column_names:
        val = split[i][col]
        # Truncate long values for readability
        val_str = str(val)
        if len(val_str) > 500:
            val_str = val_str[:500] + "..."
        print(f"  {col}: {val_str}")
