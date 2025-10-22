import re
import csv
from pathlib import Path

# Target file(s)
markdown_files = list(Path(".").rglob("*.md"))

# Regex to match markdown table rows with scientific notation
row_pattern = re.compile(r"^\s*\|?\s*([0-9.eE+-]+)\s*\|([^|]+)\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|]+)\|?", re.M)

rows = []
for md_file in markdown_files:
    text = md_file.read_text()
    matches = row_pattern.findall(text)
    for m in matches:
        try:
            row = [float(x.replace("−", "-")) if re.match(r"[0-9.eE+-]+", x.strip()) else None for x in m]
            rows.append(row)
        except Exception:
            pass

# Deduplicate and sort by first column (error_rate)
rows = sorted(set(tuple(r) for r in rows), key=lambda x: x[0] if x[0] is not None else 0)

# Write out CSV
if rows:
    with open("pr_data_sample.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["error_rate", "steane", "surface", "reed_muller", "fusion_qec_photonic"])
        writer.writerows(rows)
    print(f"✅ Extracted {len(rows)} rows to pr_data_sample.csv")
else:
    print("⚠️  No numeric tables found in markdown files.")
