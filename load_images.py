import shutil
from pathlib import Path

root = Path("./")   # <-- promijeni
src_dir = root / "utkface_aligned_cropped"     # <-- ne diraj
dst_root = Path("./utk_ethnicity")             # <-- novi dataset

race_map = {"0":"white",
            "1":"black",
            "2":"asian",
            "3":"indian",
            "4":"other"}

for name in race_map.values():
    (dst_root / name).mkdir(parents=True, exist_ok=True)

total = copied = bad = dup = 0

for img_path in src_dir.rglob("*.jpg"):
    total += 1
    parts = img_path.stem.split("_")
    if len(parts) < 3:
        bad += 1
        continue
    race = parts[2]
    if race not in race_map:
        bad += 1
        continue

    dst = dst_root / race_map[race] / img_path.name
    if dst.exists():
        dup += 1
        continue

    shutil.copy2(img_path, dst)
    copied += 1

print("Found:", total, " Copied:", copied, " Bad:", bad, " Duplicates:", dup)
print("New dataset:", dst_root.resolve())