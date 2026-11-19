import os
from inference import main  # reuse your single-pair inference function

samples_dir = "data/samples"
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# find all T1 images
t1_images = sorted([f for f in os.listdir(samples_dir) if "_T1" in f])

for t1_file in t1_images:
    t2_file = t1_file.replace("_T1", "_T2")
    t1_path = os.path.join(samples_dir, t1_file)
    t2_path = os.path.join(samples_dir, t2_file)

    output_mask = os.path.join(output_dir, t1_file.replace("_T1", "_mask.png"))

    print(f"Processing {t1_file} and {t2_file}...")
    main(t1_path, t2_path, output_mask)

