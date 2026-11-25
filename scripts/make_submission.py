"""Write the MILK10k leaderboard file from the saved 11-class model.

    python scripts/make_submission.py --test-images data/raw/MILK10k_Test_Input \
        --test-metadata data/raw/MILK10k_Test_Metadata.csv

Output: artifacts/final/submission.csv (479 rows x 12 columns: lesion + 11 classes, the argmax
class written as 1.0) and submission_probabilities.csv (the raw softmax, for analysis).
The format is checked (columns, rows, range, one class >= 0.5 per lesion) before writing.
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from milk10k import final

p = argparse.ArgumentParser()
p.add_argument("--test-images", required=True, help="folder with the 958 test images (any layout)")
p.add_argument("--test-metadata", required=True, help="MILK10k_Test_Metadata.csv")
p.add_argument("--model", default=str(final.MODEL_PATH))
p.add_argument("--out", default=str(final.RESULT_DIR / "submission.csv"))
a = p.parse_args()
sub = final.make_submission(a.test_images, a.test_metadata, a.out, model=final.FinalModel(resnet=final.load_model(a.model)))
print(f"{a.out}: {sub.shape[0]} lesions x {sub.shape[1]} columns")
print(sub.drop(columns="lesion").sum().astype(int).to_string())
