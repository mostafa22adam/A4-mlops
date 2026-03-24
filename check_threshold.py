import os
import sys

# Read forced accuracy from GitHub Actions variable
accuracy = float(os.getenv("FORCE_ACCURACY", "0"))

print(f"Model accuracy: {accuracy}")

# Threshold check
if accuracy < 0.85:
    print("❌ Model failed threshold")
    sys.exit(1)
else:
    print("✅ Model passed threshold")