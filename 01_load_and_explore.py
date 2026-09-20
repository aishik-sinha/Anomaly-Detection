"""
STEP 1: Load and Explore the NSL-KDD Dataset
----------------------------------------------
Goal of this file: just get the data loaded into Python and look at it.
No modeling yet - we need to understand what we're working with first.

REFACTORED: now uses the shared preprocessing.py module for loading logic
instead of repeating it here - see preprocessing.py for the shared code.

Before running this:
1. Download KDDTrain+.txt from Kaggle or GitHub (see README)
2. Put it in the same folder as this script
3. Run: pip install -r requirements.txt
4. Run this file with: python 01_load_and_explore.py
"""

from preprocessing import load_and_clean

# ---------------------------------------------------------------------------
# 1. LOAD THE DATA
# ---------------------------------------------------------------------------
# load_and_clean() handles reading the raw file (which has no header row),
# applying our column names, and creating the simplified binary_label
# column (normal vs attack). See preprocessing.py if you want to see
# exactly how that works.
df = load_and_clean("KDDTrain+.txt")

# ---------------------------------------------------------------------------
# 2. BASIC SANITY CHECKS
# ---------------------------------------------------------------------------
# .shape tells us (number of rows, number of columns) - a quick gut check
# that the file loaded correctly.
print("Shape of dataset (rows, columns):", df.shape)

# .head() shows the first 5 rows so we can visually confirm it looks right.
print("\nFirst 5 rows:")
print(df.head())

# .dtypes shows what pandas thinks each column's data type is.
# We expect most to be numbers (int64/float64), but protocol_type, service,
# flag, and label should show up as 'object' (text) since they're words.
print("\nColumn data types:")
print(df.dtypes)

# ---------------------------------------------------------------------------
# 3. LOOK AT THE LABEL COLUMN - THIS IS THE MOST IMPORTANT PART
# ---------------------------------------------------------------------------
# .value_counts() counts how many rows fall into each category.
# This tells us: how many "normal" rows vs how many of each attack type.
print("\nLabel distribution (count of each attack type / normal):")
print(df["label"].value_counts())

# ---------------------------------------------------------------------------
# 4. CHECK THE SIMPLIFIED BINARY LABEL (created inside load_and_clean)
# ---------------------------------------------------------------------------
print("\nSimplified binary label distribution:")
print(df["binary_label"].value_counts())

# ---------------------------------------------------------------------------
# WHAT TO LOOK FOR WHEN YOU RUN THIS:
# ---------------------------------------------------------------------------
# - Shape should be roughly (125973, 43) for the full training set
# - You should see a rough split of ~53% normal / ~47% attack rows
# - protocol_type, service, flag, label, binary_label should be 'object' type
#   (everything else should be numeric) - this confirms what we'll need to
#   encode into numbers in the next step
