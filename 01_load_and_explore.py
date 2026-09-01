"""
STEP 1: Load and Explore the NSL-KDD Dataset
----------------------------------------------
Goal of this file: just get the data loaded into Python and look at it.
No modeling yet - we need to understand what we're working with first.

Before running this:
1. Download KDDTrain+.txt from Kaggle or GitHub (see links from earlier in our chat)
2. Put it in the same folder as this script
3. Run: pip install pandas
4. Run this file with: python 01_load_and_explore.py
"""

import pandas as pd

# ---------------------------------------------------------------------------
# 1. DEFINE COLUMN NAMES
# ---------------------------------------------------------------------------
# The NSL-KDD file has NO header row - it's just raw comma-separated values.
# So we have to manually tell pandas what each of the 43 columns means.
# (41 features + 1 label column + 1 "difficulty" column some versions include)
column_names = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label",       # <-- this is the answer key: "normal" or attack type
    "difficulty"   # <-- extra column some versions include, we won't use it
]

# ---------------------------------------------------------------------------
# 2. LOAD THE CSV
# ---------------------------------------------------------------------------
# header=None tells pandas "there's no header row in the file, don't treat
# the first data row as column titles."
# names=column_names supplies the column titles we just defined above.
df = pd.read_csv("KDDTrain+.txt", header=None, names=column_names)

# ---------------------------------------------------------------------------
# 3. BASIC SANITY CHECKS
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
# 4. LOOK AT THE LABEL COLUMN - THIS IS THE MOST IMPORTANT PART
# ---------------------------------------------------------------------------
# .value_counts() counts how many rows fall into each category.
# This tells us: how many "normal" rows vs how many of each attack type.
print("\nLabel distribution (count of each attack type / normal):")
print(df["label"].value_counts())

# ---------------------------------------------------------------------------
# 5. CREATE A SIMPLIFIED BINARY LABEL: normal vs attack
# ---------------------------------------------------------------------------
# Right now "label" has ~22 different specific values (neptune, smurf, satan,
# normal, etc). For our FIRST model, we're simplifying this down to just two
# categories: "normal" or "attack". This is a common first step - once this
# works, you can go back and try predicting the specific attack type instead.
df["binary_label"] = df["label"].apply(lambda x: "normal" if x == "normal" else "attack")

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
