"""
STEP 2: Preprocess the Data (Encode + Scale + Split)
------------------------------------------------------
Goal of this file: take the raw loaded data from step 1 and turn it into
something a machine learning model can actually use.

REFACTORED: now uses the shared preprocessing.py module instead of
repeating the loading/encoding/scaling logic here.

Run this with: python 02_preprocess.py
(Make sure KDDTrain+.txt is in the same folder, same as step 1)
"""

from sklearn.model_selection import train_test_split
from preprocessing import load_and_clean, encode_features, scale_features

# ---------------------------------------------------------------------------
# 1. LOAD THE DATA
# ---------------------------------------------------------------------------
df = load_and_clean("KDDTrain+.txt")
print("Loaded data shape:", df.shape)

# ---------------------------------------------------------------------------
# 2. DROP COLUMNS WE DON'T NEED FOR MODELING
# ---------------------------------------------------------------------------
# - "label" is the detailed attack type (22 categories) - we're using the
#   simpler "binary_label" instead for our first model.
# - "difficulty" is a scoring metadata column from the dataset creators,
#   not something that exists in real traffic - not useful for prediction.
df = df.drop(columns=["label", "difficulty"])

# ---------------------------------------------------------------------------
# 3. ENCODE THE TEXT COLUMNS INTO NUMBERS
# ---------------------------------------------------------------------------
# encode_features() converts protocol_type, service, and flag into numbers
# (e.g. "tcp" -> 0, "udp" -> 1). Called with no `encoders` argument, this
# FITS new encoders - appropriate here since this is our training data.
df, encoders = encode_features(df)

print("\nEncoded categorical columns. Example - protocol_type unique values now:")
print(df["protocol_type"].unique())

# ---------------------------------------------------------------------------
# 4. SEPARATE FEATURES (X) FROM THE LABEL (y)
# ---------------------------------------------------------------------------
# X = everything the model is allowed to look at when making a prediction
# y = the answer key we're trying to teach it to predict
X = df.drop(columns=["binary_label"])
y = df["binary_label"]

# ---------------------------------------------------------------------------
# 5. SCALE THE NUMERIC FEATURES
# ---------------------------------------------------------------------------
# scale_features() puts every column on the same footing (mean=0, std=1),
# so a large-range column like src_bytes doesn't dominate just because
# its raw numbers are bigger. Called with no `scaler` argument, this FITS
# a new scaler - appropriate here since this is training data.
X_scaled, scaler = scale_features(X)

print("\nFeatures scaled. Shape of feature matrix:", X_scaled.shape)

# ---------------------------------------------------------------------------
# 6. SPLIT INTO TRAINING AND TESTING SETS
# ---------------------------------------------------------------------------
# We train the model on one chunk of data, then test it on a separate chunk
# it has NEVER seen. This is how we get an honest measure of performance -
# testing on data the model already memorized would be cheating.
#
# test_size=0.2 means 20% of data held out for testing, 80% for training.
# random_state=42 just makes the split reproducible (same split every run).
# stratify=y ensures both sets have a similar ratio of normal/attack rows.
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTrain/test split complete:")
print("Training rows:", X_train.shape[0])
print("Testing rows:", X_test.shape[0])

# ---------------------------------------------------------------------------
# WHAT TO LOOK FOR WHEN YOU RUN THIS:
# ---------------------------------------------------------------------------
# - protocol_type unique values should now show as numbers like [0 1 2]
#   instead of ['tcp' 'udp' 'icmp']
# - Feature matrix shape should be (125973, 40) - same rows as before,
#   40 columns since we dropped label/difficulty/binary_label (43 - 3 = 40)
# - Training rows should be ~100,778 (80%) and testing ~25,195 (20%)
#
# NOTE: This file doesn't save anything yet - it just proves the pipeline
# works end to end. In the next file (03_train_baseline_model.py), we'll
# reuse this exact logic and then actually train a classifier on X_train/y_train.
