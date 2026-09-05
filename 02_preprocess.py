"""
STEP 2: Preprocess the Data (Encode + Scale + Split)
------------------------------------------------------
Goal of this file: take the raw loaded data from step 1 and turn it into
something a machine learning model can actually use.

Models only understand numbers. Right now we have 3 columns that are text
(protocol_type, service, flag) and features on wildly different scales
(some range 0-3, others range into the millions). This file fixes both.

Run this with: python 02_preprocess.py
(Make sure KDDTrain+.txt is in the same folder, same as step 1)
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# 1. LOAD THE DATA (same as step 1)
# ---------------------------------------------------------------------------
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
    "label", "difficulty"
]

df = pd.read_csv("KDDTrain+.txt", header=None, names=column_names)

# Create the simplified binary label again (normal vs attack) - same as step 1
df["binary_label"] = df["label"].apply(lambda x: "normal" if x == "normal" else "attack")

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
# LabelEncoder assigns each unique text value a number, e.g.:
#   "tcp" -> 0, "udp" -> 1, "icmp" -> 2
# We do this separately for each text column because each has its own
# set of unique values (protocol_type has 3 values, service has ~70).
categorical_columns = ["protocol_type", "service", "flag"]

encoders = {}  # we save each encoder in case we need to reverse it later
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

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
# Why: src_bytes might range into the tens of thousands, while
# num_failed_logins ranges 0-5. Without scaling, the model would wrongly
# think src_bytes matters way more just because its numbers are bigger.
# StandardScaler converts every column to have mean=0, standard deviation=1,
# putting all features on the same footing.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# X_scaled is now a plain numpy array (not a DataFrame) - that's normal,
# scikit-learn models are fine working with either.

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
