"""
EXPERIMENT: Test Generalization on the REAL Held-Out Dataset
------------------------------------------------------------------
Goal of this file: address a legitimate concern - our earlier "test set"
was just a random 20% slice of KDDTrain+.txt. That's a fair test of
whether the model memorized individual ROWS, but NOT a fair test of
whether it generalizes to genuinely different data, since it came from
the exact same file/session/distribution.

NSL-KDD ships with a SEPARATE file, KDDTest+.txt, collected independently
and deliberately containing some attack sub-types that never appear in
KDDTrain+.txt at all. This is the real test of generalization.

We will:
1. Train on ALL of KDDTrain+.txt (not just 80% of it - use every row,
   since we're now testing on a totally separate file anyway)
2. Test on KDDTest+.txt
3. Compare these results against our earlier same-file test results
4. Break results down by individual attack type, not just binary
   normal/attack, since rare attack types can hide inside a good-looking
   overall score

Run this with: python 07_test_generalization.py
(You need BOTH KDDTrain+.txt AND KDDTest+.txt in this folder - download
KDDTest+.txt from the same source you got KDDTrain+.txt from)
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

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

# ---------------------------------------------------------------------------
# 1. LOAD AND PREPROCESS THE TRAINING FILE
# ---------------------------------------------------------------------------
train_df = pd.read_csv("KDDTrain+.txt", header=None, names=column_names)
train_df["binary_label"] = train_df["label"].apply(lambda x: "normal" if x == "normal" else "attack")

# Keep the detailed attack-type labels aside before we drop them - we'll
# use these later to break down performance by specific attack type.
train_detailed_labels = train_df["label"]

train_df = train_df.drop(columns=["label", "difficulty"])

# Fit encoders on TRAINING data only - this matters. If we fit on
# train+test combined, we'd be "leaking" information about the test set's
# categories into training, which would be a subtle form of cheating.
encoders = {}
categorical_columns = ["protocol_type", "service", "flag"]
for col in categorical_columns:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    encoders[col] = le

X_train = train_df.drop(columns=["binary_label"])
y_train = train_df["binary_label"]
feature_columns = list(X_train.columns)

# Fit the scaler on TRAINING data only, same reasoning as above.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

print(f"Training on all {len(train_df)} rows from KDDTrain+.txt\n")

# ---------------------------------------------------------------------------
# 2. COMPUTE SAMPLE WEIGHTS BASED ON THE DETAILED ATTACK TYPE (not binary!)
# ---------------------------------------------------------------------------
# We already tried class_weight="balanced" and found it didn't help. The
# reason: it only balances the binary label (normal/attack), which is
# already roughly 54/46 - already fairly balanced, so there was barely
# anything for it to correct. The REAL imbalance is hidden one level
# deeper - between individual attack sub-types that all get collapsed
# into a single "attack" label before training (neptune: 41,214 rows vs
# guess_passwd: 53 rows).
#
# compute_sample_weight("balanced", y=...) gives each ROW an individual
# weight based on how rare ITS class is, in whatever labels we give it.
# By passing train_detailed_labels (the original ~22 attack sub-types,
# not the collapsed binary version), a guess_passwd row gets a much
# higher weight than a neptune row - even though both simply become
# "attack" in y_train. This lets the model "feel" the true rarity of
# each attack sub-type during training, even while still only predicting
# the binary normal/attack label at the end.
sample_weights = compute_sample_weight(class_weight="balanced", y=train_detailed_labels)

print("Sample weight range:", sample_weights.min(), "to", sample_weights.max())
print("(A rare attack type like guess_passwd should have a much higher")
print(" weight here than a common one like neptune)\n")

# ---------------------------------------------------------------------------
# 3. TRAIN THE MODEL
# ---------------------------------------------------------------------------
# Note: class_weight is back to its default (None) here - we don't want to
# stack two different balancing schemes on top of each other, since that
# would make the effect unpredictable. sample_weight alone now carries the
# full responsibility of addressing the imbalance we found.
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train_scaled, y_train, sample_weight=sample_weights)

# ---------------------------------------------------------------------------
# 4. LOAD AND PREPROCESS THE SEPARATE TEST FILE
# ---------------------------------------------------------------------------
test_df = pd.read_csv("KDDTest+.txt", header=None, names=column_names)
test_df["binary_label"] = test_df["label"].apply(lambda x: "normal" if x == "normal" else "attack")

# Keep detailed labels for the per-attack-type breakdown later
test_detailed_labels = test_df["label"]

test_df = test_df.drop(columns=["label", "difficulty"])

# IMPORTANT: use .transform(), NOT .fit_transform(), and reuse the SAME
# encoders fit on training data. Also handle the case where the test file
# contains a category never seen during training (map to -1) - this can
# genuinely happen since KDDTest+.txt was collected separately.
for col, encoder in encoders.items():
    test_df[col] = test_df[col].apply(
        lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
    )

X_test = test_df[feature_columns]
y_test = test_df["binary_label"]

# Reuse the SAME scaler fit on training data - do not re-fit here.
X_test_scaled = scaler.transform(X_test)

print(f"Testing on all {len(test_df)} rows from KDDTest+.txt (a genuinely separate file/session)\n")

# ---------------------------------------------------------------------------
# 5. EVALUATE
# ---------------------------------------------------------------------------
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label="attack")
recall = recall_score(y_test, y_pred, pos_label="attack")
f1 = f1_score(y_test, y_pred, pos_label="attack")
cm = confusion_matrix(y_test, y_pred, labels=["normal", "attack"])

print("=== PERFORMANCE ON GENUINELY SEPARATE KDDTest+.txt ===")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print("Confusion Matrix (rows=actual, cols=predicted, order=[normal, attack]):")
print(cm)

print("\n--- Comparison across everything we've tried on KDDTest+.txt ---")
print("No balancing at all:              Accuracy 0.7694, Recall 0.6154, F1 0.7524")
print("class_weight='balanced' (binary): Accuracy 0.7622, Recall 0.6027, F1 0.7426  (didn't help - wrong level)")
print(f"sample_weight (detailed types):   Accuracy {accuracy:.4f}, Recall {recall:.4f}, F1 {f1:.4f}  (this run)")
print("Watch recall in particular, and check if guess_passwd's catch rate below moved off 0.000.")

# ---------------------------------------------------------------------------
# 6. HOW MANY ATTACK TYPES IN THE TEST FILE WERE NEVER SEEN IN TRAINING?
# ---------------------------------------------------------------------------
train_attack_types = set(train_detailed_labels.unique())
test_attack_types = set(test_detailed_labels.unique())
unseen_attack_types = test_attack_types - train_attack_types

print(f"\nAttack types in training data: {len(train_attack_types)}")
print(f"Attack types in test data: {len(test_attack_types)}")
print(f"Attack types in TEST but NEVER seen in TRAINING: {len(unseen_attack_types)}")
print(unseen_attack_types)

# ---------------------------------------------------------------------------
# 7. PER-ATTACK-TYPE BREAKDOWN
# ---------------------------------------------------------------------------
# This is the important part: overall accuracy can look great while
# hiding total failure on specific attack types, especially ones the
# model has never seen a single example of during training.
breakdown_df = pd.DataFrame({
    "true_detailed_label": test_detailed_labels.values,
    "predicted": y_pred
})
# A row counts as "caught" if the model predicted "attack" for anything
# that wasn't actually "normal"
breakdown_df["was_attack"] = breakdown_df["true_detailed_label"] != "normal"
breakdown_df["caught"] = (breakdown_df["predicted"] == "attack") & breakdown_df["was_attack"]

per_type = breakdown_df[breakdown_df["was_attack"]].groupby("true_detailed_label").agg(
    total_rows=("caught", "count"),
    caught=("caught", "sum")
)
per_type["catch_rate"] = (per_type["caught"] / per_type["total_rows"]).round(3)
per_type["seen_in_training"] = per_type.index.isin(train_attack_types)
per_type = per_type.sort_values("catch_rate")

print("\n=== CATCH RATE PER ATTACK TYPE (worst first) ===")
print(per_type.to_string())

# ---------------------------------------------------------------------------
# WHAT TO LOOK FOR WHEN YOU RUN THIS:
# ---------------------------------------------------------------------------
# - Compare the three-way printout above. Did sample_weight (based on
#   detailed attack sub-type) do better than class_weight='balanced'
#   (based on the collapsed binary label) at improving recall?
# - Specifically check guess_passwd's catch_rate in the table below - it
#   was 0.000 in BOTH earlier attempts despite having 1,231 training
#   examples. If sample_weight is working as intended, this is the
#   clearest single number to prove it moved.
# - Attack types with seen_in_training=False will still likely stay near
#   0.000 no matter what we do here - that's expected and not fixable by
#   reweighting, since the model has never seen an example resembling
#   them at all. Reweighting can only help with RARE-but-present classes,
#   not ABSENT ones.
#
# TALKING POINT FOR YOUR README/INTERVIEW:
# "I diagnosed that class_weight='balanced' failed because it only
# balances the binary normal/attack label, which was already roughly
# balanced - the real imbalance was hidden one level deeper, between
# individual attack sub-types collapsed into that single 'attack' label.
# Switching to per-row sample_weight computed from the original ~22
# attack sub-type distribution [improved recall on previously-missed
# rare attack types / still didn't fully solve it], which highlights that
# some of this gap is a fundamental limitation of supervised learning -
# no amount of reweighting can teach a model to recognize an attack type
# it has zero training examples of."
