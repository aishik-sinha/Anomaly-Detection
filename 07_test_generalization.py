"""
EXPERIMENT: Test Generalization on the REAL Held-Out Dataset
------------------------------------------------------------------
Goal of this file: our earlier same-file 80/20 split doesn't fairly test
generalization. NSL-KDD ships a genuinely separate file, KDDTest+.txt,
which includes attack sub-types never seen in training - this is the real
test.

REFACTORED: now uses the shared preprocessing.py module instead of
repeating loading/encoding/scaling logic here.

Run this with: python 07_test_generalization.py
(You need BOTH KDDTrain+.txt AND KDDTest+.txt in this folder)
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from preprocessing import load_and_clean, encode_features, scale_features

# ---------------------------------------------------------------------------
# 1. LOAD AND PREPROCESS THE TRAINING FILE
# ---------------------------------------------------------------------------
train_df = load_and_clean("KDDTrain+.txt")
train_detailed_labels = train_df["label"]

train_df = train_df.drop(columns=["label", "difficulty"])
train_df, encoders = encode_features(train_df)  # fits new encoders

X_train = train_df.drop(columns=["binary_label"])
y_train = train_df["binary_label"]
feature_columns = list(X_train.columns)

X_train_scaled, scaler = scale_features(X_train)  # fits new scaler

print(f"Training on all {len(train_df)} rows from KDDTrain+.txt\n")

# ---------------------------------------------------------------------------
# 2. COMPUTE SAMPLE WEIGHTS BASED ON THE DETAILED ATTACK TYPE (not binary!)
# ---------------------------------------------------------------------------
# class_weight='balanced' only balances the binary label (already ~54/46,
# so little to correct). The real imbalance lives one level deeper, between
# individual attack sub-types (neptune: 41,214 rows vs guess_passwd: 53
# rows) that all collapse into a single "attack" label. sample_weight,
# computed from the DETAILED labels, lets the model feel that true rarity.
sample_weights = compute_sample_weight(class_weight="balanced", y=train_detailed_labels)

print("Sample weight range:", sample_weights.min(), "to", sample_weights.max())
print("(A rare attack type like guess_passwd should have a much higher")
print(" weight here than a common one like neptune)\n")

# ---------------------------------------------------------------------------
# 3. TRAIN THE MODEL
# ---------------------------------------------------------------------------
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train_scaled, y_train, sample_weight=sample_weights)

# ---------------------------------------------------------------------------
# 4. LOAD AND PREPROCESS THE SEPARATE TEST FILE
# ---------------------------------------------------------------------------
test_df = load_and_clean("KDDTest+.txt")
test_detailed_labels = test_df["label"]

test_df = test_df.drop(columns=["label", "difficulty"])

# Reuse the SAME encoders fit on training data - pass encoders= so
# encode_features runs in "reuse" mode (maps unseen categories to -1
# instead of crashing, since KDDTest+.txt was collected separately)
test_df, _ = encode_features(test_df, encoders=encoders)

X_test = test_df[feature_columns]
y_test = test_df["binary_label"]

# Reuse the SAME scaler fit on training data
X_test_scaled, _ = scale_features(X_test, scaler=scaler)

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
breakdown_df = pd.DataFrame({
    "true_detailed_label": test_detailed_labels.values,
    "predicted": y_pred
})
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
