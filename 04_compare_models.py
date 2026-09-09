"""
STEP 4: Compare Models (Decision Tree vs Random Forest vs Isolation Forest)
-----------------------------------------------------------------------------
Goal of this file: train two more models and compare all three side by side.

1. Random Forest: instead of one Decision Tree, this trains MANY trees
   (each seeing a slightly different random slice of the data) and has
   them "vote" on the final answer. This usually performs better and is
   more robust than a single tree, at the cost of being harder to
   visually inspect.

2. Isolation Forest: completely different approach - this one is
   UNSUPERVISED, meaning it never even looks at the labels during
   training. Instead, it works by randomly splitting the data over and
   over; points that get isolated from the crowd in very few splits are
   flagged as anomalies. This mimics real-world security work where you
   often don't have labels for brand-new attack types.

Run this with: python 04_compare_models.py
(Make sure KDDTrain+.txt is in the same folder as the previous steps)
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

# ---------------------------------------------------------------------------
# 1. LOAD AND PREPROCESS (same as previous files)
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
df["binary_label"] = df["label"].apply(lambda x: "normal" if x == "normal" else "attack")
df = df.drop(columns=["label", "difficulty"])

categorical_columns = ["protocol_type", "service", "flag"]
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

X = df.drop(columns=["binary_label"])
y = df["binary_label"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print("Preprocessing complete.\n")

# ---------------------------------------------------------------------------
# HELPER FUNCTION: evaluate any model and return its scores
# ---------------------------------------------------------------------------
# We'll reuse this same evaluation logic for all three models so the
# comparison is fair and consistent.
def evaluate_model(name, y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label="attack")
    recall = recall_score(y_true, y_pred, pos_label="attack")
    f1 = f1_score(y_true, y_pred, pos_label="attack")
    cm = confusion_matrix(y_true, y_pred, labels=["normal", "attack"])

    print(f"--- {name} ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("Confusion Matrix (rows=actual, cols=predicted, order=[normal, attack]):")
    print(cm)
    print()

    # Return the scores so we can build a comparison table at the end
    return {"model": name, "accuracy": accuracy, "precision": precision,
            "recall": recall, "f1": f1}

results = []

# ---------------------------------------------------------------------------
# 2. MODEL 1: DECISION TREE (our baseline from file 3, for comparison)
# ---------------------------------------------------------------------------
dt_model = DecisionTreeClassifier(max_depth=10, random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
results.append(evaluate_model("Decision Tree", y_test, dt_pred))

# ---------------------------------------------------------------------------
# 3. MODEL 2: RANDOM FOREST
# ---------------------------------------------------------------------------
# n_estimators=100 means it builds 100 separate decision trees and averages
# their votes. More trees = generally more stable predictions, at the cost
# of more computation time.
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
results.append(evaluate_model("Random Forest", y_test, rf_pred))

# ---------------------------------------------------------------------------
# 4. MODEL 3: ISOLATION FOREST (unsupervised - doesn't use y_train at all!)
# ---------------------------------------------------------------------------
# contamination=0.46 tells the model roughly what % of the data we EXPECT
# to be anomalies. We can estimate this from our training data's actual
# attack ratio (roughly 46% attack in NSL-KDD) - in a real unlabeled
# scenario, you'd have to estimate this from domain knowledge instead.
attack_ratio = (y_train == "attack").mean()
print(f"Estimated contamination (attack ratio in training data): {attack_ratio:.3f}\n")

iso_model = IsolationForest(contamination=attack_ratio, random_state=42)

# Notice: .fit() only gets X_train, NOT y_train - it never sees the labels!
iso_model.fit(X_train)

# Isolation Forest predicts -1 for "anomaly" and 1 for "normal" - we convert
# those into our "attack"/"normal" labels so we can compare fairly.
iso_pred_raw = iso_model.predict(X_test)
iso_pred = pd.Series(iso_pred_raw).map({-1: "attack", 1: "normal"})

results.append(evaluate_model("Isolation Forest (unsupervised)", y_test, iso_pred))

# ---------------------------------------------------------------------------
# 5. SIDE-BY-SIDE COMPARISON TABLE
# ---------------------------------------------------------------------------
results_df = pd.DataFrame(results)
print("=== FINAL COMPARISON ===")
print(results_df.to_string(index=False))

# ---------------------------------------------------------------------------
# WHAT TO LOOK FOR WHEN YOU RUN THIS:
# ---------------------------------------------------------------------------
# - Random Forest should perform similarly to or slightly better than the
#   single Decision Tree - it's a more robust version of the same idea.
# - Isolation Forest will likely perform noticeably WORSE than the other
#   two. This is EXPECTED and worth understanding, not a bug: it never saw
#   the labels, so it's just guessing "which points look statistically
#   weird" rather than "which points match a known attack pattern." This
#   is exactly the tradeoff of unsupervised methods - lower accuracy, but
#   it works even when you have zero labeled examples of an attack.
#
# TALKING POINT FOR YOUR README/INTERVIEW:
# "Isolation Forest performed worse than the supervised models, which
# makes sense - it has no access to labels and is purely looking for
# statistical outliers. This tradeoff is valuable in the real world
# though, since new/unknown attack types won't have labeled training
# examples yet, and only an unsupervised approach could catch them."
#
# NEXT STEP: 05_scoring_tool.py - build a script/tool that takes a CSV of
# new, unlabeled traffic and prints out which rows look suspicious using
# our best-performing trained model.
