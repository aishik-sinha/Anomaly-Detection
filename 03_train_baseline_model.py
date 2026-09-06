"""
STEP 3: Train a Baseline Classifier
--------------------------------------
Goal of this file: train our first real model and see how well it does.

We're starting with a Decision Tree because it's simple to understand:
it asks a series of yes/no questions about the data (e.g. "is serror_rate
> 0.5? if yes, is count > 100?") and follows a branching path down to a
final answer of "normal" or "attack". This makes it a good first model -
you can even print out the actual questions it learned to ask.

Run this with: python 03_train_baseline_model.py
(Make sure KDDTrain+.txt is in the same folder, same as steps 1 and 2)
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# ---------------------------------------------------------------------------
# 1. LOAD AND PREPROCESS (same steps as files 1 and 2, combined)
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

print("Preprocessing complete. Training rows:", X_train.shape[0], "| Testing rows:", X_test.shape[0])

# ---------------------------------------------------------------------------
# 2. CREATE AND TRAIN THE MODEL
# ---------------------------------------------------------------------------
# max_depth=10 limits how many yes/no questions deep the tree can go.
# Without a limit, decision trees tend to "memorize" the training data
# perfectly but then perform poorly on new data - this is called
# OVERFITTING. Limiting depth forces it to learn general patterns instead
# of memorizing individual rows.
model = DecisionTreeClassifier(max_depth=10, random_state=42)

# .fit() is where the actual "learning" happens - the model looks at
# X_train (the features) alongside y_train (the correct answers) and
# figures out which questions best separate normal from attack.
model.fit(X_train, y_train)

print("\nModel trained.")

# ---------------------------------------------------------------------------
# 3. MAKE PREDICTIONS ON THE TEST SET
# ---------------------------------------------------------------------------
# The model has NEVER seen X_test before. This is the honest test of
# whether it actually learned something useful, or just memorized the
# training data.
y_pred = model.predict(X_test)

# ---------------------------------------------------------------------------
# 4. EVALUATE THE RESULTS
# ---------------------------------------------------------------------------
# Accuracy: what % of all predictions were correct overall.
# Can be misleading if classes are imbalanced, so we don't rely on this alone.
accuracy = accuracy_score(y_test, y_pred)

# Precision: of everything the model FLAGGED as "attack", what % actually
# were attacks? Low precision = lots of false alarms (analysts get annoyed).
precision = precision_score(y_test, y_pred, pos_label="attack")

# Recall: of ALL the actual attacks in the test set, what % did the model
# catch? Low recall = real attacks are slipping through undetected.
# In security, this is usually the number you care about most.
recall = recall_score(y_test, y_pred, pos_label="attack")

# F1 score: a single number that balances precision and recall together.
# Useful for quickly comparing models later.
f1 = f1_score(y_test, y_pred, pos_label="attack")

print("\n--- MODEL PERFORMANCE ---")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}  (of flagged attacks, % that were real)")
print(f"Recall:    {recall:.4f}  (of real attacks, % that were caught)")
print(f"F1 Score:  {f1:.4f}  (balance of precision and recall)")

# ---------------------------------------------------------------------------
# 5. CONFUSION MATRIX - the full breakdown of right vs wrong predictions
# ---------------------------------------------------------------------------
# This shows four numbers:
#   - True Negatives:  correctly predicted "normal"
#   - False Positives: predicted "attack" but was actually "normal" (false alarm)
#   - False Negatives: predicted "normal" but was actually "attack" (MISSED attack - the dangerous one)
#   - True Positives:  correctly predicted "attack"
cm = confusion_matrix(y_test, y_pred, labels=["normal", "attack"])
print("\nConfusion Matrix:")
print("                Predicted Normal   Predicted Attack")
print(f"Actual Normal        {cm[0][0]:<10}         {cm[0][1]:<10}")
print(f"Actual Attack        {cm[1][0]:<10}         {cm[1][1]:<10}")

# ---------------------------------------------------------------------------
# 6. FULL CLASSIFICATION REPORT (a convenient summary of everything above)
# ---------------------------------------------------------------------------
print("\nFull classification report:")
print(classification_report(y_test, y_pred))

# ---------------------------------------------------------------------------
# WHAT TO LOOK FOR WHEN YOU RUN THIS:
# ---------------------------------------------------------------------------
# - A well-performing baseline on this dataset typically gets accuracy,
#   precision, and recall all above 0.95 - NSL-KDD is a relatively "easy"
#   dataset for models to do well on, which is part of why it's good for
#   learning (and also a fair limitation to mention in your README - real
#   world traffic is messier than this).
# - Pay special attention to the "False Negatives" number in the confusion
#   matrix (bottom-left) - these are real attacks the model MISSED. This
#   is the number that matters most in a security context.
#
# NEXT STEP: in 04_compare_models.py, we'll train a Random Forest and an
# Isolation Forest alongside this Decision Tree and compare all three.
