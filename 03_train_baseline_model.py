"""
STEP 3: Train a Baseline Classifier
--------------------------------------
Goal of this file: train our first real model and see how well it does.

We're starting with a Decision Tree because it's simple to understand:
it asks a series of yes/no questions about the data (e.g. "is serror_rate
> 0.5? if yes, is count > 100?") and follows a branching path down to a
final answer of "normal" or "attack".

REFACTORED: now uses the shared preprocessing.py module instead of
repeating the loading/encoding/scaling logic here.

Run this with: python 03_train_baseline_model.py
(Make sure KDDTrain+.txt is in the same folder, same as steps 1 and 2)
"""

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from preprocessing import load_and_clean, encode_features, scale_features

# ---------------------------------------------------------------------------
# 1. LOAD AND PREPROCESS
# ---------------------------------------------------------------------------
df = load_and_clean("KDDTrain+.txt")
df = df.drop(columns=["label", "difficulty"])
df, encoders = encode_features(df)

X = df.drop(columns=["binary_label"])
y = df["binary_label"]

X_scaled, scaler = scale_features(X)

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
#   learning (and also a fair limitation to mention in your README).
# - Pay special attention to the "False Negatives" number in the confusion
#   matrix (bottom-left) - these are real attacks the model MISSED.
#
# NEXT STEP: in 04_compare_models.py, we'll train a Random Forest and an
# Isolation Forest alongside this Decision Tree and compare all three.
