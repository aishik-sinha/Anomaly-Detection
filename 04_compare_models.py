"""
STEP 4: Compare Models + Feature Importance
-----------------------------------------------------------------------------
Goal of this file: train multiple models, compare them, then use the
Random Forest's own feature importance ranking to see which features
actually matter - and test a simplified model using only the top ones.

REFACTORED:
- now uses the shared preprocessing.py module instead of repeating
  loading/encoding/scaling logic here
- saves a confusion matrix heatmap and a feature importance bar chart as
  PNG files, instead of only printing numbers to the terminal
- Isolation Forest's contamination parameter no longer uses the TRUE
  attack ratio computed from labels (that was a subtle form of data
  leakage - a real unsupervised setup wouldn't know this exact number).
  It now uses a reasonable default instead, and this file says so plainly.

Run this with: python 04_compare_models.py
(Make sure KDDTrain+.txt is in the same folder as the previous steps)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
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
feature_names = X.columns  # saved before scaling turns X into a plain array

X_scaled, scaler = scale_features(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print("Preprocessing complete.\n")

# ---------------------------------------------------------------------------
# HELPER FUNCTION: evaluate any model, print scores, save a confusion matrix
# ---------------------------------------------------------------------------
def evaluate_model(name, y_true, y_pred, save_plot_as=None):
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

    # Save a visual heatmap version of the confusion matrix, not just the
    # raw numbers - much easier to read at a glance, and something you
    # can actually drop into a README or a slide.
    if save_plot_as:
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["normal", "attack"],
                    yticklabels=["normal", "attack"])
        plt.title(f"Confusion Matrix - {name}")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.tight_layout()
        plt.savefig(save_plot_as)
        plt.close()
        print(f"Saved confusion matrix plot to {save_plot_as}\n")

    return {"model": name, "accuracy": accuracy, "precision": precision,
            "recall": recall, "f1": f1}

results = []

# ---------------------------------------------------------------------------
# 2. MODEL 1: DECISION TREE (our baseline from file 3, for comparison)
# ---------------------------------------------------------------------------
dt_model = DecisionTreeClassifier(max_depth=10, random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
results.append(evaluate_model("Decision Tree", y_test, dt_pred,
                               save_plot_as="confusion_matrix_decision_tree.png"))

# ---------------------------------------------------------------------------
# 3. MODEL 2: RANDOM FOREST
# ---------------------------------------------------------------------------
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
results.append(evaluate_model("Random Forest", y_test, rf_pred,
                               save_plot_as="confusion_matrix_random_forest.png"))

# ---------------------------------------------------------------------------
# 3b. FEATURE IMPORTANCE - ask the Random Forest which features mattered most
# ---------------------------------------------------------------------------
importances = rf_model.feature_importances_

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values("importance", ascending=False)

print("=== FEATURE IMPORTANCE RANKING (from Random Forest, all 40 features) ===")
print(importance_df.to_string(index=False))
print()

# Save a horizontal bar chart of the top 15 features - much easier to
# scan visually than a 40-row printed table.
plt.figure(figsize=(8, 6))
top_15 = importance_df.head(15).sort_values("importance")  # ascending for nicer barh order
plt.barh(top_15["feature"], top_15["importance"], color="#4C72B0")
plt.xlabel("Importance")
plt.title("Top 15 Feature Importances (Random Forest)")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()
print("Saved feature importance chart to feature_importance.png\n")

# Rebuild using only the top N features and compare against the full model.
TOP_N = 10
top_features = importance_df.head(TOP_N)["feature"].tolist()
print(f"Top {TOP_N} features selected:", top_features, "\n")

X_top = X[top_features]
X_top_scaled, _ = scale_features(X_top)

X_train_top, X_test_top, y_train_top, y_test_top = train_test_split(
    X_top_scaled, y, test_size=0.2, random_state=42, stratify=y
)

top_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
top_model.fit(X_train_top, y_train_top)
top_pred = top_model.predict(X_test_top)
results.append(evaluate_model(f"Random Forest (top {TOP_N} features only)", y_test_top, top_pred))

# ---------------------------------------------------------------------------
# 4. MODEL 3: ISOLATION FOREST (unsupervised - doesn't use y_train at all!)
# ---------------------------------------------------------------------------
# IMPORTANT CORRECTION: earlier versions of this file computed
# contamination from the TRUE attack ratio in y_train - but that's a
# subtle form of data leakage. A genuinely unsupervised setup wouldn't
# know the exact true ratio of anomalies in advance; that's the whole
# point of not having labels. Using the true ratio here was quietly
# giving Isolation Forest information it shouldn't have access to,
# making its results look better than a fair unsupervised setup would.
#
# We now use scikit-learn's default ('auto'), which is what you'd
# realistically use without label access. Expect this to look somewhat
# different (likely worse) than earlier runs - that's the more honest
# number.
print("NOTE: Isolation Forest below uses contamination='auto', NOT the true")
print("attack ratio - using the true ratio would leak label information into")
print("a model that's supposed to be unsupervised.\n")

iso_model = IsolationForest(contamination="auto", random_state=42)
iso_model.fit(X_train)  # only X_train - never sees y_train

iso_pred_raw = iso_model.predict(X_test)
iso_pred = pd.Series(iso_pred_raw).map({-1: "attack", 1: "normal"})

results.append(evaluate_model("Isolation Forest (unsupervised, contamination='auto')",
                               y_test, iso_pred,
                               save_plot_as="confusion_matrix_isolation_forest.png"))

# ---------------------------------------------------------------------------
# 5. SIDE-BY-SIDE COMPARISON TABLE
# ---------------------------------------------------------------------------
results_df = pd.DataFrame(results)
print("=== FINAL COMPARISON ===")
print(results_df.to_string(index=False))

# Also save a simple bar chart comparing F1 scores across all models
plt.figure(figsize=(7, 4))
plt.barh(results_df["model"], results_df["f1"], color="#55A868")
plt.xlabel("F1 Score")
plt.title("Model Comparison (F1 Score)")
plt.xlim(0, 1)
plt.tight_layout()
plt.savefig("model_comparison_f1.png")
plt.close()
print("\nSaved model comparison chart to model_comparison_f1.png")

# ---------------------------------------------------------------------------
# WHAT TO LOOK FOR WHEN YOU RUN THIS:
# ---------------------------------------------------------------------------
# - Random Forest should perform similarly to or slightly better than the
#   single Decision Tree.
# - Isolation Forest will likely perform noticeably worse now that it's
#   not leaking the true attack ratio - that's expected and more honest.
# - Compare "Random Forest" vs "Random Forest (top N features only)" - if
#   they're close, most of the 40 original columns aren't pulling their
#   weight.
# - Check the PNG files this script saves - these are now ready to drop
#   directly into your README instead of raw printed tables.
