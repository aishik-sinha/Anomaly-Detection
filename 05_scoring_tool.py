"""
STEP 5: The Scoring Tool
---------------------------
Goal of this file: this is the actual "product" - a tool that trains once
on known, labeled data, then can score ANY new CSV of traffic and tell you
which rows look suspicious.

REFACTORED: now uses the shared preprocessing.py module instead of
repeating loading/encoding/scaling logic here.

Uses sample_weight (based on detailed attack sub-type) when training -
validated in 07_test_generalization.py to meaningfully improve recall on
rare-but-seen attack types like guess_passwd.

Run this with: python 05_scoring_tool.py <path_to_new_data.csv>
Example:       python 05_scoring_tool.py KDDTest+.txt
"""

import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
from preprocessing import load_and_clean, encode_features, scale_features


def train_model():
    df = load_and_clean("KDDTrain+.txt")

    # Keep the ORIGINAL detailed attack-type labels before we drop them -
    # needed for sample_weight, since the collapsed binary label hides
    # the real class imbalance (see 07_test_generalization.py).
    detailed_labels = df["label"]

    df = df.drop(columns=["label", "difficulty"])
    df, encoders = encode_features(df)

    X = df.drop(columns=["binary_label"])
    y = df["binary_label"]

    X_scaled, scaler = scale_features(X)

    sample_weights = compute_sample_weight(class_weight="balanced", y=detailed_labels)

    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_scaled, y, sample_weight=sample_weights)

    print("Model trained on", len(df), "labeled rows (using sample_weight to address class imbalance).\n")

    return model, encoders, scaler, list(X.columns)


def prepare_new_data(filepath, encoders, scaler, feature_columns):
    new_df = load_and_clean(filepath)

    true_labels = new_df["binary_label"]  # available since NSL-KDD test files include labels too

    new_df = new_df.drop(columns=["label", "difficulty", "binary_label"], errors="ignore")

    # Reuse the SAME encoders fit on training data (pass encoders= so
    # encode_features runs in "reuse" mode, not "fit new" mode)
    new_df, _ = encode_features(new_df, encoders=encoders)

    new_df = new_df[feature_columns]

    # Reuse the SAME scaler fit on training data
    X_new_scaled, _ = scale_features(new_df, scaler=scaler)

    return X_new_scaled, true_labels, new_df


def score_and_report(model, X_new_scaled, true_labels, raw_df):
    predictions = model.predict(X_new_scaled)

    probabilities = model.predict_proba(X_new_scaled)
    attack_column_index = list(model.classes_).index("attack")
    attack_confidence = probabilities[:, attack_column_index]

    results = raw_df.copy()
    results["predicted"] = predictions
    results["attack_confidence"] = attack_confidence

    suspicious = results[results["predicted"] == "attack"].sort_values(
        "attack_confidence", ascending=False
    )

    print(f"Scored {len(results)} total rows.")
    print(f"Flagged {len(suspicious)} rows as suspicious ({len(suspicious)/len(results)*100:.1f}%).\n")

    print("Top 10 most suspicious rows (highest confidence first):")
    print(suspicious[["duration", "count", "serror_rate", "num_failed_logins",
                       "attack_confidence"]].head(10).to_string())

    if true_labels is not None:
        correct = (predictions == true_labels).sum()
        print(f"\n(For reference, since this test file has true labels: "
              f"{correct}/{len(true_labels)} correct = {correct/len(true_labels)*100:.2f}% accuracy)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python 05_scoring_tool.py <path_to_new_data.csv>")
        print("Example: python 05_scoring_tool.py KDDTest+.txt")
        sys.exit(1)

    new_data_path = sys.argv[1]

    model, encoders, scaler, feature_columns = train_model()
    X_new_scaled, true_labels, raw_df = prepare_new_data(
        new_data_path, encoders, scaler, feature_columns
    )
    score_and_report(model, X_new_scaled, true_labels, raw_df)
