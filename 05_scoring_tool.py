"""
STEP 5: The Scoring Tool
---------------------------
Goal of this file: this is the actual "product" - a tool that trains once
on known, labeled data, then can score ANY new CSV of traffic and tell you
which rows look suspicious. This is what you'd demo in an interview.

How to use it:
    python 05_scoring_tool.py KDDTest+.txt

It will:
1. Train the Random Forest model on KDDTrain+.txt (our known, labeled data)
2. Load whatever file you passed in as the second argument (new traffic)
3. Print out which rows it thinks are suspicious, with a confidence score

Run this with: python 05_scoring_tool.py <path_to_new_data.csv>
Example:       python 05_scoring_tool.py KDDTest+.txt
"""

import sys
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# ---------------------------------------------------------------------------
# COLUMN NAMES (same as every previous file - NSL-KDD has no header row)
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

# ---------------------------------------------------------------------------
# STEP A: TRAIN THE MODEL ON KNOWN DATA
# ---------------------------------------------------------------------------
# This function bundles up everything from files 2-4 into one reusable step:
# load the labeled training data, clean it, and train a Random Forest on it.
def train_model():
    df = pd.read_csv("KDDTrain+.txt", header=None, names=column_names)
    df["binary_label"] = df["label"].apply(lambda x: "normal" if x == "normal" else "attack")
    df = df.drop(columns=["label", "difficulty"])

    # We need to save the encoders and scaler because we'll have to apply
    # the EXACT SAME transformations to new data later - a model trained
    # on scaled/encoded data can't understand raw text or unscaled numbers.
    encoders = {}
    categorical_columns = ["protocol_type", "service", "flag"]
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df.drop(columns=["binary_label"])
    y = df["binary_label"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_scaled, y)

    print("Model trained on", len(df), "labeled rows.\n")

    # Return everything we'll need to process new data later
    return model, encoders, scaler, list(X.columns)


# ---------------------------------------------------------------------------
# STEP B: PREPARE NEW DATA THE SAME WAY WE PREPARED TRAINING DATA
# ---------------------------------------------------------------------------
# CRITICAL CONCEPT: new data must go through the EXACT SAME preprocessing
# as training data (same encoders, same scaler) - otherwise the numbers
# won't mean the same thing to the model, and predictions will be garbage.
def prepare_new_data(filepath, encoders, scaler, feature_columns):
    # We assume the new file has the same 43-column format (including a
    # label column, since NSL-KDD test files do include true labels too -
    # in a REAL deployment, new traffic wouldn't have a label column at
    # all, and you'd skip the "true_labels" parts of this function).
    new_df = pd.read_csv(filepath, header=None, names=column_names)

    # Keep the true labels aside (if present) so we can show how well we
    # did - purely for our own learning purposes, not something a real
    # deployed tool would have access to.
    true_labels = None
    if "label" in new_df.columns:
        true_labels = new_df["label"].apply(lambda x: "normal" if x == "normal" else "attack")

    new_df = new_df.drop(columns=["label", "difficulty"], errors="ignore")

    # Apply the SAME encoders that were fit on training data.
    # Using .transform() (not .fit_transform()) is important here - we're
    # reusing the categories the model already learned, not creating new ones.
    for col, encoder in encoders.items():
        # handle_unknown: if new data has a category never seen in training
        # (e.g. a brand new service name), map it to -1 rather than crashing
        new_df[col] = new_df[col].apply(
            lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
        )

    # Make sure columns are in the same order the model expects
    new_df = new_df[feature_columns]

    # Apply the SAME scaler fit on training data
    X_new_scaled = scaler.transform(new_df)

    return X_new_scaled, true_labels, new_df


# ---------------------------------------------------------------------------
# STEP C: SCORE THE NEW DATA AND REPORT SUSPICIOUS ROWS
# ---------------------------------------------------------------------------
def score_and_report(model, X_new_scaled, true_labels, raw_df):
    predictions = model.predict(X_new_scaled)

    # predict_proba gives us a CONFIDENCE score, not just a yes/no answer.
    # This is more useful in practice - a security analyst would want to
    # prioritize investigating the rows the model is MOST confident are
    # attacks, rather than treating every flagged row equally.
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

    # If we have the true labels (only possible because we're using a
    # labeled test file for practice), show how accurate we were.
    if true_labels is not None:
        correct = (predictions == true_labels).sum()
        print(f"\n(For reference, since this test file has true labels: "
              f"{correct}/{len(true_labels)} correct = {correct/len(true_labels)*100:.2f}% accuracy)")


# ---------------------------------------------------------------------------
# MAIN: tie it all together
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# WHAT TO LOOK FOR WHEN YOU RUN THIS:
# ---------------------------------------------------------------------------
# - You'll need a SECOND file to test this on - NSL-KDD comes with a
#   separate KDDTest+.txt file specifically for this purpose (different
#   rows than what we trained on). Download it the same place you got
#   KDDTrain+.txt.
# - The "Top 10 most suspicious rows" table lets you eyeball whether the
#   flagged rows make sense - e.g. do the high-confidence ones have
#   unusually high count/serror_rate/num_failed_logins, matching the
#   attack patterns we discussed earlier in this project?
#
# NEXT STEP: optionally wrap this in a Streamlit app so you can upload a
# CSV through a web interface instead of the command line - good for a
# more visual demo in interviews.
