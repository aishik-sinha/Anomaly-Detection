"""
preprocessing.py
-------------------
Shared logic used across every script in this project: column names,
loading, encoding, and scaling. Previously this exact code was copy-pasted
into 01 through 07 - pulling it into one place here means a fix or change
only needs to happen once, and each script's actual purpose is clearer
without 40 lines of setup at the top.

Import what you need, e.g.:
    from preprocessing import COLUMN_NAMES, load_and_clean, encode_features, scale_features
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ---------------------------------------------------------------------------
# NSL-KDD has no header row in the raw file, so every script needs these
# column names to make sense of the data. Defined once, here.
# ---------------------------------------------------------------------------
COLUMN_NAMES = [
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

CATEGORICAL_COLUMNS = ["protocol_type", "service", "flag"]


def load_and_clean(filepath):
    """
    Load a raw NSL-KDD file and add the simplified binary label.

    Returns the full DataFrame (still has 'label' and 'difficulty' - the
    caller decides whether/when to drop those, since some scripts need
    the detailed 'label' column later for things like per-attack-type
    breakdowns or sample_weight calculations).
    """
    df = pd.read_csv(filepath, header=None, names=COLUMN_NAMES)
    df["binary_label"] = df["label"].apply(lambda x: "normal" if x == "normal" else "attack")
    return df


def encode_features(df, encoders=None):
    """
    Convert the text columns (protocol_type, service, flag) into numbers.

    If encoders=None, this FITS new encoders (use this on training data).
    If encoders is a dict of already-fit encoders, this REUSES them
    (use this on new/test data, so categories map the same way they did
    during training).

    Returns (encoded_df, encoders_dict).
    """
    df = df.copy()

    if encoders is None:
        # Fitting mode - training data
        encoders = {}
        for col in CATEGORICAL_COLUMNS:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
    else:
        # Reuse mode - new/test data. Map unseen categories to -1 instead
        # of crashing, since new data can contain values training never saw.
        for col, encoder in encoders.items():
            df[col] = df[col].apply(
                lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
            )

    return df, encoders


def scale_features(X, scaler=None):
    """
    Scale numeric features to mean=0, std=1.

    If scaler=None, this FITS a new scaler (use this on training data).
    If scaler is an already-fit StandardScaler, this REUSES it (use this
    on new/test data).

    Returns (X_scaled, scaler).
    """
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    return X_scaled, scaler
