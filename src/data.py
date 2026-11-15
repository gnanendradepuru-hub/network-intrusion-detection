"""Data loading, label mapping and feature engineering for NSL-KDD."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

BASE_FEATURES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]
RAW_COLUMNS = BASE_FEATURES + ["label", "difficulty"]
CATEGORICAL = ["protocol_type", "service", "flag"]

CLASSES = ["normal", "dos", "probe", "r2l", "u2r"]

ATTACK_FAMILY = {
    "normal": "normal",
    # Denial of service
    **{a: "dos" for a in [
        "back", "land", "neptune", "pod", "smurf", "teardrop", "mailbomb",
        "apache2", "processtable", "udpstorm", "worm"]},
    # Probing / reconnaissance
    **{a: "probe" for a in ["ipsweep", "nmap", "portsweep", "satan", "mscan", "saint"]},
    # Remote to local (unauthorised access from outside, e.g. password guessing)
    **{a: "r2l" for a in [
        "ftp_write", "guess_passwd", "imap", "multihop", "phf", "spy",
        "warezclient", "warezmaster", "xlock", "xsnoop", "snmpguess",
        "snmpgetattack", "httptunnel", "sendmail", "named"]},
    # User to root (privilege escalation)
    **{a: "u2r" for a in [
        "buffer_overflow", "loadmodule", "perl", "rootkit", "ps", "sqlattack", "xterm"]},
}

# Flags that indicate a failed / abnormal TCP connection.
ERROR_FLAGS = {"S0", "REJ", "RSTO", "RSTOS0", "RSTR", "SH"}


def load_raw(name: str) -> pd.DataFrame:
    """Load KDDTrain+.txt or KDDTest+.txt."""
    df = pd.read_csv(DATA_DIR / name, header=None, names=RAW_COLUMNS)
    df["attack"] = df["label"].str.strip()
    df["family"] = df["attack"].map(ATTACK_FAMILY)
    unknown = df["family"].isna().sum()
    if unknown:
        raise ValueError(f"{unknown} rows have attack names missing from ATTACK_FAMILY")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicates (on features + label) so train/val do not leak."""
    before = len(df)
    df = df.drop_duplicates(subset=BASE_FEATURES + ["attack"]).reset_index(drop=True)
    print(f"  dropped {before - len(df)} duplicate rows ({before} -> {len(df)})")
    return df


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add a few domain features. Works on raw 41-field flow records."""
    out = df[BASE_FEATURES].copy()
    out["log_src_bytes"] = np.log1p(out["src_bytes"])
    out["log_dst_bytes"] = np.log1p(out["dst_bytes"])
    out["bytes_ratio"] = (out["src_bytes"] + 1) / (out["dst_bytes"] + 1)
    out["bytes_ratio"] = np.log(out["bytes_ratio"])  # symmetric around 0
    out["is_error_flag"] = out["flag"].isin(ERROR_FLAGS).astype(int)
    out["mean_error_rate"] = out[
        ["serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate"]
    ].mean(axis=1)
    out["host_error_rate"] = out[
        ["dst_host_serror_rate", "dst_host_srv_serror_rate",
         "dst_host_rerror_rate", "dst_host_srv_rerror_rate"]
    ].mean(axis=1)
    return out


class FlowPreprocessor:
    """Engineer features + one-hot encode with a schema frozen at fit time.

    Categories unseen at fit time (new services in the test set, for example)
    are simply all-zero in the one-hot block instead of crashing the model.
    """

    def __init__(self) -> None:
        self.columns_: list[str] | None = None

    def _encode(self, df: pd.DataFrame) -> pd.DataFrame:
        feats = engineer(df)
        return pd.get_dummies(feats, columns=CATEGORICAL, dtype=int)

    def fit(self, df: pd.DataFrame) -> "FlowPreprocessor":
        self.columns_ = list(self._encode(df).columns)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.columns_ is None:
            raise RuntimeError("fit() first")
        enc = self._encode(df)
        return enc.reindex(columns=self.columns_, fill_value=0)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
