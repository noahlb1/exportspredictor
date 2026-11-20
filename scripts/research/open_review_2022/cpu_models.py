"""Re-predict the 2022 UK grid with the five saved confirmation fold models, on the CPU (no training, no GPU).

Used only to test feature perturbations: each pair is still scored by the fold model that held it out
(ADR 0001). The clip ranges are refitted on the 2012–2021 training rows, as the loop fitted them.
"""
import json

import numpy as np
import polars as pl
import xgboost as xgb
from common import DATA, LOOP, OUT

from exportspredictor.features import CLIPPED_FEATURES, FEATURES, FeatureTransform

FOLD_MODELS = ["a9314febb8f2e546", "47a1d779af67057c", "f5531a1394b61d75", "6a511c331b5547e0", "2b7d1510b9dca2b0"]
TRANSFORM = OUT / "transform_2012_2021.json"


def transform() -> FeatureTransform:
    if TRANSFORM.exists():
        return FeatureTransform.from_dict(json.loads(TRANSFORM.read_text()))
    files = [str(DATA / "derived" / "features_V202601" / f"training_{y}.parquet") for y in range(2012, 2022)]
    scan = pl.scan_parquet(files).select("year", "sector", *CLIPPED_FEATURES)
    t = FeatureTransform.fit(scan, list(range(2012, 2022)))
    TRANSFORM.write_text(json.dumps(t.to_dict()))
    return t


def models() -> list[tuple[xgb.Booster, int]]:
    out = []
    for key in FOLD_MODELS:
        b = xgb.Booster(model_file=str(LOOP / "models" / f"{key}.ubj"))
        b.set_param({"device": "cpu", "nthread": 16})
        out.append((b, json.loads((LOOP / "models" / f"{key}.json").read_text())["rounds"]))
    return out


_MODELS = None


def predict(df: pl.DataFrame) -> np.ndarray:
    """Model-alone Export Potential (USD) of untransformed feature rows with a `fold` column."""
    global _MODELS
    _MODELS = _MODELS or models()
    rows = transform().apply(df)
    out = np.zeros(df.height)
    fold = df["fold"].to_numpy()
    for k, (b, rounds) in enumerate(_MODELS):
        m = fold == k
        x = np.ascontiguousarray(rows.filter(pl.Series(m)).select(FEATURES).to_numpy(), dtype=np.float32)
        out[m] = b.inplace_predict(x, iteration_range=(0, rounds))
    return out


if __name__ == "__main__":
    import time

    from common import frame
    df = frame()
    t = time.time()
    p = predict(df)
    print("seconds", round(time.time() - t), "max rel diff vs saved v2:",
          float(np.max(np.abs(p - df["v2"].to_numpy()) / np.maximum(df["v2"].to_numpy(), 1))),
          "median", float(np.median(np.abs(p - df["v2"].to_numpy()) / np.maximum(df["v2"].to_numpy(), 1))))
