"""Fast contract checks on cleaned data; run before any experiment."""
import argparse

import numpy as np
import pandas as pd

from prepare import CAT, NUM

LEAKY = {"price", "log_price", "price_per_m2", "name", "description"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    a = ap.parse_args()
    df = pd.read_parquet(a.data)

    assert len(df) > 0, "empty dataset after scoping"
    assert not LEAKY & set(NUM + CAT), f"leaky feature: {LEAKY & set(NUM + CAT)}"
    assert df.price.gt(0).all() and np.isfinite(df.price).all(), "target must be positive finite"
    assert df.area.between(10, 500).all(), "area outside declared study domain"
    assert df.property_type_name.eq("Nhà").all(), "scope drifted off Nha"
    assert df.province_name.eq("Hà Nội").all(), "scope drifted off Ha Noi"
    assert df.group_id.notna().all() and df.group_id.nunique() >= 5

    print(f"rows={len(df):,} groups={df.group_id.nunique():,}")
    print(df[NUM].describe().loc[["count", "50%", "max"]].round(2).to_string())
    print("missing:", df[NUM + CAT].isna().mean().round(3).to_dict())


if __name__ == "__main__":
    main()
