"""Sinh notebook .ipynb chạy trực tiếp trên Kaggle (Add Input dataset, không cần tải về)."""
import json
from pathlib import Path

cells = []


def md(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(True)})


def code(src):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": src.strip("\n").splitlines(True)})


md("""
# Phân tích và ước lượng giá rao bán bất động sản tại Hà Nội

**Dataset:** `locngodev/vietnam-real-estate` — 1 file `raw_data_all.csv`, ~3,5 triệu dòng, 19 cột.

Notebook này chạy **trên Kaggle**: chỉ cần *Add Input* dataset, không tải 3,2 GB về máy.

**Phạm vi:** `province_name == "Hà Nội"`, **giữ mọi loại hình** (Nhà, Căn hộ, Đất, ...) để so sánh tổng thể và dùng `property_type_name` làm biến phân loại trong mô hình.
**Mục tiêu dự đoán:** `price` — đây là **giá rao bán**, không phải giá giao dịch thực tế.
""")

md("## 0. Môi trường và đường dẫn dữ liệu")
code("""
import glob, hashlib, json, os, sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE") and not Path("/kaggle").exists():
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

sns.set_theme(style="whitegrid", palette="colorblind")
pd.set_option("display.width", 140)

OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("outputs")
OUT.mkdir(parents=True, exist_ok=True)

override = os.environ.get("VIETNAM_RAW")
cands = [override] if override else sorted(glob.glob("/kaggle/input/**/raw_data_all.csv", recursive=True))
assert cands, "Chưa gắn dataset: panel phải → Add Input → locngodev/vietnam-real-estate"
RAW = Path(cands[0])
print("RAW     :", RAW)
print("Kích cỡ :", f"{RAW.stat().st_size / 1e9:.2f} GB")
print("Python  :", sys.version.split()[0], "| pandas", pd.__version__, "| numpy", np.__version__)
""")

md("""
## 1. Quét toàn file theo chunk — đo "phễu" lọc dữ liệu

File 3,2 GB không đọc hết vào RAM. Đọc từng chunk 200.000 dòng, chỉ giữ các dòng thuộc phạm vi.
Đếm số dòng còn lại sau **từng bước lọc** để báo cáo trung thực.
""")
code("""
COLS = ["name", "property_type_name", "province_name", "district_name", "ward_name",
        "street_name", "project_name", "price", "area", "floor_count",
        "frontage_width", "bedroom_count", "bathroom_count", "house_direction",
        "published_at"]
CHUNK = 200_000

funnel = {"doc": 0, "ha_noi": 0, "gia_dien_tich_duong": 0}
parts = []

for chunk in pd.read_csv(RAW, usecols=COLS, chunksize=CHUNK, dtype="string"):
    funnel["doc"] += len(chunk)
    step = chunk[chunk.province_name.str.normalize("NFC").str.strip() == "Hà Nội"].copy()
    funnel["ha_noi"] += len(step)
    for c in ["price", "area", "floor_count", "frontage_width", "bedroom_count", "bathroom_count"]:
        step[c] = pd.to_numeric(step[c], errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan)
    step = step[step.area.gt(0) & step.price.gt(0)]
    funnel["gia_dien_tich_duong"] += len(step)
    if len(step):
        parts.append(step)

if not parts:
    raise ValueError("Không có dòng Hà Nội với giá và diện tích dương; kiểm tra dataset/nhãn.")
df = pd.concat(parts, ignore_index=True)
del parts
import gc; gc.collect()

# Audit, không tự kết luận cực đoan là lỗi và không lọc để làm hình đẹp.
df["price_per_m2"] = df.price / df.area
if not np.isfinite(df.price_per_m2).all():
    raise ValueError("price/area vượt miền số hữu hạn; cần kiểm tra bản ghi trước khi tiếp tục.")
pd.DataFrame([funnel]).T.rename(columns={0: "rows"}).to_csv(OUT / "funnel.csv")
df.nlargest(50, "price").to_csv(OUT / "audit_top_price.csv", index=False)
df.nlargest(50, "price_per_m2").to_csv(OUT / "audit_top_price_per_m2.csv", index=False)
print(json.dumps(funnel, indent=2, ensure_ascii=False))
print("Giữ tất cả loại hình và giá trị cực đoan dương; audit riêng, không tự gán lỗi đơn vị.")
print(f"Tỷ lệ giữ trước chống trùng: {len(df) / max(funnel['doc'], 1):.2%}")
""")

md("""
## 2. Chuẩn hoá, chống trùng tin, tạo feature

Cùng một căn thường được đăng lại nhiều ngày. Dataset không có ID bất động sản, nên dùng
**hash(tỉnh + quận + tiêu đề chuẩn hoá)**; thiếu tiêu đề thì mỗi dòng một nhóm làm *proxy nhóm* để chia train/test không chồng lấn.
Đây là proxy — tin sửa tiêu đề vẫn có thể lọt.
""")
code("""
def norm_text(s):
    return s.astype("string").str.normalize("NFC").str.strip()

for c in ["district_name", "ward_name", "street_name", "project_name",
          "house_direction", "property_type_name", "province_name"]:
    df[c] = norm_text(df[c]).replace("", pd.NA)

df["property_type_name"] = df.property_type_name.fillna("Không rõ")

df["published_at"] = pd.to_datetime(df.published_at, errors="coerce", format="mixed")
df["month"] = df.published_at.dt.month
df["year"] = df.published_at.dt.year

# giá/m2 chỉ để AUDIT, không đưa vào feature
df["price_per_m2"] = df.price / df.area

title = norm_text(df.name).str.lower().str.replace(r"\\s+", " ", regex=True)
title = title.where(title.notna() & title.ne(""), "__missing_title_row_" + df.index.astype(str))
keys = (df.province_name.fillna("") + "|" + df.district_name.fillna("") + "|"
        + title)
df["group_id"] = keys.map(lambda s: hashlib.sha256(s.encode()).hexdigest())

before = len(df)
df = df.drop_duplicates(subset=["group_id", "price", "area", "district_name", "property_type_name"])
df = df.reset_index(drop=True)

clean = df[["district_name", "ward_name", "property_type_name", "area", "bedroom_count", "bathroom_count",
            "frontage_width", "price", "price_per_m2", "month", "year", "group_id"]]
clean.to_parquet(OUT / "clean.parquet", index=False)

print(f"Trước dedupe: {before:,} | sau: {len(df):,} | nhóm tiêu đề: {df.group_id.nunique():,}")
print("\\nTỉ lệ thiếu:")
print(clean.isna().mean().round(3).to_string())
del chunk, step
import gc; gc.collect()
display(clean.describe(percentiles=[.01, .5, .99]).round(1))
""")

md("## 3. EDA — bốn thư viện: Pandas, NumPy, Matplotlib, Seaborn")

md("### 3.1 Phân phối giá và diện tích — zoom đúng bins, không lọc P99")
code("""
from matplotlib.ticker import FuncFormatter, MaxNLocator


def zoom_hist(axis, values, label, color):
    values = np.asarray(values, dtype=float)
    hi = float(np.quantile(values, .99))
    lo = 0.0
    hi = max(hi, np.finfo(float).eps)
    edges = np.linspace(lo, hi, 61)
    counts, _ = np.histogram(values, bins=edges)
    # Kiểm tra histogram chỉ đếm miền hiển thị, không dồn cả đuôi vào cột đầu.
    assert counts.sum() == np.count_nonzero((values >= lo) & (values <= hi))
    axis.hist(values, bins=edges, color=color, edgecolor="white", linewidth=.35)
    q25, median, q75 = np.quantile(values, [.25, .5, .75])
    axis.axvspan(q25, q75, color="black", alpha=.07, label="Khoảng Q1–Q3")
    axis.axvline(median, color="#222222", ls="--", lw=1.4,
                label=f"Trung vị: {median:,.1f}")
    outside = int((values > hi).sum())
    axis.set(xlim=(lo, hi), xlabel=label, ylabel="Số tin",
             title=f"Zoom đến P99 = {hi:,.1f} | ngoài khung: {outside:,} ({outside / len(values):.2%})")
    axis.legend(fontsize=9, loc="upper right")
    axis.yaxis.set_major_locator(MaxNLocator(integer=True))  # đếm tin: không hiện 0,5
    axis.xaxis.set_major_locator(MaxNLocator(6))
    return counts


fig, ax = plt.subplots(2, 2, figsize=(15, 10), layout="constrained")
fig.suptitle(f"Hà Nội — mọi loại hình | {len(df):,} dòng sau chống trùng", fontsize=15)
zoom_hist(ax[0, 0], df.price / 1e9, "Giá rao bán (tỷ VND)", "#D55E00")
# log10 giá theo tỷ: toàn miền; ghi giá thật trên ticks để dễ đọc.
log_price = np.log10(df.price.to_numpy() / 1e9)
ax[0, 1].hist(log_price, bins=70, color="#0072B2", edgecolor="white", linewidth=.35)
ax[0, 1].xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{10.0 ** x:,.3g}"))
ax[0, 1].yaxis.set_major_locator(MaxNLocator(integer=True))
ax[0, 1].set(title="Toàn miền giá — bins đều theo log10, không cắt đuôi",
             xlabel="Giá rao bán (tỷ VND; khoảng cách theo log10)", ylabel="Số tin")
zoom_hist(ax[1, 0], df.area, "Diện tích (m²)", "#CC79A7")
zoom_hist(ax[1, 1], df.price_per_m2 / 1e6, "Giá rao bán/m² (triệu VND/m²)", "#009E73")
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"fig1_distributions.{ext}", dpi=300, bbox_inches="tight")
plt.show()
print("Zoom chỉ thay cách vẽ; df, clean và dữ liệu mô hình không bị cắt P99.")
print(df[["price", "area", "price_per_m2"]]
      .describe(percentiles=[.01, .25, .5, .75, .99]).T.round(0).to_string())
""")

md("""
**Cách đọc:** so sánh đuôi phân phối trước/sau log. Không mặc định dữ liệu đã phân phối chuẩn; chỉ kết luận sau khi xem kết quả. Giá trị cực đoan chưa được xác minh nên có thể làm sai lệch đồ thị và metric.
""")

md("""
### 3.2 So sánh loại hình bất động sản

Bảng giữ tất cả nhãn; hình hiển thị tối đa 15 loại phổ biến nhất, ghi rõ nếu có nhóm không vẽ.
Trung vị theo loại không kiểm soát khác biệt vị trí/diện tích, không phải tác động nhân quả.
Thiếu phòng ngủ của Đất không mặc định là 0: giữ missing và thêm missing indicator trong pipeline.
""")
code("""
type_summary = df.groupby("property_type_name", dropna=False).agg(
    so_tin=("price", "size"),
    gia_trung_vi_ty=("price", lambda x: x.median() / 1e9),
    dien_tich_trung_vi_m2=("area", "median"),
    gia_m2_trung_vi_trieu=("price_per_m2", lambda x: x.median() / 1e6),
).sort_values("so_tin", ascending=False)
type_summary["ty_trong_phan_tram"] = 100 * type_summary.so_tin / len(df)
type_summary.to_csv(OUT / "property_type_summary.csv")
display(type_summary.round(2))
missing_type = df.groupby("property_type_name")[["bedroom_count", "bathroom_count", "frontage_width"]].agg(
    lambda x: x.isna().mean() * 100)
missing_type.to_csv(OUT / "missing_by_property_type.csv")
print("Tỷ lệ thiếu theo loại hình (%):")
display(missing_type.round(1))

order = type_summary.head(15).index
fig, axes = plt.subplots(1, 3, figsize=(19, max(5, .45 * len(order))), layout="constrained")
bars = axes[0].barh(order, type_summary.loc[order, "so_tin"], color="#0072B2")
axes[0].bar_label(bars, labels=[f"{v:,}" for v in type_summary.loc[order, "so_tin"]], padding=3, fontsize=9)
axes[0].invert_yaxis()
axes[0].set(xlabel="Số tin", title="Cơ cấu loại hình (tối đa 15 nhãn)")
axes[0].margins(x=.22)
plot_types = df[df.property_type_name.isin(order)].copy()
plot_types["gia_ty"] = plot_types.price / 1e9
plot_types["gia_m2_trieu"] = plot_types.price_per_m2 / 1e6
for axis, col, label, color in [
    (axes[1], "gia_ty", "Giá rao bán (tỷ VND; thang log)", "#D55E00"),
    (axes[2], "gia_m2_trieu", "Giá/m² (triệu VND; thang log)", "#009E73"),
]:
    sns.boxplot(data=plot_types, x=col, y="property_type_name", order=order,
                showfliers=False, color=color, ax=axis)
    axis.set_xscale("log")
    axis.set(xlabel=label, ylabel="", title="Boxplot — ẩn điểm ngoài râu, không xoá dữ liệu")
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"fig_property_types.{ext}", dpi=300, bbox_inches="tight")
plt.show()
print(f"Hình hiển thị {len(order)}/{len(type_summary)} loại; bảng CSV giữ tất cả.")
del plot_types
""")

md("### 3.3 Quan hệ diện tích – giá")
code("""
samp = df.sample(min(20_000, len(df)), random_state=42).copy()
samp["gia_ty"] = samp.price / 1e9
fig, ax = plt.subplots(figsize=(12, 6), layout="constrained")
sns.scatterplot(data=samp, x="area", y="gia_ty", hue="property_type_name",
                s=12, alpha=.35, linewidth=0, ax=ax)
ax.set(xscale="log", yscale="log", xlabel="Diện tích (m²; thang log)",
       ylabel="Giá rao bán (tỷ VND; thang log)",
       title=f"Diện tích và giá theo loại hình — mẫu ngẫu nhiên {len(samp):,} tin")
ax.legend(title="Loại hình", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
fig.savefig(OUT / "fig2_area_price.png", dpi=300, bbox_inches="tight")
plt.show()
print("Pearson area–price:", round(df.area.corr(df.price), 3))
print("Spearman area–price:", round(df.area.corr(df.price, method="spearman"), 3))
print("Tương quan tổng thể có thể bị cơ cấu loại hình/vị trí và cực đoan chi phối.")
""")

md("### 3.4 Giá theo quận và tương quan giữa các biến số")
code("""
top = df.district_name.value_counts().head(10).index
sub = df[df.district_name.isin(top)]

fig, ax = plt.subplots(1, 2, figsize=(15, 6))
sns.boxplot(data=sub, x=sub.price / 1e9, y="district_name", order=top, ax=ax[0],
            color="#ff6a41", showfliers=False)
ax[0].set(title="Giá rao bán theo quận (bỏ outlier khỏi hình)", xlabel="Giá (tỷ VND; thang log)", ylabel="")
ax[0].set_xscale("log")

num = ["price", "area", "price_per_m2", "bedroom_count", "bathroom_count", "frontage_width"]
sns.heatmap(df[num].corr(), annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1,
            square=True, ax=ax[1])
ax[1].set_title("Ma trận tương quan")

plt.tight_layout()
plt.savefig(OUT / "fig3_district_corr.png", dpi=130, bbox_inches="tight")
plt.show()

med = (df.groupby("district_name").price.median().div(1e9).sort_values(ascending=False))
print("Trung vị giá theo quận (tỷ VND), top 10:")
print(med.head(10).round(2).to_string())
""")

md("""
**Cảnh báo diễn giải:** `price_per_m2` *được tính từ* `price`; mức tương quan phụ thuộc dữ liệu, không tự khẳng định cao. So sánh quận tổng thể còn phụ thuộc cơ cấu loại hình.
Biến này **không được** đưa vào mô hình — chỉ dùng để kiểm tra outlier.
Tương quan không chứng minh quan hệ nhân quả.
""")

md("""
## 4. Mô hình

- Chia tập theo **nhóm tiêu đề** (`GroupShuffleSplit`) — cùng một tin không nằm cả hai phía.
- Baseline: dự đoán trung vị, để biết mô hình có hơn "đoán bừa" hay không.
- `LinearRegression`, và `Ridge` với target `log1p`.
- Đo trên **VND gốc**: MAE, RMSE, MdAPE; R² báo trên thang giá gốc. Giá trị âm có thể xảy ra nếu mô hình kém mốc dự đoán trung bình.
""")
code("""
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUM = ["area", "bedroom_count", "bathroom_count", "frontage_width"]
CAT = ["district_name", "ward_name", "property_type_name"]

X = df[NUM + CAT].copy()
for c in CAT:
    X[c] = X[c].astype(object).where(X[c].notna(), np.nan)
y = df.price.to_numpy(float)

assert df.group_id.nunique() >= 5, "Cần ít nhất 5 nhóm tin hợp lệ để chia tập."
tr, te = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42)
              .split(X, y, groups=df.group_id))
assert set(df.group_id.iloc[tr]).isdisjoint(df.group_id.iloc[te])
print(f"train {len(tr):,} dòng | test {len(te):,} dòng | {df.group_id.iloc[tr].nunique():,} nhóm train")


def build(est, log_target):
    prep = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True,
                                                keep_empty_features=True)),
                          ("sc", StandardScaler())]), NUM),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="Unknown")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), CAT),
    ])
    pipe = Pipeline([("prep", prep), ("model", est)])
    return (TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)
            if log_target else pipe)


models = {
    "Baseline (trung vị)": build(DummyRegressor(strategy="median"), False),
    "LinearRegression": build(LinearRegression(), False),
    "Ridge + log(target)": build(Ridge(alpha=10.0, solver="lsqr"), True),
}

rows, preds = [], {}
for name, m in models.items():
    m.fit(X.iloc[tr], y[tr])
    p = m.predict(X.iloc[te])
    assert np.isfinite(p).all()
    preds[name] = p
    rows.append({
        "Mô hình": name,
        "MAE (tỷ VND)": mean_absolute_error(y[te], p) / 1e9,
        "RMSE (tỷ VND)": np.sqrt(mean_squared_error(y[te], p)) / 1e9,
        "MdAPE (%)": 100 * np.median(np.abs(y[te] - p) / y[te]),
        "R² (giá gốc)": r2_score(y[te], p),
        "Số dự đoán âm": int((p < 0).sum()),
    })

metrics = pd.DataFrame(rows).set_index("Mô hình")
metrics.round(3).to_csv(OUT / "metrics.csv")
display(metrics.round(3))
print(f"\\nTrung vị giá test: {np.median(y[te])/1e9:.2f} tỷ | trung bình: {y[te].mean()/1e9:.2f} tỷ")
""")

md("""
### 4.1 Đánh giá riêng từng loại hình

Split nhóm không phân tầng theo loại: báo cả số train/test và loại chỉ xuất hiện ở test.
Báo lỗi của cả ba mô hình theo loại, tránh điểm tổng bị loại phổ biến nhất che lấp.
Nhóm test dưới 30 dòng chỉ tham khảo, không đủ kết luận ổn định.
""")
code("""
coverage = pd.concat([
    df.property_type_name.iloc[tr].value_counts().rename("train"),
    df.property_type_name.iloc[te].value_counts().rename("test"),
], axis=1).fillna(0).astype(int)
coverage.to_csv(OUT / "type_split_counts.csv")
display(coverage)
by_type = []
for name, prediction in preds.items():
    for kind in sorted(df.property_type_name.iloc[te].unique()):
        mask = df.property_type_name.iloc[te].eq(kind).to_numpy()
        truth, guess = y[te][mask], prediction[mask]
        by_type.append({
            "model": name, "property_type_name": kind, "test_rows": len(truth),
            "train_rows": int(coverage.loc[kind, "train"]),
            "MAE_ty": mean_absolute_error(truth, guess) / 1e9,
            "RMSE_ty": np.sqrt(mean_squared_error(truth, guess)) / 1e9,
            "MdAPE_pct": 100 * np.median(np.abs(truth - guess) / truth),
            "small_test": len(truth) < 30,
        })
metrics_type = pd.DataFrame(by_type)
metrics_type.to_csv(OUT / "metrics_by_property_type.csv", index=False)
display(metrics_type.round(3))
""")

md("### 4.2 Sai số của Ridge log — mô hình chọn trước")
code("""
best = "Ridge + log(target)"  # chọn trước, không chọn mô hình bằng test
p = preds[best]
resid = np.log1p(y[te]) - np.log1p(np.clip(p, 1, None))

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ax[0].scatter(y[te] / 1e9, p / 1e9, s=7, alpha=.3, color="#2f6fed")
lim = [0, np.percentile(y[te] / 1e9, 99.5)]
ax[0].plot(lim, lim, "r--", lw=1)
ax[0].set(title=f"{best}: thực tế vs dự đoán", xlabel="Giá thực (tỷ VND)", ylabel="Dự đoán (tỷ VND)")

ax[1].hist(resid, bins=60, color="#ff6a41")
ax[1].axvline(0, color="k", lw=1)
ax[1].set(title="Phân phối sai số (thang log)", xlabel="log(giá thực) − log(dự đoán)")

plt.tight_layout()
plt.savefig(OUT / "fig4_residuals.png", dpi=130, bbox_inches="tight")
plt.show()

# sai số theo phân khúc giá
band = pd.qcut(y[te], min(4, len(te)), duplicates="drop")
seg = pd.DataFrame({"band": band, "ape": np.abs(y[te] - p) / y[te] * 100}).groupby("band", observed=True).ape.median()
print("MdAPE (%) theo phân khúc giá:")
print(seg.round(1).to_string())
""")

md("### 4.3 Hệ số — liên hệ trong mô hình, không phải nhân quả")
code("""
lin = models["Ridge + log(target)"].regressor_.named_steps["model"]
feat = models["Ridge + log(target)"].regressor_.named_steps["prep"].get_feature_names_out()
coef = pd.Series(lin.coef_, index=feat).sort_values()

print("Hệ số Ridge log, sau chuẩn hóa cột số:")
print(pd.concat([coef.head(8), coef.tail(8)]).round(3).to_string())
print("\\nKhông suy diễn nhân quả. Cột số đã chuẩn hóa; one-hot đầy đủ, không bỏ nhóm gốc.\\nVới cùng biến phân loại, so sánh chênh lệch giữa hai hệ số thay vì coi một hệ số là mức tăng so nhóm gốc.")
""")

md("""
## 5. Kết luận và giới hạn

**Viết trong báo cáo:**

1. Dữ liệu là **giá rao bán**, không phải giá giao dịch — mọi kết luận chỉ nói về hành vi định giá của người đăng tin.
2. Giữ mọi loại hình Hà Nội với giá và diện tích hữu hạn, dương. Không áp ngưỡng giá/diện tích chung cho Nhà, Đất, Căn hộ. Giá trị cực đoan cần xác minh; chỉ zoom biểu đồ, không xoá khỏi dữ liệu/mô hình. Metric giá gốc có thể bị chi phối mạnh.
3. Chia tập theo nhóm tiêu đề chỉ là proxy chống trùng; tin sửa tiêu đề vẫn có thể rò rỉ giữa train và test.
4. Không dùng `price_per_m2` làm feature (nó là hàm của target). Đã kiểm tra điều này bằng assert bên dưới.
5. So với baseline trên các chỉ số thật; không khẳng định Ridge tốt hơn trước khi chạy. Nếu cần tune, dùng group cross-validation trên train, không tune bằng test.
6. Kiểm tra sai số theo phân khúc; nếu cao ở nhóm nhà đắt, xem lại nhiễu giá, độ đại diện và đặc trưng. Chưa đủ dữ kiện khẳng định nguyên nhân.
7. Split nhóm tiêu đề không phải kiểm định dự báo tương lai. Kết quả là ước lượng trong phạm vi dataset đã lọc.
""")
code("""
# kiểm tra chống rò rỉ
assert not {"price", "price_per_m2", "log_price"} & set(NUM + CAT)
assert np.isfinite(df.price).all() and (df.price > 0).all()
import joblib
joblib.dump(models["Ridge + log(target)"], OUT / "ridge_log.joblib")
state = {"evaluation": "SMOKE_TEST_ONLY" if len(df) < 1000 else "BASELINE_NOT_TUNED",
         "train_rows": len(tr), "test_rows": len(te), "seed": 42,
         "target": "asking_price_VND", "province": "Hà Nội",
         "property_types": sorted(df.property_type_name.unique().tolist()),
         "filters": "finite positive price and area only",
         "note": "Extreme prices not individually verified; P99 affects plots only"}
(OUT / "run.json").write_text(json.dumps(state, ensure_ascii=False, indent=2))

print("Đã ghi:")
for f in sorted(OUT.iterdir()):
    print(" -", f.name, f"{f.stat().st_size/1024:.0f} KB")
""")

nb = {
    "cells": [dict(c, id=f"cell-{i:03d}") for i, c in enumerate(cells)],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

p = Path(__file__).with_name("kaggle_hanoi_all_property_types.ipynb")
p.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print(p, p.stat().st_size, "bytes,", len(cells), "cells")
