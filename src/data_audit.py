"""Reproducible, non-destructive audit; no preprocessing fit or model training.

Run from the repository root: python -m src.data_audit
"""

from __future__ import annotations

import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TARGET = "state_of_health_pct"
SEED = 42
EXPECTED_SHA256 = "cacdf6afcf50db86583be3754207a89acbfd60b04dfff8f69b71e29cfd78218d"
SOURCE = "https://www.kaggle.com/datasets/vinay24baghira/ev-battery-health-and-degradation-dataset"
DATE_COLUMNS = ["manufacture_date", "observation_date", "last_service_date"]
CATEGORICAL = ["chemistry", "manufacturer", "vehicle_model", "climate_zone", "bms_firmware_version"]
PROXY_COLUMNS = ["internal_resistance_mohm", "voltage_avg_v", "current_avg_a", "charge_time_avg_min"]
OTHER_TARGETS = ["remaining_useful_life_cycles", "failure_risk_label", "degradation_category"]


def load_dataset(root: Path = ROOT) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    path = root / "data/raw/ev_battery_health.csv"
    if not path.is_file():
        raise FileNotFoundError("Unduh ev_battery_health.csv ke data/raw; lihat data/README.md.")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if sha != EXPECTED_SHA256:
        raise ValueError("Versi CSV berbeda. Audit ulang versi/split; jangan menimpa hasil lama secara diam-diam.")
    frame = pd.read_csv(path)
    dictionary = pd.read_csv(root / "data/raw/data_dictionary .csv")
    if set(frame.columns) != set(dictionary["column"]) or len(frame.columns) != len(dictionary):
        raise ValueError("Skema CSV dan kamus data tidak sesuai.")
    if frame["battery_id"].isna().any() or frame["observation_id"].duplicated().any():
        raise ValueError("ID hilang atau observation_id tidak unik; split perlu ditinjau.")
    if frame[TARGET].isna().any():
        raise ValueError("Target hilang; tetapkan kebijakan sebelum split.")
    return frame, dictionary, sha


def feature_contract(frame: pd.DataFrame, sha: str) -> dict:
    excluded = {
        "battery_id": "ID kelompok untuk split, bukan fitur.",
        "observation_id": "ID observasi untuk pelacakan, bukan fitur.",
        "capacity_fade_pct": "Identitas langsung: 100 - SOH; kebocoran target.",
        **{c: "Target alternatif/label kesehatan; bukan input regresi SOH." for c in OTHER_TARGETS},
        **{c: "Tanggal mentah dikeluarkan dari baseline; gunakan battery_age_days yang tersedia." for c in DATE_COLUMNS},
        **{c: "Kandidat proxy sensor sintetis; hanya untuk ablation terpisah setelah meninjau asal/availability." for c in PROXY_COLUMNS},
    }
    primary = [c for c in frame if c not in excluded and c != TARGET]
    return {
        "task": "Regresi SOH saat observasi untuk baterai yang tidak ada di train (bukan forecasting/RUL)",
        "target": TARGET, "target_unit": "persen; error MAE/RMSE dalam poin persentase",
        "dataset_sha256": sha, "seed": SEED,
        "split_unit": "battery_id", "split_ratios_batteries": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "primary_features": primary,
        "categorical_features": [c for c in primary if c in CATEGORICAL],
        "numeric_features": [c for c in primary if c not in CATEGORICAL and c != "recall_flag"],
        "binary_features": ["recall_flag"],
        "excluded_features": excluded,
        "proxy_ablation_candidates": PROXY_COLUMNS,
        "availability_assumption": "Ringkasan penggunaan/lingkungan dan metadata diketahui pada saat observasi. Jendela agregasi tidak dirinci sumber; tidak mengklaim prediksi sebelum sesi atau masa depan.",
        "timesteps": "Belum dipakai: tanggal hilang dan penghitung kumulatif tidak selalu monoton; perlu audit temporal lanjutan.",
        "preprocessing_status": "Belum fit. Imputasi/encoding/scaling dan seleksi berbasis data hanya boleh belajar dari train.",
        "balancing": "Tidak menerapkan balancing kelas karena target kontinu.",
    }


def build_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    ids = np.array(sorted(frame["battery_id"].unique()))
    shuffled = np.random.default_rng(SEED).permutation(ids)
    # Integer arithmetic avoids int(2600 * 0.70) rounding down to 1819.
    n_train, n_validation = len(ids) * 70 // 100, len(ids) * 15 // 100
    labels = np.repeat(["train", "validation", "test"], [n_train, n_validation, len(ids)-n_train-n_validation])
    manifest = pd.DataFrame({"battery_id": shuffled, "split": labels}).sort_values("battery_id").reset_index(drop=True)
    row_split = frame["battery_id"].map(manifest.set_index("battery_id")["split"])
    groups = {s: set(manifest.loc[manifest["split"].eq(s), "battery_id"]) for s in ["train", "validation", "test"]}
    assert not groups["train"] & groups["validation"]
    assert not groups["train"] & groups["test"]
    assert not groups["validation"] & groups["test"]
    assert len(set.union(*groups.values())) == len(ids) and row_split.notna().all()
    counts = pd.DataFrame({
        "split": ["train", "validation", "test"],
        "batteries": [len(groups[s]) for s in ["train", "validation", "test"]],
        "rows": [int(row_split.eq(s).sum()) for s in ["train", "validation", "test"]],
    })
    counts["row_pct"] = counts["rows"] / len(frame) * 100
    return manifest, row_split, counts


def audit_quality(frame: pd.DataFrame, dictionary: pd.DataFrame, contract: dict) -> tuple[pd.DataFrame, dict]:
    roles = {c: "excluded" for c in contract["excluded_features"]}
    roles.update({c: "primary_feature" for c in contract["primary_features"]})
    roles[TARGET] = "target"
    schema = dictionary.copy()
    schema["actual_dtype"] = schema["column"].map(frame.dtypes.astype(str))
    schema["missing_count"] = schema["column"].map(frame.isna().sum())
    schema["missing_pct"] = schema["missing_count"] / len(frame) * 100
    schema["unique_non_null"] = schema["column"].map(frame.nunique())
    schema["role"] = schema["column"].map(roles)
    schema["decision"] = schema["column"].map(contract["excluded_features"]).fillna("Target").where(schema["role"].ne("primary_feature"), "Kandidat baseline; preprocessing belum fit.")
    dates = frame[DATE_COLUMNS].apply(pd.to_datetime, errors="coerce")
    valid_date = frame["observation_date"].notna()
    delta = (dates["observation_date"] - dates["manufacture_date"]).dt.days - frame["battery_age_days"]
    fade = frame["capacity_fade_pct"].notna()
    summary = {
        "rows": len(frame), "columns": len(frame.columns), "unique_batteries": int(frame["battery_id"].nunique()),
        "duplicate_rows": int(frame.duplicated().sum()), "duplicate_observation_ids": int(frame["observation_id"].duplicated().sum()),
        "missing_targets": int(frame[TARGET].isna().sum()),
        "missing_observation_dates": int((~valid_date).sum()),
        "duplicate_battery_date_non_null": int(frame.loc[valid_date].duplicated(["battery_id", "observation_date"]).sum()),
        "date_parse_errors": {c: int((frame[c].notna() & dates[c].isna()).sum()) for c in DATE_COLUMNS},
        "observation_before_manufacture": int((dates["observation_date"] < dates["manufacture_date"]).sum()),
        "service_after_observation": int((dates["last_service_date"] > dates["observation_date"]).sum()),
        "service_before_manufacture": int((dates["last_service_date"] < dates["manufacture_date"]).sum()),
        "age_check_eligible": int(delta.notna().sum()), "age_date_mismatches": int(delta.dropna().ne(0).sum()),
        "fade_identity_eligible": int(fade.sum()),
        "fade_identity_max_abs_error": float((frame.loc[fade, "capacity_fade_pct"] + frame.loc[fade, TARGET] - 100).abs().max()),
        "observations_per_battery": frame.groupby("battery_id").size().describe().to_dict(),
        "date_ranges": {c: {"min": str(dates[c].min().date()), "max": str(dates[c].max().date())} for c in DATE_COLUMNS},
        "service_missing_count_zero": int((dates["last_service_date"].isna() & frame["service_count"].eq(0)).sum()),
        "service_missing_count_positive": int((dates["last_service_date"].isna() & frame["service_count"].gt(0)).sum()),
        "service_missing_count_missing": int((dates["last_service_date"].isna() & frame["service_count"].isna()).sum()),
        "infinite_numeric_values": int(np.isinf(frame.select_dtypes("number")).sum().sum()),
    }
    violations = {}
    for c in ["cycle_count", "battery_age_days", "total_energy_throughput_kwh"]:
        ordered = frame.assign(parsed_date=dates["observation_date"]).dropna(subset=["parsed_date", c]).sort_values(["battery_id", "parsed_date", "observation_id"])
        lag = ordered.groupby("battery_id")[["parsed_date", c]].shift()
        eligible = ordered["parsed_date"].gt(lag["parsed_date"])
        violations[c] = {"eligible_pairs": int(eligible.sum()), "decreasing_pairs": int((eligible & ordered[c].lt(lag[c])).sum())}
    summary["temporal_counter_checks"] = violations
    summary["within_battery_static_conflicts"] = {c: int(frame.groupby("battery_id")[c].nunique().gt(1).sum()) for c in ["chemistry", "manufacturer", "vehicle_model", "pack_capacity_kwh", "manufacture_date", "climate_zone"]}
    ranges = {"state_of_health_pct": (0, 100), "capacity_fade_pct": (0, 100), "fast_charge_ratio": (0, 1), "avg_depth_of_discharge_pct": (0, 100), "avg_state_of_charge_pct": (0, 100), "humidity_pct": (0, 100), "recall_flag": (0, 1)}
    summary["outside_basic_ranges"] = {c: int((frame[c].notna() & ~frame[c].between(lo, hi)).sum()) for c, (lo, hi) in ranges.items()}
    summary["unexpected_recall_values"] = int((frame["recall_flag"].notna() & ~frame["recall_flag"].isin([0, 1])).sum())
    return schema, summary


def profile_train(train: pd.DataFrame, contract: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    numeric = contract["numeric_features"] + [TARGET] + PROXY_COLUMNS
    statistics = train[numeric].describe(percentiles=[.01, .25, .5, .75, .99]).T.rename_axis("column").reset_index()
    flags = []
    for c in numeric:
        q1, q3 = train[c].quantile([.25, .75])
        lo, hi = q1 - 1.5*(q3-q1), q3 + 1.5*(q3-q1)
        valid = train[c].notna()
        n = int((valid & ~train[c].between(lo, hi)).sum())
        flags.append({"column": c, "iqr_lower_train": lo, "iqr_upper_train": hi, "flagged_rows": n, "non_missing_rows": int(valid.sum()), "flagged_pct_of_non_missing": n / valid.sum() * 100})
    bins = [-np.inf, 70, 80, 90, 95, np.inf]
    bands = pd.cut(train[TARGET], bins=bins, right=False, labels=["<70", "70–<80", "80–<90", "90–<95", ">=95"])
    distribution = bands.value_counts(sort=False).rename_axis("soh_band_pct").reset_index(name="train_rows")
    distribution["train_pct"] = distribution["train_rows"] / len(train) * 100
    return statistics, pd.DataFrame(flags), distribution


def make_figures(train: pd.DataFrame, contract: dict, output: Path) -> list[Path]:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 120, "savefig.facecolor": "white"})
    paths = []
    def save(fig, name):
        path = output / name
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    missing = train.isna().mean().mul(100).sort_values()
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.barh(missing.index, missing.values, color="#207c83")
    ax.set(xlabel="Data kosong pada train (%)", title="Missing value per kolom — train saja", xlim=(0, max(42, missing.max()+3)))
    ax.grid(axis="x", alpha=.18)
    save(fig, "01_missing_train.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(train[TARGET], bins=np.arange(55, 102, 2), color="#207c83", edgecolor="white")
    axes[0].set(xlabel="SOH (%)", ylabel="Jumlah observasi", title="Distribusi target train")
    chemistry = train["chemistry"].fillna("Missing").value_counts()
    axes[1].bar(chemistry.index, chemistry.values, color="#3665a0")
    axes[1].set(xlabel="Kimia baterai", ylabel="Jumlah observasi", title="Komposisi kimia train (per baris)")
    save(fig, "02_target_chemistry_train.png")

    selected = ["battery_age_days", "cycle_count", "fast_charge_ratio", "avg_ambient_temp_c", "avg_depth_of_discharge_pct", "internal_resistance_mohm"]
    plot = train.sample(min(5000, len(train)), random_state=SEED)
    labels = {
        "battery_age_days": "Umur baterai (hari)",
        "cycle_count": "Siklus ekuivalen (siklus)",
        "fast_charge_ratio": "Rasio pengisian cepat (0–1)",
        "avg_ambient_temp_c": "Suhu lingkungan (°C)",
        "avg_depth_of_discharge_pct": "Depth of discharge (%)",
        "internal_resistance_mohm": "Resistansi internal (mΩ)",
    }
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, c in zip(axes.flat, selected):
        ax.scatter(plot[c], plot[TARGET], s=5, alpha=.2, color="#207c83", rasterized=True)
        ax.set(xlabel=labels[c], ylabel="SOH (%)")
        if c == "internal_resistance_mohm":
            ax.set_title("Proxy sensor: bukan fitur baseline", fontsize=9)
    fig.suptitle("Hubungan fitur–target: sampel tetap 5.000 baris train", y=1.01)
    fig.tight_layout()
    save(fig, "03_relationships_train.png")

    sensor = ["avg_ambient_temp_c", "internal_resistance_mohm", "current_avg_a", "charge_time_avg_min"]
    fig, axes = plt.subplots(1, 4, figsize=(12, 4))
    titles = ["Suhu lingkungan", "Resistansi internal", "Arus rata-rata", "Durasi pengisian"]
    units = ["°C", "mΩ", "A", "menit"]
    for ax, c, title, unit in zip(axes, sensor, titles, units):
        ax.boxplot(train[c].dropna(), widths=.4, flierprops={"markersize": 2, "alpha": .3})
        ax.set(title=title, ylabel=unit, xticks=[])
    fig.suptitle("Rentang dan kandidat outlier sensor — train, tanpa penghapusan")
    fig.tight_layout()
    save(fig, "04_sensor_ranges_train.png")
    return paths


def markdown_table(frame: pd.DataFrame) -> str:
    def cell(value):
        if pd.isna(value):
            return "—"
        return str(value).replace("|", "/").replace("\n", " ")
    lines = ["| " + " | ".join(map(str, frame.columns)) + " |", "| " + " | ".join(["---"]*len(frame.columns)) + " |"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join(lines)


def write_report(summary, schema, contract, counts, train, bands, outliers, output):
    temporal = pd.DataFrame(summary["temporal_counter_checks"]).T.rename_axis("counter").reset_index()
    missing = schema.loc[schema["missing_count"].gt(0), ["column", "missing_count", "missing_pct"]].sort_values("missing_count", ascending=False)
    report = f"""# Audit Dataset dan EDA Awal Train

## Keputusan utama

- Tugas: **regresi SOH saat observasi pada baterai baru**, target `{TARGET}`.
- Data sintetis menurut README sumber, bukan telemetry armada nyata. Tidak ada model dilatih pada tahap ini.
- Seluruh data: **{summary['rows']:,} baris, {summary['columns']} kolom, {summary['unique_batteries']:,} baterai**.
- Split 70/15/15 berdasarkan **battery_id**, seed {SEED}; tidak ada baterai melintasi subset. Persentase ini berlaku pada baterai; jumlah baris tidak harus tepat 70/15/15.
- EDA distribusi/relasi dan batas IQR hanya memakai train. Pemeriksaan global dibatasi pada integritas skema, ID, tanggal, missing, rentang dasar, dan identitas target; test belum digunakan untuk seleksi atau penilaian model.
- `{len(contract['primary_features'])}` kolom kandidat baseline **sebelum encoding**. Jumlah neuron input baru diketahui setelah preprocessing fit pada train.

## Sumber dan versi

[Dataset Kaggle]({SOURCE}). Definisi bersumber dari `data/raw/README.md` dan `data/raw/data_dictionary .csv` dalam arsip yang diunduh. README sumber menyebut lisensi **CC0** dan data sepenuhnya sintetis. Nama kamus dalam arsip memiliki spasi sebelum `.csv`; tidak diubah. Arsip tidak berisi sample notebook meskipun README menyebutnya.

SHA-256 CSV: `{contract['dataset_sha256']}`.
Lingkungan dan hash dokumentasi sumber tersimpan di `provenance.json`. Manifest split di `../../data/splits/battery_split.csv` dikunci ke versi CSV ini. Semua hasil dapat dibuat ulang tanpa mengubah CSV.

## Manifest split

{markdown_table(counts.round(3))}

Manifest menyimpan satu baris per baterai. Untuk mengambil baris data, lakukan merge/map pada battery_id; jangan melakukan random split per observasi. Algoritma menggunakan NumPy default_rng(42), ID diurutkan sebelum permutasi, dan urutan train → validation → test. Tidak ada stratifikasi berdasarkan target dan tidak memilih seed dari skor model.

## Integritas data global

- Duplikasi seluruh baris: {summary['duplicate_rows']}; duplikasi observation_id: {summary['duplicate_observation_ids']}.
- Target kosong: {summary['missing_targets']}; nilai numerik tak hingga: {summary['infinite_numeric_values']}.
- observation_date kosong: {summary['missing_observation_dates']}. Duplikasi battery_id + tanggal pada baris dengan tanggal terisi: {summary['duplicate_battery_date_non_null']}. Missing tanggal tidak dianggap sebagai pengukuran pada tanggal yang sama.
- Format tanggal yang terisi gagal diparse: {sum(summary['date_parse_errors'].values())}.
- Observasi sebelum manufaktur: {summary['observation_before_manufacture']}; servis sesudah observasi: {summary['service_after_observation']}; servis sebelum manufaktur: {summary['service_before_manufacture']}.
- Usia tidak sesuai selisih tanggal: {summary['age_date_mismatches']} dari {summary['age_check_eligible']:,} baris yang dapat diperiksa.
- Jumlah pengamatan per baterai: {int(summary['observations_per_battery']['min'])}–{int(summary['observations_per_battery']['max'])}, rerata {summary['observations_per_battery']['mean']:.2f}. README menyebut sekitar 10–40; angka audit aktual yang dipakai.
- Tidak ditemukan konflik nilai nonkosong dalam atribut statis yang diperiksa (kimia, produsen, model, kapasitas nominal, tanggal manufaktur, iklim). Ini tidak membuktikan semua kolom selalu konstan.

## Missing value

Tabel berikut merupakan audit kualitas seluruh CSV, bukan statistik imputer. Statistik imputasi harus dihitung ulang hanya dari train.

{markdown_table(missing.round(3))}

`last_service_date` tidak selalu hilang karena kerusakan data: {summary['service_missing_count_zero']:,} baris tanggal servis kosong memiliki service_count=0, {summary['service_missing_count_missing']:,} juga memiliki service_count kosong, dan {summary['service_missing_count_positive']:,} memiliki service_count>0. Jangan mengimputasi semua tanggal sebagai tanggal servis nyata. Tanggal ini dikeluarkan dari baseline.

## Kebocoran target dan kandidat proxy

Pada {summary['fade_identity_eligible']:,} baris yang memiliki capacity_fade_pct, **capacity_fade_pct + SOH = 100** dengan maksimum selisih absolut {summary['fade_identity_max_abs_error']}. Ini identitas langsung, sehingga capacity_fade_pct dilarang menjadi input.

Tiga target alternatif (`remaining_useful_life_cycles`, `failure_risk_label`, `degradation_category`) juga dikeluarkan. README sumber menyatakan pengukuran listrik diturunkan dari SOH dan kimia, dengan internal resistance disebut eksplisit. Karena itu internal resistance, voltage, current, dan charge time dikeluarkan secara konservatif dari baseline dan hanya menjadi kandidat ablation sensor terpisah. Tidak diasumsikan semua empat kolom punya rumus ketergantungan yang sama. Sensor terukur dapat sah untuk estimasi saat ini jika benar tersedia, tetapi data sintetis ini membuat interpretasi proxy perlu kehati-hatian.

Daftar fitur beserta alasan tersimpan di `../../configs/feature_contract.json` dan `schema_audit.csv`. Fitur utama meliputi kimia/metadata, umur/pemakaian, dan lingkungan. Availability seluruh ringkasan pada saat observasi adalah asumsi; dokumentasi tidak menjelaskan interval agregasi sehingga belum sah untuk klaim prediksi sebelum sesi.

## Kelayakan temporal

{markdown_table(temporal)}

Pemeriksaan membandingkan pengukuran valid berurutan setelah mengeluarkan tanggal/nilai counter yang kosong, dalam baterai yang sama, dan hanya saat tanggal bertambah. Pasangan bisa melompati observasi yang kosong. cycle_count dan throughput yang didefinisikan kumulatif ternyata sering menurun. Fakta ini perlu dibahas sebagai ketidakkonsistenan data sintetis, bukan langsung diperbaiki dengan sorting nilai, cumulative maximum, atau penghapusan massal.

**Keputusan:** baseline tabular tanpa timesteps. Forecasting/windowing ditunda sampai kualitas urutan, nilai hilang, dan makna horizon ditinjau. Group split ini menguji baterai yang belum terlihat, bukan jaminan generalisasi ke periode masa depan.

## EDA awal train

Jumlah train: {len(train):,}. SOH train: min {train[TARGET].min():.2f}, median {train[TARGET].median():.2f}, rerata {train[TARGET].mean():.2f}, maksimum {train[TARGET].max():.2f} persen.

{markdown_table(bands.round(3))}

Sebaran target tidak merata. Karena target kontinu, SMOTE/balancing kelas tidak diterapkan. Evaluasi akhir harus menyertakan error per rentang SOH beserta jumlah sampelnya; jangan hanya mengandalkan MAE agregat.

![Missing train](01_missing_train.png)

Missing tersebar pada fitur numerik maupun kategori. Median numerik dan kategori missing eksplisit layak menjadi baseline, tetapi belum ada imputer/scaler yang di-fit dalam audit ini.

![Target dan kimia](02_target_chemistry_train.png)

Komposisi kimia dihitung per observasi, bukan per baterai. Observasi satu baterai saling berkaitan. Proporsi baris tidak boleh ditafsirkan sebagai prevalensi armada nyata.

![Relasi fitur-target](03_relationships_train.png)

Grafik memakai sampel deterministik maksimal 5.000 baris train dan mengabaikan pasangan kosong saat plotting. Pola menunjukkan hubungan observasional dalam simulator, bukan bukti sebab akibat. Resistance ditampilkan untuk membahas risiko proxy, bukan dimasukkan diam-diam ke baseline.

![Rentang sensor](04_sensor_ranges_train.png)

Titik di luar whisker adalah kandidat pemeriksaan, bukan otomatis data salah. Tabel `outlier_flags_train.csv` memakai aturan 1.5×IQR train, tanpa menghapus atau memotong nilai. Outlier sensor diinjeksi menurut sumber, tetapi sumber tidak menyediakan mask baris mana yang diinjeksi.

## Langkah berikutnya

1. Terima skenario estimasi SOH saat ini pada baterai baru dan daftar fitur utama, atau catat perubahan sebelum training.
2. Perdalam EDA **train** pada notebook 01: distribusi setiap fitur, korelasi/redundansi, kategori langka, error domain, dan temuan yang memengaruhi seleksi.
3. Buat pipeline imputasi, encoding kategori tak dikenal, standardisasi numerik, dan penanganan recall_flag. Fit hanya pada train. Jangan melatih model sebelum pipeline dan input diuji.
4. Catat p setelah encoding, lalu buat baseline MLP dan matriks eksperimen sesuai CONTEXT.md. Belum memilih framework atau mengklaim performa.

Artefak ini berisi audit dan EDA awal, **bukan EDA lengkap, preprocessing final, atau hasil prediksi**.
"""
    (output / "AUDIT_REPORT.md").write_text(report, encoding="utf-8")


def run_audit(root: Path = ROOT) -> dict:
    frame, dictionary, sha = load_dataset(root)
    contract = feature_contract(frame, sha)
    manifest, row_split, counts = build_split(frame)
    schema, summary = audit_quality(frame, dictionary, contract)
    train = frame.loc[row_split.eq("train")].copy()
    stats, outliers, bands = profile_train(train, contract)
    output = root / "results/audit"
    output.mkdir(parents=True, exist_ok=True)
    (root / "configs").mkdir(exist_ok=True)
    (root / "data/splits").mkdir(exist_ok=True)
    manifest_path = root / "data/splits/battery_split.csv"
    if manifest_path.exists():
        old = pd.read_csv(manifest_path)
        if not old.equals(manifest):
            raise ValueError("Manifest sudah ada dan berbeda; jangan menimpa split yang terkunci.")
    manifest.to_csv(manifest_path, index=False, lineterminator="\n")
    for name, table in {"schema_audit": schema, "split_summary": counts, "numeric_train": stats, "outlier_flags_train": outliers, "target_bands_train": bands}.items():
        table.to_csv(output / f"{name}.csv", index=False, lineterminator="\n")
    for path, value in [(root / "configs/feature_contract.json", contract), (output / "quality_summary.json", summary)]:
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    provenance = {
        "source_url": SOURCE, "dataset_sha256": sha,
        "source_documentation_sha256": {name: hashlib.sha256((root / "data/raw" / name).read_bytes()).hexdigest() for name in ["README.md", "data_dictionary .csv"]},
        "split_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "packages": {p: version(p) for p in ["numpy", "pandas", "matplotlib", "nbformat", "nbclient", "ipykernel"]},
        "algorithm": "numpy.random.default_rng(42).permutation(sorted(unique battery_id)); first 70%, next 15%, remainder test",
        "scope": "Global integrity checks; train-only exploratory profiles/plots. No preprocessing fit or modeling.",
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    make_figures(train, contract, output)
    write_report(summary, schema, contract, counts, train, bands, outliers, output)
    return {"frame": frame, "dictionary": dictionary, "contract": contract, "manifest": manifest, "row_split": row_split, "split_summary": counts, "schema": schema, "quality": summary, "train": train, "statistics_train": stats, "outliers_train": outliers, "bands_train": bands, "output": output}


if __name__ == "__main__":
    result = run_audit()
    print(result["split_summary"].to_string(index=False))
    print("Primary raw features:", len(result["contract"]["primary_features"]))
    print("Report:", result["output"] / "AUDIT_REPORT.md")
