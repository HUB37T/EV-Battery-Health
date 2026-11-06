"""Train-only fitted preprocessing, reusable at inference and in fold-local CV.

Run from repository root: python -m src.preprocessing
"""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from src.data_audit import ROOT, TARGET, load_dataset, markdown_table

MISSING = "__MISSING__"
NONNEGATIVE = {
    "battery_age_days", "cycle_count", "avg_daily_distance_km", "avg_charging_power_kw",
    "charge_cycles_per_month", "total_energy_throughput_kwh", "service_count",
}
BOUNDS = {"fast_charge_ratio": (0, 1), "avg_depth_of_discharge_pct": (0, 100),
          "avg_state_of_charge_pct": (0, 100), "humidity_pct": (0, 100)}
FORBIDDEN = {"battery_id", "observation_id", TARGET, "capacity_fade_pct", "remaining_useful_life_cycles",
             "failure_risk_label", "degradation_category", "manufacture_date", "observation_date",
             "last_service_date", "internal_resistance_mohm", "voltage_avg_v", "current_avg_a", "charge_time_avg_min"}


class BatteryPreprocessor(TransformerMixin, BaseEstimator):
    """Strict named-column input -> dense float32 MLP input; target never used.

    Numeric, categorical, and binary rules have fixed output order. Unknown
    categories never refit the encoder. Missing indicators exist even if a
    feature happened to be complete during fit. No cross-row fill is performed.
    """

    def __init__(self, numeric_features, categorical_features, binary_features,
                 log_features=(), clip_features=(), clip_quantiles=(.005, .995)):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.binary_features = binary_features
        self.log_features = log_features
        self.clip_features = clip_features
        self.clip_quantiles = clip_quantiles

    @property
    def required_columns(self):
        return list(self.numeric_features) + list(self.binary_features) + list(self.categorical_features)

    def _clean(self, X):
        if not isinstance(X, pd.DataFrame) or not X.columns.is_unique:
            raise TypeError("Input harus DataFrame dengan nama kolom unik.")
        required = self.required_columns
        missing, extra = set(required)-set(X.columns), set(X.columns)-set(required)
        if missing or extra:
            raise ValueError(f"Skema fitur berbeda: missing={sorted(missing)}, extra={sorted(extra)}")
        if len(required) != len(set(required)) or set(required) & FORBIDDEN:
            raise ValueError("Kolom duplikat, ID, target, tanggal, atau proxy dilarang dalam pipeline utama.")
        cleaned = X.loc[:, required].copy()
        invalid = {}
        for c in list(self.numeric_features) + list(self.binary_features):
            raw = cleaned[c]
            value = pd.to_numeric(raw, errors="coerce").astype(float)
            value = value.where(np.isfinite(value))
            if c in NONNEGATIVE:
                value = value.where(value.ge(0))
            if c == "pack_capacity_kwh":
                value = value.where(value.gt(0))
            if c in BOUNDS:
                value = value.where(value.between(*BOUNDS[c]))
            if c == "service_count":
                value = value.where(np.isclose(value, np.round(value), equal_nan=False))
            if c in self.binary_features:
                value = value.where(value.isin([0, 1]))
            invalid[c] = int((raw.notna() & value.isna()).sum())
            cleaned[c] = value
        for c in self.categorical_features:
            value = cleaned[c].astype("string").str.strip()
            if value.eq(MISSING).any():
                raise ValueError(f"Token {MISSING} reserved untuk missing, bukan kategori asli {c}.")
            invalid[c] = int((value.notna() & value.eq("")).sum())
            cleaned[c] = value.mask(value.eq("")).fillna(MISSING).astype(str)
        return cleaned, invalid

    def _numeric_values(self, clean):
        numeric = clean[list(self.numeric_features)].fillna(self.medians_).copy()
        for c in self.clip_features:
            numeric[c] = numeric[c].clip(self.clip_bounds_[c][0], self.clip_bounds_[c][1])
        for c in self.log_features:
            numeric[c] = np.sign(numeric[c])*np.log1p(np.abs(numeric[c]))
        return numeric

    def fit(self, X, y=None):
        # Accept y for sklearn Pipeline compatibility, but never inspect/use it.
        del y
        clean, invalid = self._clean(X)
        if not len(clean):
            raise ValueError("Train kosong.")
        if not set(self.log_features).issubset(self.numeric_features) or not set(self.clip_features).issubset(self.numeric_features):
            raise ValueError("Log/cap hanya berlaku untuk fitur numerik yang dipilih.")
        qlo, qhi = self.clip_quantiles
        if not 0 <= qlo < qhi <= 1:
            raise ValueError("Quantile clipping tidak valid.")
        self.medians_ = clean[list(self.numeric_features)].median()
        if self.medians_.isna().any():
            raise ValueError("Ada fitur numerik seluruhnya kosong pada fit; perlu keputusan eksplisit.")
        self.binary_modes_ = {}
        for c in self.binary_features:
            mode = clean[c].mode()
            if mode.empty:
                raise ValueError(f"Fitur biner {c} seluruhnya kosong pada fit.")
            self.binary_modes_[c] = float(mode.iloc[0])
        self.clip_bounds_ = {c: tuple(float(v) for v in clean[c].quantile([qlo, qhi])) for c in self.clip_features}
        self.scaler_ = StandardScaler().fit(self._numeric_values(clean))
        # Always allocate a distinct missing category, including future missing.
        categories = [sorted(set(clean[c]) | {MISSING}) for c in self.categorical_features]
        self.encoder_ = OneHotEncoder(categories=categories, handle_unknown="ignore", sparse_output=False, dtype=np.float32)
        self.encoder_.fit(clean[list(self.categorical_features)])
        numeric_names = [f"num__{c}" for c in self.numeric_features]
        numeric_missing = [f"missing__{c}" for c in self.numeric_features]
        binary_names = [f"binary__{c}" for c in self.binary_features]
        binary_missing = [f"missing__{c}" for c in self.binary_features]
        category_names = [f"cat__{c}" for c in self.encoder_.get_feature_names_out(self.categorical_features)]
        self.output_features_ = np.array(numeric_names + numeric_missing + binary_names + binary_missing + category_names, dtype=str)
        if len(set(self.output_features_)) != len(self.output_features_):
            raise ValueError("Nama fitur output tidak unik.")
        self.fit_rows_ = len(clean)
        self.invalid_counts_train_ = invalid
        self.feature_names_in_ = np.array(self.required_columns, dtype=object)
        self.n_features_in_ = len(self.required_columns)
        self.n_features_out_ = len(self.output_features_)
        return self

    def transform(self, X):
        check_is_fitted(self, "output_features_")
        clean, _ = self._clean(X)
        numeric = self.scaler_.transform(self._numeric_values(clean))
        num_missing = clean[list(self.numeric_features)].isna().to_numpy(dtype=np.float32)
        binary = clean[list(self.binary_features)].fillna(self.binary_modes_).to_numpy(dtype=np.float32)
        binary_missing = clean[list(self.binary_features)].isna().to_numpy(dtype=np.float32)
        categorical = self.encoder_.transform(clean[list(self.categorical_features)])
        result = np.hstack([numeric, num_missing, binary, binary_missing, categorical]).astype(np.float32)
        if result.shape != (len(X), self.n_features_out_) or not np.isfinite(result).all():
            raise ValueError("Output shape berubah atau mengandung nilai nonfinite.")
        return result

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "output_features_")
        return self.output_features_.copy()

    def diagnostic(self, X):
        check_is_fitted(self, "output_features_")
        clean, invalid = self._clean(X)
        unknown = {c: int((~clean[c].isin(self.encoder_.categories_[i])).sum()) for i, c in enumerate(self.categorical_features)}
        capped = {c: int((clean[c].notna() & ~clean[c].between(*self.clip_bounds_[c])).sum()) for c in self.clip_features}
        return {"invalid_to_missing": invalid, "unknown_categories": unknown, "clipped_rows_by_feature": capped}


def read_settings(root=ROOT):
    contract = json.loads((root/"configs/feature_contract.json").read_text(encoding="utf-8"))
    settings = json.loads((root/"configs/preprocessing.json").read_text(encoding="utf-8"))
    if settings["categorical_missing_token"] != MISSING:
        raise ValueError("Token missing pada konfigurasi tidak cocok dengan versi pipeline.")
    return contract, settings


def make_preprocessor(recipe, contract, settings):
    spec = settings["recipes"][recipe]
    dropped = set(spec["drop_features"])
    if not dropped.issubset(contract["primary_features"]):
        raise ValueError("Daftar fitur yang dikeluarkan tidak dikenal.")
    return BatteryPreprocessor(
        numeric_features=[c for c in contract["numeric_features"] if c not in dropped],
        categorical_features=[c for c in contract["categorical_features"] if c not in dropped],
        binary_features=[c for c in contract["binary_features"] if c not in dropped],
        log_features=spec["log_features"], clip_features=spec["clip_features"],
        clip_quantiles=spec["clip_quantiles"],
    )


def load_splits(root=ROOT):
    frame, _, sha = load_dataset(root)
    manifest_path = root/"data/splits/battery_split.csv"
    audit = json.loads((root/"results/audit/provenance.json").read_text(encoding="utf-8"))
    if sha != audit["dataset_sha256"] or hashlib.sha256(manifest_path.read_bytes()).hexdigest() != audit["split_manifest_sha256"]:
        raise ValueError("Dataset/manifest berbeda dari audit yang dikunci.")
    manifest = pd.read_csv(manifest_path)
    if not manifest.battery_id.is_unique:
        raise ValueError("Manifest baterai tidak unik.")
    labels = frame.battery_id.map(manifest.set_index("battery_id").split)
    if labels.isna().any() or set(labels) != {"train", "validation", "test"}:
        raise ValueError("Observasi tidak tercakup oleh split yang dikunci.")
    splits = {s: frame.loc[labels.eq(s)].copy() for s in ["train", "validation", "test"]}
    groups = {s: set(f.battery_id) for s, f in splits.items()}
    if groups["train"] & groups["validation"] or groups["train"] & groups["test"] or groups["validation"] & groups["test"]:
        raise ValueError("Baterai melintasi split.")
    return splits, audit


def parameter_tables(prep):
    numeric = []
    for i,c in enumerate(prep.numeric_features):
        lo,hi = prep.clip_bounds_.get(c,(np.nan,np.nan))
        numeric.append({"feature":c,"median_train_original":prep.medians_[c],"cap_lower_train":lo,"cap_upper_train":hi,
                        "signed_log1p":c in prep.log_features,"scaler_mean_after_transform_train":prep.scaler_.mean_[i],
                        "scaler_scale_after_transform_train":prep.scaler_.scale_[i]})
    categories = [{"feature":c,"category":value,"block_index":j} for i,c in enumerate(prep.categorical_features) for j,value in enumerate(prep.encoder_.categories_[i])]
    output = pd.DataFrame({"output_index":range(prep.n_features_out_),"output_feature":prep.get_feature_names_out()})
    return pd.DataFrame(numeric),pd.DataFrame(categories),output


def load_prepared_split(recipe, split, root=ROOT):
    """Load trusted local artefacts; reject accidental mixing of recipes/runs."""
    _,settings=read_settings(root)
    if recipe not in settings["recipes"] or split not in {"train","validation","test"}:
        raise ValueError("Resep/split tidak dikenal.")
    bundle=joblib.load(root/"models/preprocessing"/f"{recipe}.joblib")
    with np.load(root/"data/processed"/recipe/f"{split}.npz",allow_pickle=False) as stored:
        data={key:stored[key] for key in stored.files}
    actual_state=joblib.hash((bundle["preprocessor"],bundle["target_scaler"]))
    if data["recipe"].item()!=recipe or data["split"].item()!=split or bundle["recipe"]!=recipe:
        raise ValueError("Identitas resep/split berbeda.")
    if actual_state!=bundle["state_hash"] or data["pipeline_state_hash"].item()!=actual_state:
        raise ValueError("Array dan fitted pipeline berasal dari run/state berbeda.")
    if not np.array_equal(data["feature_names"],bundle["feature_names"]):
        raise ValueError("Urutan fitur array dan pipeline berbeda.")
    if data["X"].shape!=(len(data["observation_id"]),len(bundle["feature_names"])) or not np.isfinite(data["X"]).all():
        raise ValueError("Array tidak valid.")
    n=len(data["observation_id"])
    if data["X"].dtype!=np.float32 or len(data["battery_id"])!=n:
        raise ValueError("Dtype atau panjang identitas array tidak sesuai.")
    for key in ["y_scaled","y_original"]:
        if data[key].shape!=(n,) or not np.isfinite(data[key]).all():
            raise ValueError("Target array tidak valid.")
    return data,bundle


def make_figures(train, variants, output):
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    figures=[]
    def save(fig,name):
        fig.savefig(output/name,dpi=140,bbox_inches="tight",facecolor="white")
        plt.close(fig);figures.append(name)
    selected=["cycle_count","avg_ambient_temp_c","total_energy_throughput_kwh"]
    labels={"cycle_count":"Siklus ekuivalen (siklus)","avg_ambient_temp_c":"Suhu lingkungan (°C)","total_energy_throughput_kwh":"Energi kumulatif (kWh)"}
    fig,axes=plt.subplots(3,3,figsize=(12,10))
    for row,c in enumerate(selected):
        axes[row,0].hist(train[c].dropna(),bins=40,color="#247e88",edgecolor="white")
        axes[row,0].set_title("Nilai terisi sebelum preprocessing",fontsize=9)
        axes[row,0].set_xlabel(labels[c])
        for col,recipe in enumerate(["baseline_standard","log_sensor_caps"],1):
            prep=variants[recipe]["preprocessor"]
            idx=list(prep.get_feature_names_out()).index(f"num__{c}")
            axes[row,col].hist(variants[recipe]["train_X"][:,idx],bins=40,color="#3674b5" if col==1 else "#c47a21",edgecolor="white")
            axes[row,col].set_title(recipe,fontsize=9)
            axes[row,col].set_xlabel("Nilai terstandar")
        for ax in axes[row]:ax.set_ylabel("Observasi train")
    fig.suptitle("Efek transformasi pada train: tidak ada observasi dibuang")
    fig.tight_layout(rect=(0,0,1,.97));save(fig,"01_before_after_train.png")
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for recipe in ["baseline_standard","log_sensor_caps"]:
        prep=variants[recipe]["preprocessor"]
        values=variants[recipe]["train_X"][:,:len(prep.numeric_features)]
        axes[0].plot(range(len(prep.numeric_features)),values.mean(axis=0),"o-",label=recipe)
        axes[1].plot(range(len(prep.numeric_features)),values.std(axis=0),"o-",label=recipe)
    axes[0].axhline(0,color="gray",ls="--");axes[1].axhline(1,color="gray",ls="--")
    axes[0].set(title="Mean numerik train sesudah scaling",ylabel="Mean")
    axes[1].set(title="Std numerik train sesudah scaling (ddof=0)",ylabel="Standard deviation")
    for ax in axes:
        ax.set_xlabel("Indeks fitur numerik (lihat tabel parameter)");ax.legend(fontsize=8);ax.ticklabel_format(axis="y",style="plain",useOffset=False)
    axes[0].set_ylim(-.01,.01);axes[1].set_ylim(.99,1.01)
    fig.tight_layout();save(fig,"02_scaler_train_checks.png")
    fig,ax=plt.subplots(figsize=(8,4))
    names=list(variants);counts=[variants[n]["preprocessor"].n_features_out_ for n in names]
    bars=ax.barh(names,counts,color=["#247e88","#3674b5","#c47a21"])
    for bar,count in zip(bars,counts):ax.text(count+.5,bar.get_y()+bar.get_height()/2,str(count),va="center")
    ax.set(xlabel="Neuron input setelah preprocessing",title="Dimensi input MLP per resep (belum ada model)",xlim=(0,max(counts)+8))
    fig.tight_layout();save(fig,"03_input_dimensions.png")
    return figures


def write_report(summary,checks,target_scaler,settings,variants,output):
    dimensions=summary.loc[summary.split.eq("train"),["recipe","rows","raw_features","input_neurons","numeric_missing_after","finite"]]
    recipes=[]
    for name,entry in variants.items():
        prep=entry["preprocessor"]
        diag=prep.diagnostic(entry["train_raw"])
        recipes.append({"recipe":name,"raw_features":len(prep.required_columns),"input_neurons":prep.n_features_out_,"capped_train_feature_cells":sum(diag["clipped_rows_by_feature"].values()),"log_features":", ".join(prep.log_features) or "Tidak ada"})
    report=f"""# Preprocessing untuk MLP Backpropagation

## Status dan tujuan

Pipeline sudah di-fit **hanya pada 34.929 observasi / 1.820 baterai train**. Validation (7.499 baris) dan test (7.572 baris) hanya ditransform memakai parameter yang telah dibekukan. Tidak ada MLP yang dilatih, tuning, pemilihan berdasarkan test, atau klaim bahwa overfitting/underfitting sudah teratasi.

Preprocessing membantu skala optimasi, menangani ketiadaan informasi, dan menyediakan ablation redundansi. Generalisasi tetap harus dinilai dari train–validation learning curves, kapasitas MLP, regularisasi, early stopping, dan variasi seed. Daftar kandidat fitur awal dan manifest split tidak diubah.

## Tiga resep yang dapat dibandingkan nanti

{markdown_table(pd.DataFrame(recipes))}

Default awal: **{settings['default_recipe']}**, karena mempertahankan seluruh 21 fitur sah sebagai acuan. Resep lain adalah alternatif yang telah ditetapkan sebelum training, bukan pemenang eksperimen. Full vs log_sensor_caps menguji gabungan log/cap; efek log dan cap belum diisolasi satu per satu. Compact vs log_sensor_caps menguji pengurangan dua fitur. Jika ingin atribusi efek log vs cap secara terpisah, tambahkan eksperimen terkontrol ketika training.

## 1. Urutan pemrosesan

1. Validasi skema bernama: ID, target, label turunan, tanggal, dan proxy sensor tidak boleh ikut masuk pipeline utama. Urutan kolom boleh berbeda; kolom wajib hilang/kolom tambahan ditolak.
2. Konversi numerik dan pemeriksaan nilai finite. Nilai yang melanggar batas definisi (rasio, persentase, biner, nonnegatif tertentu) dijadikan missing dengan catatan; nilai ekstrem yang masih sah tidak otomatis dibuang. Altitude negatif tetap sah. Tidak melakukan perbaikan monoton, forward/backward fill, atau pengisian metadata lintas waktu/baterai.
3. Numerik: imputasi median train. Indikator missing tetap dibuat untuk setiap fitur numerik, termasuk bila lengkap saat fit, agar missing baru pada inferensi dapat ditangani.
4. Pada resep alternatif: cap hanya sensor yang ditetapkan menggunakan quantile 0.5% dan 99.5% **nilai train terisi sebelum imputasi**. Tidak menyaring seluruh baris, tidak menghapus baterai tua atau SOH rendah. Cap adalah hipotesis ketahanan terhadap ekor, bukan identifikasi pasti kesalahan sensor.
5. Signed log1p untuk fitur berekor kanan terpilih. Untuk nilai nonnegatif sama dengan log1p; untuk altitude negatif memakai sign(x)×log1p(abs(x)). Bukan log target.
6. StandardScaler fit pada numerik train **setelah** imputasi/cap/log. Mean mendekati 0 dan std populasi mendekati 1 pada train yang bervariasi. Validation/test tidak dipaksa mean=0/std=1 dan tidak diskalakan ulang.
7. Recall flag diimputasi modus train dan tetap 0/1, dengan indikator missing. Flag langka tidak terkena aturan outlier IQR.
8. Kategori: whitespace pinggir dibersihkan; kosong memakai kategori eksplisit `{MISSING}`; one-hot tanpa urutan ordinal buatan. Kategori baru memakai blok all-zero; tidak menjadi kategori missing dan tidak memperluas dimensi. Token missing dialokasikan sejak fit, termasuk bila suatu fitur lengkap pada train. Unknown count tersedia lewat diagnostic.
9. Gabungkan blok numerik, indikator numerik, biner, indikator biner, lalu one-hot ke array dense **float32**. Urutan/nama output disimpan. Target disimpan terpisah.

Kategori tidak diisi dari pengamatan masa depan atau nilai grup baterai seperti label pengelompokan EDA. Pipeline inferensi hanya memerlukan satu baris fitur yang tersedia saat prediksi.

## 2. Feature selection yang dapat dipertanggungjawabkan

- capacity_fade_pct (100−SOH), RUL, label risiko/degradasi, dan ID tetap dikeluarkan untuk menghindari kebocoran/hafalan. Empat proxy sensor tetap terpisah dari lingkup utama.
- Resep compact mengeluarkan total_energy_throughput_kwh karena korelasi Spearman train dengan cycle_count 0.984, serta peak_temp_during_fast_charge_c karena korelasi dengan suhu lingkungan 0.920.
- Umur dan siklus tetap dipertahankan meski berkorelasi, karena mewakili calendar aging dan penggunaan berbeda. Fitur dengan korelasi target kecil tidak otomatis dihapus: MLP dapat memanfaatkan interaksi.
- Pengurangan redundansi dapat mengurangi input/parameter jaringan, tetapi juga bisa membuang informasi. Karena itu compact **bukan feature selection final**; pilih resep melalui validation kelak.

## 3. Target dan balancing

SOH asli dipertahankan, tanpa clipping target, binning menjadi kelas, oversampling, SMOTE, atau penghapusan SOH rendah. Balancing kelas tidak relevan untuk target kontinu. Distribusi evaluasi mengikuti split semula.

Target scaler fit hanya pada y_train: mean **{target_scaler.mean_[0]:.6f}**, scale **{target_scaler.scale_[0]:.6f}** poin persentase. Target terstandar memudahkan MLP output linear memulai di sekitar nol. Ini tidak memperbaiki distribusi ekor secara otomatis dan tidak menjamin peningkatan skor.

Output array: `y_scaled` untuk optimasi dan `y_original` untuk evaluasi. Kembalikan prediksi melalui `bundle['target_scaler'].inverse_transform(...)`. MAE/RMSE wajib dilaporkan dalam poin persentase SOH asli. Bila membandingkan Huber, nilai delta harus dijelaskan pada skala target yang dipakai; perubahan skala juga mengubah besar loss/gradien.

## 4. Bukti teknis kesiapan input

{markdown_table(dimensions)}

Seluruh split dipertahankan dengan pasangan ID/X/y yang sama; ringkasan holdout terbatas pada shape dan validitas input, tanpa distribusi target atau penilaian model. Semua array finite, nama output unik dan konsisten, serta target round-trip diperiksa. Detail semua split di `split_readiness.csv`.

![Sebelum sesudah](01_before_after_train.png)

Grafik hanya train. Spike pada nilai median sebagian merupakan konsekuensi imputasi; indikator missing mempertahankan informasi ketiadaan, bukan berpura-pura median adalah pengukuran asli. Log/cap dapat meredam ekor dan sekaligus menghilangkan detail—bandingkan resep pada validation nanti.

![Scaling](02_scaler_train_checks.png)

Grafik mean/std mengacu pada blok numerik saja; one-hot, biner, dan indikator tidak dipaksa unit variance. Standardisasi per fitur tidak menghilangkan korelasi antarf fitur atau membuat matriks input sepenuhnya bebas redundansi.

![Dimensi input](03_input_dimensions.png)

Jumlah neuron input dihitung **setelah preprocessing**, termasuk indikator dan one-hot. Nama, indeks, dan parameter masing-masing resep tersedia pada tabel CSV. Untuk jaringan hidden [64,32] output satu, parameter dihitung sebagai `(p×64+64)+(64×32+32)+(32×1+1)`; ini ilustrasi kapasitas, bukan model yang sudah dilatih.

## 5. Pemeriksaan dan kegagalan yang ditangani

{markdown_table(checks)}

Pengujian otomatis tambahan pada `tests/test_preprocessing.py` mencakup parameter yang benar-benar berasal dari train, kategori/missing baru, nilai tidak valid, urutan kolom, skema bocor, serialisasi, serta memastikan transform holdout tidak mengubah state fit. Kasus synthetic kecil dipakai untuk pemeriksaan perilaku; bukan sampel pengganti dataset atau hasil eksperimen model.

## 6. Artefak dan pemakaian

- `data/processed/<recipe>/train.npz`, `validation.npz`, `test.npz`: X float32, y_scaled, y_original, observation_id, battery_id, dan feature_names; disimpan lokal, diabaikan Git.
- `models/preprocessing/<recipe>.joblib`: bundle berisi preprocessor, target_scaler, resep, identitas sumber/split, nama fitur, dan versi paket; lokal, diabaikan Git. Hanya muat joblib dari artefak tepercaya. load_prepared_split memeriksa identitas resep/split, urutan fitur, dan hash state agar array tidak tercampur dengan pipeline lain.
- `results/preprocessing/`: laporan, pemeriksaan, tiga gambar train, parameter train, kategori, nama output, dan provenance; dapat dibagikan lewat Git.
- `configs/preprocessing.json`: definisi resep yang dapat direview. Kontrak fitur awal tidak ditimpa.

```python
from src.preprocessing import load_prepared_split
data, bundle = load_prepared_split('baseline_standard', 'train')
X_train, y_train = data['X'], data['y_scaled']
# Untuk data baru gunakan kolom mentah persis yang diperlukan:
# X_new = bundle['preprocessor'].transform(new_rows[bundle['required_columns']])
# y_pred_soh = bundle['target_scaler'].inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
```

Precomputed arrays khusus split tetap saat ini. Jika memakai cross-validation di dalam train, **clone/buat ulang dan fit preprocessor serta target scaler pada setiap fold train**; jangan memakai array yang sudah di-fit pada seluruh train untuk fold-validation. Library scikit-learn di sini hanya menyediakan transformasi; tidak ada model prediksi di luar MLP.

Reproduksi: `python -m src.preprocessing`, lalu `python scripts/run_notebook.py notebooks/02_preprocessing.ipynb`. Pilih lingkungan dengan requirements.txt. Tidak mengubah CSV asli, manifest split, atau konfigurasi fitur awal.

## 7. Yang masih perlu dibuktikan saat eksperimen

1. Mulai dari baseline_standard untuk acuan, lalu bandingkan alternatif menggunakan arsitektur/seed/kebijakan training yang sama.
2. Train loss tinggi dan validation tinggi dapat menunjukkan underfitting atau masalah optimasi; periksa kapasitas, learning rate, epoch, dan fitur.
3. Train membaik tetapi validation memburuk dapat menunjukkan overfitting; periksa regularisasi L2/dropout, ukuran hidden, early stopping, dan stabilitas seed.
4. Jangan menambah/hapus fitur atau memilih cap berdasarkan test. Test hanya untuk evaluasi final model terkunci.

## Referensi implementasi

- [scikit-learn: common pitfalls dan data leakage](https://scikit-learn.org/stable/common_pitfalls.html).
- [StandardScaler: parameter train dan sensitivitas terhadap outlier](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html).
- [OneHotEncoder: kategori tak dikenal dan handle_unknown](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html).

Versi yang dijalankan tetap scikit-learn 1.7.2 dan tercatat dalam provenance, bukan otomatis mengikuti versi dokumentasi stable terbaru. Temuan dataset merujuk laporan EDA train sebelumnya.
"""
    (output/"PREPROCESSING_REPORT.md").write_text(report,encoding="utf-8",newline="\n")


def run_preprocessing(root: Path=ROOT):
    splits,audit=load_splits(root)
    train=splits["train"]
    contract,settings=read_settings(root)
    target_scaler=StandardScaler().fit(train[[TARGET]])
    target_state_before=joblib.hash(target_scaler)
    output=root/"results/preprocessing";output.mkdir(parents=True,exist_ok=True)
    model_dir=root/"models/preprocessing";model_dir.mkdir(parents=True,exist_ok=True)
    summary=[];checks=[];variants={}
    source_info={"dataset_sha256":audit["dataset_sha256"],"split_manifest_sha256":audit["split_manifest_sha256"],
                 "fit_split":"train","fit_rows":len(train),"fit_batteries":train.battery_id.nunique(),
                 "fit_observation_ids_sha256":hashlib.sha256("\n".join(train.observation_id).encode()).hexdigest(),
                 "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 "settings_sha256":hashlib.sha256((root/"configs/preprocessing.json").read_bytes()).hexdigest(),
                 "packages":{p:version(p) for p in ["numpy","pandas","scikit-learn","joblib","matplotlib"]},
                 "target":{"column":TARGET,"mean_train":float(target_scaler.mean_[0]),"scale_train":float(target_scaler.scale_[0]),"unit":"percentage points for inverse-transformed MAE/RMSE"},
                 "default_recipe":settings["default_recipe"],"selection":"not ranked by model scores; no predictive model fitted"}
    for name in settings["recipes"]:
        prep=make_preprocessor(name,contract,settings)
        X_raw=train.loc[:,prep.required_columns]
        prep.fit(X_raw)
        params,categories,names=parameter_tables(prep)
        params.to_csv(output/f"{name}_numeric_parameters.csv",index=False,lineterminator="\n")
        categories.to_csv(output/f"{name}_categories.csv",index=False,lineterminator="\n")
        names.to_csv(output/f"{name}_output_features.csv",index=False,lineterminator="\n")
        state_before=joblib.hash(prep)
        pipeline_state=joblib.hash((prep,target_scaler))
        data_dir=root/"data/processed"/name;data_dir.mkdir(parents=True,exist_ok=True)
        arrays={}
        max_roundtrip=0.
        for split,frame in splits.items():
            X=prep.transform(frame.loc[:,prep.required_columns])
            ys=target_scaler.transform(frame[[TARGET]]).ravel().astype(np.float32)
            restored=target_scaler.inverse_transform(ys.astype(float).reshape(-1,1)).ravel()
            max_roundtrip=max(max_roundtrip,float(np.max(np.abs(restored-frame[TARGET].to_numpy()))))
            if not np.isfinite(ys).all() or not np.isfinite(X).all():raise ValueError("Nilai nonfinite sesudah transform.")
            np.savez_compressed(data_dir/f"{split}.npz",X=X,y_scaled=ys,y_original=frame[TARGET].to_numpy(),
                                battery_id=frame.battery_id.to_numpy(dtype=str),observation_id=frame.observation_id.to_numpy(dtype=str),
                                feature_names=prep.get_feature_names_out(),recipe=np.array(name),split=np.array(split),
                                pipeline_state_hash=np.array(pipeline_state))
            summary.append({"recipe":name,"split":split,"rows":len(frame),"batteries":frame.battery_id.nunique(),"raw_features":len(prep.required_columns),"input_neurons":X.shape[1],"numeric_missing_after":int(np.isnan(X).sum()),"finite":bool(np.isfinite(X).all()),"dtype":str(X.dtype)})
            arrays[split]=X
        unchanged=state_before==joblib.hash(prep)
        if not unchanged:raise AssertionError("Transform mengubah state fit.")
        if max_roundtrip>1e-5:raise AssertionError("Target inverse error terlalu besar.")
        bundle={"preprocessor":prep,"target_scaler":target_scaler,"recipe":name,"state_hash":pipeline_state,"required_columns":prep.required_columns,"feature_names":prep.get_feature_names_out(),"provenance":source_info}
        bundle_path=model_dir/f"{name}.joblib";joblib.dump(bundle,bundle_path)
        loaded=joblib.load(bundle_path)
        restored_X=loaded["preprocessor"].transform(X_raw.iloc[:16])
        if not np.array_equal(restored_X,arrays["train"][:16]):raise AssertionError("Serialisasi mengubah output.")
        checks.extend([{"recipe":name,"check":"transform_state_unchanged","result":"PASS"},
                       {"recipe":name,"check":"all_split_float32_finite_and_same_width","result":"PASS"},
                       {"recipe":name,"check":"target_inverse_max_error_pp","result":f"{max_roundtrip:.9f}"},
                       {"recipe":name,"check":"serialization_exact_first_16_rows","result":"PASS"}])
        variants[name]={"preprocessor":prep,"train_raw":X_raw,"train_X":arrays["train"],"bundle_path":bundle_path}
    summary=pd.DataFrame(summary);checks=pd.DataFrame(checks)
    if target_state_before != joblib.hash(target_scaler):
        raise AssertionError("Transform mengubah state target scaler.")
    summary.to_csv(output/"split_readiness.csv",index=False,lineterminator="\n")
    checks.to_csv(output/"integrity_checks.csv",index=False,lineterminator="\n")
    figures=make_figures(train,variants,output)
    write_report(summary,checks,target_scaler,settings,variants,output)
    source_info["recipes"]={name:{"raw_features":len(v["preprocessor"].required_columns),"input_neurons":v["preprocessor"].n_features_out_,"train_diagnostic":v["preprocessor"].diagnostic(v["train_raw"])} for name,v in variants.items()}
    (output/"provenance.json").write_text(json.dumps(source_info,indent=2,ensure_ascii=False)+"\n",encoding="utf-8",newline="\n")
    return {"splits":splits,"variants":variants,"summary":summary,"checks":checks,"target_scaler":target_scaler,"settings":settings,"contract":contract,"provenance":source_info,"figures":figures,"output":output}


if __name__=="__main__":
    # Persist objects under their importable module, not __main__.
    from src.preprocessing import run_preprocessing as run_canonical
    result=run_canonical()
    print(result["summary"].to_string(index=False))
    print(result["output"]/"PREPROCESSING_REPORT.md")
