"""Report feature engineering ablations without rewriting the original study."""

import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_audit import ROOT, markdown_table
from src.feature_engineering import load_engineered_data, load_raw_observations
from src.mlp_experiments import load_training_data


NAMES={"control_no_domain_features":"Kontrol tanpa fitur domain","fe_usage_rates":"Rasio penggunaan","fe_context":"Selisih suhu + fast charge–DOD","fe_chemistry":"Interaksi kimia–paparan","fe_combined":"Gabungan semua fitur"}


def read_results(root=ROOT):
    output=root/"results/feature_engineering"
    study=json.loads((output/"study.json").read_text(encoding="utf-8"))
    if study["source_sha256"]!=hashlib.sha256((root/"src/feature_engineering.py").read_bytes()).hexdigest():raise ValueError("Hasil FE berasal dari source versi lain; jalankan study kembali.")
    tables={name:pd.read_csv(output/f"{name}.csv") for name in ["screening","seed_confirmation","seed_summary","paired_seed_improvements","validation_bands","run_records"]}
    return output,study,tables


def build_report(root=ROOT):
    output,study,tables=read_results(root)
    best=study["selected_engineered_variant"]
    summary=tables["seed_summary"].set_index("variant")
    control=summary.loc["control_no_domain_features"];fe=summary.loc[best]
    selected=tables["screening"].set_index("variant").loc[best]
    control_selected=tables["screening"].set_index("variant").loc["control_no_domain_features"]
    bands=tables["validation_bands"].groupby(["variant","band"],sort=False).agg(n_rows=("rows","first"),batteries=("batteries","first"),mae_mean_pp=("mae_pp","mean"),mae_sd_pp=("mae_pp","std")).reset_index()
    bands.to_csv(output/"validation_band_summary.csv",index=False,lineterminator="\n")
    base=load_training_data(study["reference_config"]["recipe"],root)
    raw=load_raw_observations(base,root)
    full=load_engineered_data("fe_combined",root)
    prep=full["bundle"]["preprocessor"]
    extras=prep._arithmetic(raw["train"])
    diagnostics=prep.describe_train(raw["train"])
    diagnostics["raw_min_train"]=[extras[c].min() for c in diagnostics.feature]
    diagnostics["raw_max_train"]=[extras[c].max() for c in diagnostics.feature]
    y=pd.Series(base["train"]["y_original"],index=extras.index)
    diagnostics["spearman_target_train"]=[extras[c].corr(y,method="spearman") for c in diagnostics.feature]
    diagnostics.to_csv(output/"train_feature_diagnostics.csv",index=False,lineterminator="\n")
    figures=[]
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    def save(fig,name):
        fig.savefig(output/name,dpi=140,bbox_inches="tight",facecolor="white")
        plt.close(fig);figures.append(name)
    screen=tables["screening"]
    fig,ax=plt.subplots(figsize=(10,4.5))
    ax.barh([NAMES[n] for n in screen.variant],screen.validation_mae_pp,color=["#aebcc3" if n!=best else "#247e88" for n in screen.variant])
    for i,r in enumerate(screen.itertuples()):ax.text(r.validation_mae_pp+.01,i,f"{r.validation_mae_pp:.4f} · p={r.input_neurons}",va="center",fontsize=9)
    ax.set(xlabel="Validation MAE (poin SOH), seed 42",title="Screening hanya mengubah blok fitur; model/prosedur tetap",xlim=(0,screen.validation_mae_pp.max()*1.24))
    fig.tight_layout();save(fig,"01_feature_screening.png")

    confirmations=tables["seed_confirmation"]
    gains=tables["paired_seed_improvements"]
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for seed,group in confirmations.groupby("seed"):
        tab=group.set_index("variant").reindex(["control_no_domain_features",best])
        axes[0].plot([0,1],tab.validation_mae_pp,"o-",label=f"seed {seed}")
    axes[0].set(xticks=[0,1],xticklabels=["Kontrol","FE terbaik"],ylabel="Validation MAE (poin SOH)",title="Perbandingan berpasangan pada tiga seed");axes[0].legend(fontsize=8)
    axes[1].bar(gains.seed.astype(str),gains.mae_improvement_pp,color="#247e88")
    axes[1].axhline(study["mean_validation_mae_improvement_pp"],color="#a4548e",label="Rerata gain tiga seed")
    axes[1].axhline(study["decision_threshold_pp"],color="gray",ls="--",label="Ambang promosi untuk rerata")
    axes[1].axhline(0,color="black",lw=.6)
    axes[1].set(xlabel="Seed",ylabel="MAE kontrol − FE (poin SOH)",title="Gain positif = FE lebih kecil; ambang bukan uji statistik")
    axes[1].legend(fontsize=8)
    fig.tight_layout();save(fig,"02_paired_seed_improvements.png")

    order=["<70","70–<80","80–<90","90–<95",">=95"]
    fig,ax=plt.subplots(figsize=(10,4.5))
    for offset,variant,color,label in [(-.18,"control_no_domain_features","#aebcc3","Kontrol"),(.18,best,"#247e88","FE terbaik")]:
        tab=bands[bands.variant.eq(variant)].set_index("band").reindex(order)
        ax.bar(np.arange(5)+offset,tab.mae_mean_pp,width=.35,yerr=tab.mae_sd_pp,capsize=4,color=color,label=label)
    counts=bands[bands.variant.eq(best)].set_index("band").reindex(order).n_rows
    ax.set(xticks=range(5),xticklabels=[f"{b}\nn={int(n)}" for b,n in zip(order,counts)],ylabel="MAE (poin SOH), mean ± SD tiga seed",xlabel="Rentang SOH aktual validation (%)",title="Periksa ekor SOH rendah, bukan hanya skor keseluruhan")
    ax.legend();fig.tight_layout();save(fig,"03_validation_tail_errors.png")

    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for run_id,label,color in [(control_selected.run_id,"Kontrol","#aebcc3"),(selected.run_id,"FE terbaik","#247e88")]:
        history=pd.read_csv(root/"results/experiments/runs"/run_id/"history.csv")
        history=history.loc[history.epoch.ge(5)]
        axes[0].plot(history.epoch,history.validation_mae_pp,label=label,color=color)
    axes[0].set(xlabel="Epoch (zoom mulai 5)",ylabel="Validation MAE (poin SOH)",title="Kurva kandidat seed 42; early stopping tetap sama");axes[0].legend()
    axes[1].bar(["Kontrol","FE terbaik"],[control.parameters,fe.parameters],color=["#aebcc3","#247e88"])
    axes[1].set(ylabel="Jumlah parameter MLP",title="Hidden tetap [32], tetapi dimensi input bertambah")
    fig.tight_layout();save(fig,"04_learning_and_complexity.png")
    formulas=pd.DataFrame([
        {"block":"usage_rates","feature":"fe_cycles_per_day","formula":"cycle_count / battery_age_days","unit":"siklus/hari; rasio snapshot"},
        {"block":"usage_rates","feature":"fe_energy_per_cycle_kwh","formula":"total_energy_throughput_kwh / cycle_count","unit":"kWh/siklus"},
        {"block":"context","feature":"fe_charge_ambient_delta_c","formula":"peak_temp_during_fast_charge_c - avg_ambient_temp_c","unit":"°C"},
        {"block":"context","feature":"fe_fastcharge_dod_interaction","formula":"fast_charge_ratio * avg_depth_of_discharge_pct / 100","unit":"interaksi rasio"},
        {"block":"chemistry","feature":"fe_chem_<kimia>_log_cycles","formula":"I(chemistry=kimia) * log1p(cycle_count)","unit":"interaksi; 4 kolom dari kimia train"},
        {"block":"chemistry","feature":"fe_chem_<kimia>_log_age_years","formula":"I(chemistry=kimia) * log1p(battery_age_days / 365.25)","unit":"interaksi; 4 kolom dari kimia train"},
    ])
    formulas.to_csv(output/"feature_definitions.csv",index=False,lineterminator="\n")
    recommendation="pertahankan kontrol tanpa fitur domain sebagai kandidat utama" if not study["promoted_by_prespecified_gate"] else "promosikan FE terbaik sebagai kandidat sementara berdasarkan validation"
    percentage=study["mean_validation_mae_improvement_pp"]/control.mae_mean_pp*100
    complexity=(fe.parameters/control.parameters-1)*100
    source=f"""# Ablation Feature Engineering untuk MLP

## Hasil dan rekomendasi

**Rekomendasi: {recommendation}.** Fitur baru benar-benar diuji; peningkatan kecil tidak otomatis dijadikan alasan menambah kompleksitas.

- Kontrol adalah kandidat notebook 03, **bukan** baseline awal [64,32] ReLU.
- Mean validation MAE tiga seed: kontrol **{control.mae_mean_pp:.5f} ± {control.mae_sd_pp:.5f}**, FE terbaik **{fe.mae_mean_pp:.5f} ± {fe.mae_sd_pp:.5f}** poin SOH.
- Gain berpasangan rerata **{study['mean_validation_mae_improvement_pp']:.5f} poin** ({percentage:.2f}% relatif terhadap MAE kontrol), menang pada **{study['seed_wins']}/3** seed.
- Input kontrol 67, FE terbaik {int(selected.input_neurons)}. Parameter {int(control.parameters):,} → {int(fe.parameters):,} (**+{complexity:.1f}%**), hidden tetap [32].
- Ambang promosi yang ditetapkan sebelum training: gain rerata minimal **{study['decision_threshold_pp']:.2f} poin** dan menang minimal dua seed. Lolos: **{study['promoted_by_prespecified_gate']}**. Ini heuristik praktis, bukan uji signifikansi atau ambang fisika baterai.
- Test tidak ditransform, tidak dimuat sebagai array, dan tidak dievaluasi. Data sintetis tetap membatasi generalisasi.

## 1. Pertanyaan dan protokol

Apakah fitur turunan menurunkan error MLP jika hidden, aktivasi, output, loss, learning rate, batch, seed, dan aturan checkpoint/early stopping sama? Kontrol dibekukan dari run `{study['reference_candidate_run_id']}`: log_sensor_caps, hidden [32], sigmoid, output linear, MAE, Adam lr=0.001, batch 256, weight_decay=1e-4, dropout 0, max 200 epoch, patience 15.

Screening empat varian dilakukan pada seed 42, lalu hanya FE dengan MAE screening terkecil dikonfirmasi pada 42/123/2026. Tiga run kontrol yang sudah ada dipakai ulang dengan fingerprint yang sama. Ada **{study['new_training_runs']} run training baru** dalam versi final ini; riwayat study notebook 03 tetap utuh.

Validation dipakai untuk screening dan konfirmasi, sehingga hasil ini belum evaluasi independen akhir. Tidak menjalankan pencarian tanpa batas sampai hasil terlihat bagus dan tidak memilih seed terbaik.

## 2. Definisi fitur

{markdown_table(formulas)}

Chemistry dipelajari dari kategori nonmissing pada encoder train yang telah dibekukan. Untuk kimia diketahui, interaksi kategori yang tidak cocok bernilai nol; untuk kimia missing/baru atau exposure missing, fitur interaksi menjadi missing. Tidak menyamaratakan unknown menjadi salah satu kimia yang dikenal.

Rasio hanya dihitung jika denominator >0 dan kedua operand terisi setelah pemeriksaan domain. Tidak memakai epsilon/floor yang mengubah arti pembagian. Nilai yang tidak tersedia dipertahankan sebagai missing sebelum imputasi ekstra. Tidak memakai target, capacity_fade_pct, RUL, label risiko/degradasi, ID, data masa depan, atau group fill.

Rasio kumulatif adalah ringkasan snapshot, bukan estimasi laju degradasi atau bukti mekanisme fisik. Counter siklus/energi yang tidak selalu monoton tetap menjadi keterbatasan sumber; feature engineering tidak memperbaikinya secara otomatis.

## 3. Preprocessing ekstra dan fairness

Seluruh **67 output base pipeline tetap bit-for-bit sama** sebagai prefix input. Fitur baru diappend bersama indikator missing masing-masing. Median, batas cap 0.5%/99.5% untuk dua rasio dan delta suhu, serta scaler ekstra di-fit hanya train. Dua rasio memakai signed log1p setelah cap; interaksi kimia sudah dibentuk pada log exposure dan tidak dilog dua kali. Fast charge–DOD tidak dicap karena sudah berupa produk rasio yang valid.

Target scaler dipakai ulang tanpa refit. Tidak menghapus observasi atau SOH rendah. Tidak ada array test baru. CSV sumber dibaca secara chunk dengan kolom input+ID saja; baris test dibuang saat loading sebelum aritmetika/statistik. Label target tidak diambil dari pembacaan CSV ini.

Konfigurasi MLP sama kecuali jumlah input dan parameter yang mengikuti dimensi. Karena parameter bertambah, ablation ini mengukur paket representasi fitur+kapasitas input, bukan mengisolasi efek kausal fitur pada jumlah parameter identik. Seluruh jumlah parameter dicatat.

## 4. Screening

{markdown_table(tables['screening'].round(5))}

![Screening](01_feature_screening.png)

Pemenang screening adalah `{best}`. Sebuah blok dapat memperburuk hasil; bukan semua fitur turunan harus dipakai. Tidak menghapus hasil varian yang kalah. Screening satu seed dapat sensitif terhadap optimasi, sehingga disertai konfirmasi tiga seed. Hanya pemenang screening yang dikonfirmasi; selisih sangat kecil dengan varian rasio tidak membuktikan peringkat tersebut stabil pada seed lain.

## 5. Konfirmasi berpasangan

{markdown_table(tables['seed_summary'].round(5))}

{markdown_table(tables['paired_seed_improvements'].round(5))}

![Seed](02_paired_seed_improvements.png)

SD adalah sample SD tiga seed pada split tetap, bukan CI populasi atau bukti signifikansi. Ambang 0.01 poin berlaku pada **rerata gain**, bukan syarat tiap seed. Menang tiga seed dapat menunjukkan arah konsisten pada seed yang diuji, tetapi n=3 dan besarnya gain tetap perlu dipertimbangkan.

## 6. Ekor SOH rendah dan kompleksitas

{markdown_table(bands.round(5))}

![Rentang SOH](03_validation_tail_errors.png)

Baca mean error tiap band bersama jumlah observasi/baterai dan variasi seed. Jumlah baterai antarband dapat tumpang tindih. Tidak mempromosikan suatu fitur hanya karena satu kelompok validation kecil tampak lebih bagus.

![Learning dan parameter](04_learning_and_complexity.png)

Learning curves adalah validation seed 42, zoom epoch ≥5. Perbedaan epoch aktual mengikuti aturan early stopping yang sama. Penambahan fitur meningkatkan jumlah parameter; menambah fitur bukan jaminan menghilangkan overfitting/underfitting.

## 7. Diagnostik fitur train

`train_feature_diagnostics.csv` berisi missing, rentang raw, median, cap, scaler, dan korelasi target **train**. Kolom parameter per varian dan nama fitur output juga disimpan. Korelasi hanya membantu interpretasi; tidak dipakai untuk memilih subset tambahan setelah melihat hasil validation. Pada train, dua rasio kosong sekitar 10.38% dan 11.07%, sebab jika salah satu operand kosong hasil juga kosong. Fitur turunan tidak menambah pengukuran independen dan dapat meningkatkan redundansi/kompleksitas, sehingga peningkatan kecil masuk akal tanpa membuktikan penyebabnya. Fitur rasio yang kuat berkaitan dengan fitur asal dapat redundan, dan interaksi kimia bukan informasi fisik baru.

## 8. Artefak dan reproduksi

- Notebook `03b_feature_engineering_ablation.ipynb`, terpisah agar study 03 tidak ditulis ulang.
- Config: `configs/feature_engineering.json`.
- Kode: `src/feature_engineering.py`, memakai training loop MLP yang sama tanpa modifikasi.
- Array train/validation FE: `data/processed/<variant>/`, lokal dan diabaikan Git.
- Bundle inference: `models/feature_engineering/<variant>.joblib`; `bundle['preprocessor'].transform(new_rows[bundle['required_columns']])` menghitung base+fitur ekstra konsisten. Load hanya artefak tepercaya.
- Checkpoint MLP: `models/mlp/<run_id>/`, lokal. Config/history/summary/status ada di `results/experiments/runs/<run_id>/`.
- Ringkasan study: `results/feature_engineering/study.json`. Kontrol tetap tercatat di `results/experiments/study.json`; tidak diganti diam-diam.

Reproduksi: `python -m src.feature_engineering`, `python -m src.feature_reporting`, kemudian jalankan notebook 03b. Run dengan fingerprint source/data/config sama dipakai ulang. Untuk cross-validation, fit base serta ekstra pada tiap fold-train; jangan memakai fitted global-train untuk fold-validation.

## 9. Batas kesimpulan dan langkah berikutnya

Hasil saat ini mendukung **{recommendation}** menurut aturan praktis yang telah ditetapkan, sambil menyimpan FE sebagai hasil ablation yang sah. Tidak ada kebutuhan memaksakan FE agar selalu menang. Keputusan ini belum klaim statistik populasi, skor test, atau validitas pada baterai nyata.

Langkah berikutnya adalah meninjau rekomendasi, lalu mengunci konfigurasi/seed untuk evaluasi test di notebook 04. Jangan memakai test untuk memilih fitur baru atau menyesuaikan ambang keputusan.
"""
    (output/"FEATURE_ENGINEERING_REPORT.md").write_text(source,encoding="utf-8",newline="\n")
    return {"figures":figures,"report":output/"FEATURE_ENGINEERING_REPORT.md"}


if __name__=="__main__":print(build_report()["report"])
