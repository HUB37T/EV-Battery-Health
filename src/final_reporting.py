"""Report frozen final test results; no selection, fitting, or training."""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_audit import ROOT,markdown_table

COLORS={"baseline":"#a5b4bd","candidate":"#247e88"}
NAMES={"baseline":"Baseline MLP","candidate":"Kandidat utama"}


def read_results(root=ROOT):
    output=root/"results/final_evaluation"
    result=json.loads((output/"evaluation.json").read_text(encoding="utf-8"))
    lock=json.loads((output/"evaluation_lock.json").read_text(encoding="utf-8"))
    if result["status"]!="complete" or result["model_selection_from_test"] or result["training_or_fit_performed"]:
        raise ValueError("Hasil final belum lengkap atau melanggar protokol.")
    if result["lock_fingerprint"]!=lock["fingerprint"]:raise ValueError("Lock dan hasil final berbeda.")
    tables={name:pd.read_csv(output/f"{name}.csv") for name in ["metrics_by_seed","metrics_summary","test_band_summary","test_battery_summary","validation_test_comparison","constant_references"]}
    return output,result,lock,tables


def build_report(root=ROOT):
    output,result,lock,tables=read_results(root)
    summary=tables["metrics_summary"].set_index("group")
    candidate=summary.loc["candidate"];baseline=summary.loc["baseline"]
    metrics=tables["metrics_by_seed"]
    bands=tables["test_band_summary"]
    batteries=tables["test_battery_summary"]
    selected_seed=lock["payload"]["plan"]["seeds"][0]  # predeclared first seed, not best test seed
    predictions={g:pd.read_csv(output/"predictions"/f"{g}_seed_{selected_seed}.csv") for g in NAMES}
    paired=batteries.pivot(index="battery_id",columns="group",values="mae_mean_pp")
    paired["baseline_minus_candidate_mae_pp"]=paired.baseline-paired.candidate
    paired.reset_index().to_csv(output/"paired_battery_mae.csv",index=False,lineterminator="\n")
    worst=batteries.loc[batteries.group.eq("candidate")].sort_values("mae_mean_pp",ascending=False).head(10)
    worst.to_csv(output/"largest_candidate_battery_errors.csv",index=False,lineterminator="\n")
    figures=[]
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    def save(fig,name):
        fig.savefig(output/name,dpi=140,bbox_inches="tight",facecolor="white")
        plt.close(fig);figures.append(name)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,(mean,sd,title) in zip(axes,[("test_mae_mean_pp","test_mae_sd_pp","MAE (poin SOH)"),("test_rmse_mean_pp","test_rmse_sd_pp","RMSE (poin SOH)"),("test_r2_mean","test_r2_sd","R² (bukan accuracy)")]):
        values=summary.reindex(["baseline","candidate"])
        ax.bar([0,1],values[mean],yerr=values[sd],capsize=5,color=list(COLORS.values()))
        for i,value in enumerate(values[mean]):ax.text(i,value+max(values[mean])*.035,f"{value:.4f}",ha="center",fontsize=10)
        ax.set(xticks=[0,1],xticklabels=["Baseline","Kandidat"],title=title,ylim=(0,max(values[mean])*1.17))
    fig.suptitle("Hasil test: mean ± sample SD tiga seed pada 390 baterai yang sama")
    fig.tight_layout(rect=(0,0,1,.94));save(fig,"01_test_metrics.png")

    fig,ax=plt.subplots(figsize=(9,4.5))
    x=np.arange(3)
    for offset,group in [(-.18,"baseline"),(.18,"candidate")]:
        rows=metrics.loc[metrics.group.eq(group)]
        fields=["train_mae_pp","validation_mae_pp","test_mae_pp"]
        means=[rows[f].mean() for f in fields];deviations=[rows[f].std(ddof=1) for f in fields]
        ax.bar(x+offset,means,width=.35,yerr=deviations,capsize=4,color=COLORS[group],label=NAMES[group])
    ax.set(xticks=x,xticklabels=["Train","Validation","Test"],ylabel="MAE (poin SOH), mean ± SD tiga seed",title="Perbandingan generalisasi tanpa mengubah bobot dari hasil test")
    ax.legend();fig.tight_layout();save(fig,"02_train_validation_test.png")

    lo=np.floor(min(min(p.actual_soh.min(),p.predicted_soh.min()) for p in predictions.values())/5)*5
    hi=np.ceil(max(max(p.actual_soh.max(),p.predicted_soh.max()) for p in predictions.values())/5)*5
    fig,axes=plt.subplots(1,2,figsize=(11,5))
    for ax,(group,pred) in zip(axes,predictions.items()):
        ax.scatter(pred.actual_soh,pred.predicted_soh,s=5,alpha=.18,color=COLORS[group],rasterized=True)
        ax.plot([lo,hi],[lo,hi],"--",color="gray")
        ax.set(xlabel="SOH aktual (%)",ylabel="SOH prediksi (%)",title=f"{NAMES[group]} · seed {selected_seed}",xlim=(lo,hi),ylim=(lo,hi))
        ax.set_aspect("equal",adjustable="box")
    fig.suptitle("Prediksi test tanpa clipping; seed pertama dalam rencana, bukan dipilih dari test")
    fig.tight_layout(rect=(0,0,1,.94));save(fig,"03_actual_predictions.png")

    order=lock["payload"]["plan"]["soh_band_labels"]
    fig,axes=plt.subplots(1,2,figsize=(13,4.5))
    x=np.arange(5)
    for offset,group in [(-.18,"baseline"),(.18,"candidate")]:
        sub=bands.loc[bands.group.eq(group)].set_index("soh_band").reindex(order)
        axes[0].bar(x+offset,sub.mae_mean_pp,width=.35,yerr=sub.mae_sd_pp,capsize=4,color=COLORS[group],label=NAMES[group])
        axes[1].plot(x,sub.bias_mean_pp,"o-",color=COLORS[group],label=NAMES[group])
    counts=bands.loc[bands.group.eq("candidate")].set_index("soh_band").reindex(order).rows
    for ax in axes:
        ax.set(xticks=x,xticklabels=[f"{b}\nn={int(n)}" for b,n in zip(order,counts)],xlabel="Rentang SOH aktual test (%)")
        ax.legend(fontsize=8)
    axes[0].set(ylabel="MAE (poin SOH), mean ± SD tiga seed",title="Keunggulan agregat tidak merata di semua rentang")
    axes[1].axhline(0,color="gray",ls="--");axes[1].set(ylabel="Rerata prediksi − aktual (poin SOH)",title="Bias positif berarti SOH terlalu tinggi diprediksi")
    fig.tight_layout();save(fig,"04_test_soh_bands.png")

    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for group in NAMES:
        values=np.sort(batteries.loc[batteries.group.eq(group),"mae_mean_pp"])
        axes[0].plot(values,np.arange(1,len(values)+1)/len(values),color=COLORS[group],label=NAMES[group])
    axes[0].set(xlabel="MAE per baterai, dirata-rata tiga seed (poin SOH)",ylabel="Proporsi kumulatif baterai",title="Setiap baterai mempunyai bobot sama");axes[0].legend(fontsize=8)
    axes[1].hist(paired.baseline_minus_candidate_mae_pp,bins=40,color=COLORS["candidate"],edgecolor="white")
    axes[1].axvline(0,color="gray",ls="--")
    axes[1].set(xlabel="MAE baseline − kandidat per baterai (poin SOH)",ylabel="Baterai",title="Positif: kandidat lebih rendah; ada baterai yang memburuk")
    fig.tight_layout();save(fig,"05_battery_errors.png")

    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for group,pred in predictions.items():
        axes[0].hist(pred.error_pp,bins=60,histtype="step",linewidth=1.5,color=COLORS[group],label=NAMES[group])
    axes[0].axvline(0,color="gray",ls="--");axes[0].set(xlabel="Prediksi − aktual (poin SOH)",ylabel="Observasi test",title=f"Residual contoh seed {selected_seed}");axes[0].legend(fontsize=8)
    view=worst.sort_values("mae_mean_pp")
    axes[1].barh(view.battery_id,view.mae_mean_pp,color=COLORS["candidate"])
    axes[1].set(xlabel="MAE rerata tiga seed (poin SOH)",title="10 baterai test dengan error kandidat terbesar")
    fig.tight_layout();save(fig,"06_residuals_and_largest_errors.png")

    improvement=baseline.test_mae_mean_pp-candidate.test_mae_mean_pp
    relative=improvement/baseline.test_mae_mean_pp*100
    low=bands.loc[bands.soh_band.eq("<70")].set_index("group")
    primary_config=lock["payload"]["plan"]["groups"]["candidate"]
    candidate_record=json.loads((root/"results/experiments/runs"/primary_config["run_ids_by_seed"][str(selected_seed)]/"summary.json").read_text(encoding="utf-8"))
    compact_summary=summary.reset_index()[["group","test_mae_mean_pp","test_mae_sd_pp","test_rmse_mean_pp","test_rmse_sd_pp","test_r2_mean","test_r2_sd","test_battery_mae_mean_pp","parameters"]]
    better=int(paired.baseline_minus_candidate_mae_pp.gt(0).sum())
    source=f"""# Evaluasi Akhir MLP Backpropagation

## Hasil utama

Evaluasi final selesai pada **7.572 observasi dari 390 baterai test** yang tidak muncul dalam train/validation. Enam checkpoint dan preprocessing dikunci sebelum prediksi test dihitung. Tidak ada fit, retraining, pemilihan seed, penambahan FE, clipping prediksi, atau tuning setelah melihat test.

- Kandidat utama yang ditetapkan sebelum test: mean MAE **{candidate.test_mae_mean_pp:.5f} ± {candidate.test_mae_sd_pp:.5f} poin SOH**, RMSE **{candidate.test_rmse_mean_pp:.5f} ± {candidate.test_rmse_sd_pp:.5f}**, R² **{candidate.test_r2_mean:.5f} ± {candidate.test_r2_sd:.5f}**.
- Baseline MLP: mean MAE **{baseline.test_mae_mean_pp:.5f} ± {baseline.test_mae_sd_pp:.5f} poin**. Selisih **{improvement:.5f} poin**, sekitar **{relative:.2f}%** relatif terhadap MAE baseline pada test ini.
- Parameter baseline {int(baseline.parameters):,}, kandidat {int(candidate.parameters):,}. Semua hasil memakai seed 42/123/2026, tidak memilih seed test terbaik dan tidak melakukan ensemble.
- Keterbatasan penting: pada **SOH <70%**, MAE kandidat **{low.loc['candidate','mae_mean_pp']:.4f}** lebih besar daripada baseline **{low.loc['baseline','mae_mean_pp']:.4f}**, dengan {int(low.loc['candidate','rows'])} observasi / {int(low.loc['candidate','batteries'])} baterai. Keunggulan agregat tidak berarti kandidat unggul di semua kondisi.

## 1. Apa yang dikunci sebelum test?

Rencana `configs/final_evaluation.json` menetapkan baseline/kandidat dan tiga seed, best checkpoint berdasarkan validation, rentang SOH, serta kandidat sebagai keluarga model utama. Kandidat memakai log_sensor_caps, input 67 → hidden [32] sigmoid → output linear, loss MAE, Adam lr=0.001, weight_decay=1e-4. Baseline memakai input 67 → [64,32] ReLU → output linear, MSE, Adam dengan kebijakan batch/stopping yang sama. Epoch terbaik berasal dari validation, bukan test.

Feature engineering tetap hasil ablation validation. Rekomendasinya sebelum test mempertahankan kandidat 03 tanpa fitur domain tambahan. FE tidak diam-diam dimasukkan kembali ke pembandingan test.

`evaluation_lock.json` dibuat pada `{lock['locked_at_utc']}` sebelum inference test, fingerprint `{lock['fingerprint']}`. Lock mencatat SHA-256 enam checkpoint, dua fitted pipeline, array test, konfigurasi, manifest, bukti pemilihan validation/FE, dan source evaluator. Semua berkas yang dikunci diperiksa tetap sama sesudah inference.

## 2. Metrik akhir dan arti satuannya

{markdown_table(compact_summary.round(5))}

![Metrik test](01_test_metrics.png)

MAE/RMSE dihitung setelah inverse target scaling. MAE 0.95 berarti rata-rata selisih absolut sekitar 0.95 **poin persentase SOH**, bukan accuracy 99.05% dan bukan jaminan semua prediksi memiliki error sebesar itu. R² juga bukan accuracy klasifikasi.

SD adalah sample SD (ddof=1) tiga seed training pada baterai test yang sama; bukan confidence interval populasi, bukan ketidakpastian semua split, dan bukan tiga dataset test independen. Rerata R²/RMSE adalah rerata metrik per seed, bukan metrik ensemble prediksi.

## 3. Hasil setiap seed

{markdown_table(tables['metrics_by_seed'][['group','seed','run_id','best_epoch_from_validation','test_mae_pp','test_rmse_pp','test_r2','test_battery_mean_mae_pp']].round(5))}

Semua seed tetap ditampilkan, termasuk yang error-nya lebih tinggi. Gambar scatter menggunakan seed {selected_seed}, yaitu seed pertama dalam rencana yang sudah ada, bukan seed yang dipilih setelah melihat ranking test. Perbandingan utama tetap mean/SD ketiga seed.

## 4. Apakah pola validation bertahan?

{markdown_table(tables['validation_test_comparison'].round(5))}

![Train validation test](02_train_validation_test.png)

Test MAE kandidat berada sekitar {candidate.test_mae_mean_pp-candidate.validation_mae_mean_pp:+.5f} poin dari validation. Baseline juga memiliki skor test sedikit berbeda. Perbedaan tersebut dapat berasal dari komposisi baterai/sampel pada split; bukan alasan mengubah scaler atau bobot agar hasil cocok. Skor test lebih rendah daripada validation tidak membuktikan kebocoran maupun menjamin generalisasi pada distribusi lain.

Kandidat dipilih sebelum test. Hasil ini mendukung kegunaan kandidat pada test sintetis yang ditahan, tetapi tidak membuktikan bebas overfitting untuk semua data. Kurva train–validation dan proses seleksi tetap dibahas pada notebook 03; test tidak menjadi dataset tuning baru.

## 5. Aktual vs prediksi dan residual

![Prediksi](03_actual_predictions.png)

![Residual dan baterai sulit](06_residuals_and_largest_errors.png)

Prediksi tidak dipotong ke 0..100. Rerata proporsi prediksi kandidat di luar rentang itu adalah **{candidate.test_outside_0_100_mean_pct:.4f}%**; nilai tersebut dilaporkan sebagai keterbatasan output linear. Residual positif berarti model memprediksi SOH lebih tinggi daripada aktual. Rerata bias kandidat pada seluruh test adalah {candidate.test_bias_mean_pp:+.4f} poin, yang tidak boleh menutupi bias lebih besar pada subkelompok.

## 6. Rentang SOH rendah masih menjadi kelemahan

{markdown_table(bands.round(5))}

![Rentang SOH](04_test_soh_bands.png)

Pada SOH <70%, kandidat mempunyai MAE **{low.loc['candidate','mae_mean_pp']:.4f} ± {low.loc['candidate','mae_sd_pp']:.4f}**, baseline **{low.loc['baseline','mae_mean_pp']:.4f} ± {low.loc['baseline','mae_sd_pp']:.4f}**. Bias kandidat di rentang ini adalah **{low.loc['candidate','bias_mean_pp']:+.4f} poin**: kesehatan baterai sintetis yang lebih buruk cenderung diperkirakan terlalu tinggi.

Subkelompok ini hanya {int(low.loc['candidate','rows'])} observasi dari {int(low.loc['candidate','batteries'])} baterai; tidak menyimpulkan superioritas statistik dari mean/SD tiga seed. Namun temuan tersebut wajib dilaporkan, karena target train didominasi SOH tinggi menurut EDA. Tidak mengubah loss, fitur, weighting, atau threshold dari temuan test ini. Jika dikembangkan lagi, diperlukan validasi/evaluasi independen baru.

## 7. Evaluasi per baterai

![Per baterai](05_battery_errors.png)

MAE utama memberi bobot sama per observasi. `test_battery_mean_mae_pp` merata-ratakan MAE setiap baterai agar baterai dengan banyak observasi tidak mendominasi. Kandidat: **{candidate.test_battery_mae_mean_pp:.5f} ± {candidate.test_battery_mae_sd_pp:.5f} poin**, baseline **{baseline.test_battery_mae_mean_pp:.5f} ± {baseline.test_battery_mae_sd_pp:.5f}**.

Setelah MAE setiap baterai dirata-rata tiga seed, kandidat memiliki error lebih rendah pada {better} dari {len(paired)} baterai. Ini analisis deskriptif, bukan 390 eksperimen training independen. Sepuluh baterai dengan error kandidat terbesar disajikan berikut; daftar ini dipilih eksplisit untuk diagnosis akhir, bukan untuk menyetel model terhadap test.

{markdown_table(worst.round(5))}

## 8. Kontrol konstanta dari train

{markdown_table(tables['constant_references'].round(5))}

Median/rerata dipelajari hanya dari y_train sebelum test dan dipakai sebagai pemeriksaan kewajaran. Ini bukan algoritma ML tambahan; model yang dilatih tetap MLP/backpropagation. Hasil akhir menunjukkan manfaat di atas menebak pusat distribusi target saja, tanpa menyamakan perbaikan tersebut dengan validasi fisik baterai.

## 9. Kesimpulan yang dapat dipertanggungjawabkan

Pada protokol group split dataset sintetis ini, kandidat MLP kecil yang dipilih dari validation menghasilkan MAE test agregat lebih rendah daripada baseline pada tiga seed yang sudah dikunci. Model memiliki lebih sedikit parameter, dan arah perbandingan validation bertahan pada test.

Klaim ini dibatasi pada estimasi SOH saat observasi untuk baterai baru dalam simulator yang sama. Tidak menguji forecasting, sisa umur, data kendaraan nyata, atau distribution shift. Kelemahan pada SOH rendah dan prediksi di luar rentang fisik tetap terbuka. Tidak menyatakan “akurasi 95%”, bebas overfitting secara universal, atau siap digunakan pada baterai nyata.

Keluarga kandidat tetap model utama karena sudah dipilih **sebelum** test; test hanya evaluasi akhir. Feature engineering tetap dilaporkan sebagai ablation validation yang memberi gain kecil dengan kompleksitas lebih besar.

## 10. Artefak dan reproduksi

- `notebooks/04_final_evaluation.ipynb`: protokol, metrik, visual, analisis error, dan pemeriksaan.
- `configs/final_evaluation.json`: model/seed yang dikunci.
- `evaluation_lock.json`: identitas berkas dan waktu lock sebelum test inference.
- `metrics_by_seed.csv`, `metrics_summary.csv`: seluruh hasil seed dan ringkasan.
- `test_band_summary.csv`, `test_battery_summary.csv`: hasil per rentang dan baterai; tabel seed rinci juga tersedia.
- `predictions/`: prediksi per observasi test tersimpan lokal, diabaikan Git.
- Bobot dan fitted preprocessing tetap di folder models lokal.

Perintah: `python -m src.final_evaluation`, `python -m src.final_reporting`, kemudian jalankan notebook 04. Evaluator memakai ulang hasil yang lengkap dengan lock yang sama; tidak melatih ulang atau mengganti checkpoint. Pemeriksaan deterministik ulang terhadap hasil yang sama bukan evaluasi independen baru.

Untuk clone/lingkungan baru, laporan dapat dibaca tanpa bobot. Jika perlu mereproduksi setelah checkpoint/preprocessing dibuat ulang dan hash biner berbeda, gunakan `python -m src.final_evaluation --reproduce`; output berada di `data/processed/final_evaluation_reproduction`, terpisah dari hasil rilis. Perbandingan protokol tetap sama, bukan izin tuning dari test. Jangan menghapus lock rilis untuk mengganti hasil akhir.

Langkah proyek berikutnya adalah menyusun laporan/presentasi dari hasil yang sudah ada, dengan angka, satuan, prosedur seleksi, dan keterbatasan yang konsisten.
"""
    (output/"FINAL_REPORT.md").write_text(source,encoding="utf-8",newline="\n")
    return {"figures":figures,"report":output/"FINAL_REPORT.md"}


if __name__=="__main__":print(build_report()["report"])
