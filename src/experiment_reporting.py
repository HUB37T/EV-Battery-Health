"""Read completed train/validation experiments; generate figures and discussion."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

from src.data_audit import ROOT, markdown_table


def read_results(root=ROOT):
    output=root/"results/experiments"
    study=json.loads((output/"study.json").read_text(encoding="utf-8"))
    if study["status"]!="complete" or study["test_loaded"]:
        raise ValueError("Study belum lengkap atau batas test dilanggar.")
    runs=pd.read_csv(output/"run_summary.csv")
    tables={name:pd.read_csv(output/f"{name}.csv") for name in ["stage_comparisons","epoch_budget_comparison","seed_confirmation","seed_summary","constant_references","candidate_validation_bands"]}
    summaries={}
    histories={}
    for run_id in runs.run_id:
        summaries[run_id]=json.loads((output/"runs"/run_id/"summary.json").read_text(encoding="utf-8"))
        histories[run_id]=pd.read_csv(output/"runs"/run_id/"history.csv")
    return output,study,runs,tables,summaries,histories


def build_figures(root=ROOT):
    output,study,runs,tables,summaries,histories=read_results(root)
    figures=[]
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    def save(fig,name):
        fig.savefig(output/name,dpi=140,bbox_inches="tight",facecolor="white")
        plt.close(fig);figures.append(name)
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for row,(label,key) in enumerate([("Baseline","baseline_run_id"),("Kandidat seed 42","candidate_run_id")]):
        run=summaries[study[key]];history=histories[study[key]]
        for split,color in [("train","#247e88"),("validation","#c47a21")]:
            axes[row,0].plot(history.epoch,history[f"{split}_objective_scaled"],color=color,label=split)
            zoom=history.loc[history.epoch.ge(5)]
            axes[row,1].plot(zoom.epoch,zoom[f"{split}_mae_pp"],color=color,label=split)
        for ax in axes[row]:ax.axvline(run["best_epoch"],ls="--",color="#a4548e",lw=1,label=f"best epoch {run['best_epoch']}");ax.set_xlabel("Epoch");ax.legend(fontsize=8)
        axes[row,0].set(title=f"{label}: {run['config']['loss']} pada y terstandar",ylabel="Objective loss")
        axes[row,1].set(title=f"{label}: MAE SOH asli, zoom epoch ≥5",ylabel="MAE (poin persentase)")
    fig.suptitle("Learning curves end-of-epoch, evaluasi tanpa dropout; checkpoint berdasarkan validation MAE")
    fig.tight_layout(rect=(0,0,1,.96));save(fig,"01_learning_curves.png")

    stages=tables["stage_comparisons"]
    fig,axes=plt.subplots(3,2,figsize=(12,12))
    for ax,(stage,group) in zip(axes.flat,stages.groupby("stage",sort=False)):
        if stage=="preprocessing":labels=group.recipe
        elif stage=="hidden_layers":labels=group.hidden_layers
        elif stage=="activation":labels=group.activation
        elif stage=="loss":labels=group.loss
        else:labels=[f"weight_decay={r.weight_decay:g}, dropout={r.dropout:g}" for r in group.itertuples()]
        ax.barh(list(labels),group.validation_mae_pp,color=["#247e88" if bool(v) else "#aebcc3" for v in group.selected])
        for i,value in enumerate(group.validation_mae_pp):ax.text(value+.01,i,f"{value:.3f}",va="center",fontsize=9)
        ax.set(title=stage,xlabel="Validation MAE (poin persentase)",xlim=(0,group.validation_mae_pp.max()*1.25))
    axes.flat[-1].axis("off")
    axes.flat[-1].text(0,.8,"Hijau: konfigurasi terpilih pada tahap itu\n\nSetiap tahap meneruskan pemenang sebelumnya.\nBukan full factorial atau perbandingan independen.\n\nRun identik dipakai ulang.\nSemua pemilihan memakai validation, tanpa test.",va="top",fontsize=11)
    fig.suptitle("Eksperimen bertahap: perubahan faktor yang dapat ditelusuri")
    fig.tight_layout(rect=(0,0,1,.97));save(fig,"02_stage_comparisons.png")

    candidate=summaries[study["candidate_run_id"]]
    config=candidate["config"]
    blocks=[f"Input\n{candidate['input_features']} fitur"]+[f"Hidden {i+1}\n{units} · {config['activation']}" for i,units in enumerate(config["hidden_layers"])]+["Output\n1 · linear"]
    fig,ax=plt.subplots(figsize=(12,3.7));ax.set_xlim(0,len(blocks)*2);ax.set_ylim(-1,2);ax.axis("off")
    for i,label in enumerate(blocks):
        x=i*2+.15
        ax.add_patch(FancyBboxPatch((x,.6),1.65,.9,boxstyle="round,pad=.06",facecolor="#e1f0ed",edgecolor="#247e88"))
        ax.text(x+.825,1.05,label,ha="center",va="center",fontsize=11)
        if i:ax.annotate("",xy=(x-.08,1.05),xytext=(x-.3,1.05),arrowprops={"arrowstyle":"->","color":"#247e88"})
    ax.annotate("",xy=(.3,.1),xytext=(len(blocks)*2-.3,.1),arrowprops={"arrowstyle":"->","color":"#a4548e","lw":2})
    ax.text(len(blocks),-.23,"Backpropagation: gradien loss ke seluruh bobot/bias; Adam memperbarui parameter",ha="center",fontsize=10)
    ax.text(len(blocks),1.8,f"MLP kandidat: {candidate['parameters']:,} parameter · loss {config['loss']} · Adam lr={config['learning_rate']}",ha="center",fontsize=13)
    save(fig,"03_candidate_architecture.png")

    fixed=tables["epoch_budget_comparison"]
    history=histories[study["fixed_epoch_run_id"]]
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for split,color in [("train","#247e88"),("validation","#c47a21")]:
        axes[0].plot(history.epoch,history[f"{split}_mae_pp"],color=color,label=split)
        axes[0].scatter(fixed.epoch,fixed[f"{split}_mae"],color=color,zorder=3)
    axes[0].set(title="Satu trajectory tanpa early stopping",xlabel="Epoch",ylabel="MAE (poin persentase)");axes[0].legend()
    x=np.arange(len(fixed))
    axes[1].bar(x-.18,fixed.train_mae,width=.35,color="#247e88",label="train endpoint")
    axes[1].bar(x+.18,fixed.validation_mae,width=.35,color="#c47a21",label="validation endpoint")
    axes[1].set(xticks=x,xticklabels=fixed.epoch.astype(str),xlabel="Epoch tetap",ylabel="MAE (poin persentase)",title="Bobot pada akhir budget, bukan best checkpoint")
    axes[1].legend(fontsize=8)
    fig.tight_layout();save(fig,"04_epoch_budget.png")

    seeds=tables["seed_confirmation"];seed_summary=tables["seed_summary"].set_index("group")
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for seed in sorted(seeds.seed.unique()):
        pair=seeds[seeds.seed.eq(seed)].set_index("group").reindex(["baseline","candidate"])
        axes[0].plot([0,1],pair.validation_mae_pp,"o-",label=f"seed {seed}")
    axes[0].set(xticks=[0,1],xticklabels=["Baseline","Kandidat"],ylabel="Validation MAE (poin persentase)",title="Semua seed dilaporkan, tidak memilih seed terbaik");axes[0].legend(fontsize=8)
    selected=seed_summary.reindex(["baseline","candidate"])
    axes[1].bar([0,1],selected.validation_mae_mean,yerr=selected.validation_mae_std,capsize=6,color=["#aebcc3","#247e88"])
    axes[1].set(xticks=[0,1],xticklabels=["Baseline","Kandidat"],ylabel="Validation MAE (poin persentase)",title="Mean ± sample SD, tiga seed; bukan CI populasi")
    fig.tight_layout();save(fig,"05_seed_confirmation.png")

    predictions=pd.read_csv(output/"runs"/study["candidate_run_id"]/"validation_predictions.csv")
    error=predictions.predicted_soh-predictions.actual_soh
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    axes[0].scatter(predictions.actual_soh,predictions.predicted_soh,s=5,alpha=.18,color="#247e88",rasterized=True)
    axes[0].plot([50,105],[50,105],ls="--",color="gray");axes[0].set(xlabel="SOH aktual (%)",ylabel="SOH prediksi (%)",title="Validation, kandidat seed 42")
    axes[1].hist(error,bins=50,color="#247e88",edgecolor="white");axes[1].axvline(0,ls="--",color="gray");axes[1].set(xlabel="Prediksi − aktual (poin persentase)",ylabel="Observasi",title="Residual tanpa clipping prediksi")
    bands=tables["candidate_validation_bands"]
    axes[2].bar(bands.soh_band,bands.mae_pp,color="#c47a21")
    for i,row in enumerate(bands.itertuples()):axes[2].text(i,row.mae_pp+.03,f"n={row.rows}",ha="center",fontsize=8)
    axes[2].set(xlabel="Rentang SOH aktual (%)",ylabel="MAE (poin persentase)",title="Error per rentang; dukungan tidak sama");axes[2].tick_params(axis="x",rotation=25)
    fig.tight_layout();save(fig,"06_validation_errors.png")
    return figures


def write_report(root=ROOT):
    output,study,runs,tables,summaries,histories=read_results(root)
    baseline=summaries[study["baseline_run_id"]];candidate=summaries[study["candidate_run_id"]]
    bcfg=baseline["config"];cfg=candidate["config"]
    confirmation=tables["seed_summary"].set_index("group")
    constants=tables["constant_references"]
    epochs=tables["epoch_budget_comparison"]
    fixed_summary=summaries[study["fixed_epoch_run_id"]]
    comparison=pd.DataFrame([{"model":label,"recipe":r["config"]["recipe"],"hidden":str(r["config"]["hidden_layers"]),"activation":r["config"]["activation"],"loss":r["config"]["loss"],"input":r["input_features"],"parameters":r["parameters"],"best_epoch":r["best_epoch"],"epochs_actual":r["epochs_actual"],"train_MAE_pp":r["train"]["mae"],"validation_MAE_pp":r["validation"]["mae"],"validation_RMSE_pp":r["validation"]["rmse"],"validation_R2":r["validation"]["r2"]} for label,r in [("Baseline seed 42",baseline),("Kandidat seed 42",candidate)]])
    source=f"""# Eksperimen MLP dengan Backpropagation

## Hasil utama dan status

- **{study['unique_training_runs']} run unik** selesai, termasuk konfirmasi tiga seed. Run identik digunakan ulang, bukan dihitung sebagai eksperimen baru.
- Baseline: validation MAE **{baseline['validation']['mae']:.4f}** poin persentase SOH, train **{baseline['train']['mae']:.4f}**; best epoch {baseline['best_epoch']} dari {baseline['epochs_actual']} epoch aktual.
- Kandidat seed 42: validation MAE **{candidate['validation']['mae']:.4f}**, RMSE **{candidate['validation']['rmse']:.4f}**, R² **{candidate['validation']['r2']:.4f}**. Ini hasil validation yang dipakai berulang selama pencarian, bukan performa test independen.
- Konfirmasi baseline: mean MAE **{confirmation.loc['baseline','validation_mae_mean']:.4f} ± {confirmation.loc['baseline','validation_mae_std']:.4f}**; kandidat **{confirmation.loc['candidate','validation_mae_mean']:.4f} ± {confirmation.loc['candidate','validation_mae_std']:.4f}**, tiga seed pada split tetap.
- Semua model prediksi adalah **MLP feedforward dengan backpropagation**. Test tidak dimuat/dievaluasi. Belum ada evaluasi akhir notebook 04 atau klaim generalisasi pada baterai nyata.

## 1. Data dan batas evaluasi

Train 34.929 baris, validation 7.499 baris, grup baterai tidak tumpang tindih. Input berasal dari tiga fitted preprocessing recipe, semuanya fit train-only. Target dilatih pada skala terstandar dan metrik dikembalikan ke SOH asli melalui target scaler yang cocok.

Runner hanya meminta split `train` dan `validation`; ada pengujian yang menolak jika loader meminta `test`. Tidak menjalankan ulang preprocessing, membentuk fitur domain baru, mengubah split, atau memakai label turunan target. Feature engineering domain masih merupakan eksperimen lanjutan setelah baseline ini.

## 2. Arsitektur, forward pass, dan backpropagation

Untuk layer l: `z[l] = W[l] a[l-1] + b[l]`, lalu `a[l] = activation(z[l])`. Output berupa satu neuron linear yang memprediksi SOH terstandar. Semua linear layer memakai Xavier uniform gain=1 dan bias nol sebagai kebijakan inisialisasi yang sama antaraktivasi.

Di setiap mini-batch:
1. Bersihkan gradien sebelumnya (`optimizer.zero_grad`).
2. Hitung prediksi dengan forward pass dan loss terhadap y_scaled.
3. `loss.backward()` menghitung gradien setiap bobot/bias dengan chain rule.
4. Adam memperbarui parameter (`optimizer.step`).

PyTorch 2.11.0 menyediakan autograd; ini backpropagation, bukan ELM atau algoritma JST lain. Implementasi manual dari nol tidak menjadi asumsi tugas. Test terpisah mencocokkan gradien satu jaringan kecil dengan contoh hitung manual di CONTEXT.md.

![Arsitektur kandidat](03_candidate_architecture.png)

Semua kandidat memiliki minimal satu hidden layer nonlinear. Bias, jumlah parameter, output shape `(N,)` termasuk batch satu, dan hilangnya dropout saat evaluasi diperiksa. Tidak ada CNN/RNN/Transformer atau model non-MLP.

## 3. Konfigurasi acuan

Baseline: input 67, hidden [64,32], ReLU, loss MSE, output linear, Adam learning rate 0.001, batch **256**, maksimum 200 epoch, patience 15, min_delta 0.001 poin SOH, tanpa dropout/weight decay. Batch 256 menggantikan rekomendasi awal 32 demi pelatihan CPU yang praktis; nilai ini tetap pada seluruh pembanding dan tidak dituning dari validation. CPU dua thread, seed 42 untuk pencarian.

{markdown_table(comparison.round(4))}

![Learning curves](01_learning_curves.png)

Kurva objective train/validation dihitung pada bobot yang sama di akhir epoch, dalam mode eval sehingga dropout nonaktif. `batch_objective_scaled` di history adalah rata-rata mini-batch saat bobot berubah dan dropout aktif; jangan menyamakannya dengan full-train evaluation loss. Panel objective menampilkan sejak epoch 0; panel MAE diperbesar mulai epoch 5 untuk keterbacaan generalization gap. Epoch 0 adalah jaringan sebelum update dan hanya titik acuan, bukan checkpoint terpilih.

## 4. Early stopping dan checkpoint

Checkpoint `best.pt` selalu menyimpan **MAE validation minimum persis** pada epoch training 1..N, bukan hanya perbaikan yang melewati min_delta. Patience direset bila ada perbaikan signifikan >min_delta terhadap referensi patience. Dengan demikian perbaikan kecil tetap tersimpan meski tidak mereset patience.

`last.pt` menyimpan bobot pada epoch terakhir. File menyertakan state_dict, optimizer, konfigurasi, nama input, fingerprint data, epoch, serta RNG torch/shuffle. Checkpoint terbaik dimuat ulang dan MAE-nya direkonsiliasi dengan nilai minimum history. Cache hanya menggunakan run **selesai** dengan fingerprint dan file checkpoint yang cocok. Run terputus dapat diulang dari awal; runner belum menawarkan resume tengah-epoch.

Semua checkpoint di `models/mlp/<run_id>/` disimpan lokal dan diabaikan Git. Konfigurasi, history, ringkasan, dan daftar run tetap dibagikan. Rerun dari clone baru melatih ulang bila checkpoint lokal belum ada; tidak menyajikan bobot rekaan.

## 5. Matriks eksperimen bertahap

![Perbandingan tahap](02_stage_comparisons.png)

Tahap: preprocessing → hidden layer → aktivasi → loss → dua probe regularisasi. Pemenang MAE validation seed 42 diteruskan ke tahap berikutnya. Arsitektur [32], [64], [64,32], [128,64,32]; aktivasi ReLU/tanh/sigmoid; loss MSE/MAE/Huber. Huber delta=1 pada target terstandar, setara {7.1926727624524105:.4f} poin persentase SOH pada versi data ini.

File `stage_comparisons.csv` mencatat setiap kandidat dan pilihan. Ini pencarian berurutan, **bukan full factorial**: tidak mengisolasi seluruh interaksi faktor dan tidak membuktikan optimum global. Tidak membandingkan angka loss mentah lintas MSE/MAE/Huber; keputusan memakai MAE pada skala asli. Weight decay berlaku pada Adam; objective loss yang diplot tidak menambahkan penalti optimizer secara eksplisit.

Resep log/cap mengubah beberapa transformasi sekaligus; compact menguji pengurangan dua fitur. Kesimpulan manfaat tiap transformasi sendiri masih memerlukan ablation tersendiri. Probe dropout 0.1 dan weight decay 1e-4 menguji regularisasi, tanpa otomatis menyimpulkan keduanya pasti memperbaiki generalisasi.

## 6. Eksperimen epoch yang benar-benar berbeda

![Budget epoch](04_epoch_budget.png)

{markdown_table(epochs[['epoch','train_mae','validation_mae','validation_rmse','validation_r2']].round(4))}

Tabel memakai bobot **endpoint epoch 50, 100, 200** dari satu trajectory tanpa early stopping, dengan seed/urutan batch yang sama. Endpoint bukan checkpoint minimum sebelum epoch tersebut. Karena trajectory bersarang, ini eksperimen budget terkontrol, bukan tiga replikasi independen. Model fixed-epoch tidak dipakai memilih ulang kandidat; tabel ini menjelaskan efek melanjutkan training. Best checkpoint diagnostik trajectory 200 epoch ada di epoch {fixed_summary['best_epoch']} (MAE {fixed_summary['validation']['mae']:.4f}), berbeda dari endpoint 200. Perbaikan yang muncul setelah early stopping menunjukkan patience hanyalah heuristik dan dapat melewatkan perbaikan terlambat. Run diagnostik ini sengaja tidak dipromosikan diam-diam menjadi kandidat dari protokol early stopping.

## 7. Konfirmasi seed dan batas ketidakpastian

![Konfirmasi seed](05_seed_confirmation.png)

{markdown_table(tables['seed_confirmation'].round(4))}

SD menggunakan sample standard deviation (ddof=1) dari tiga seed inisialisasi/pengacakan training. Bukan confidence interval baterai/populasi atau ketidakpastian semua kemungkinan split. Konfigurasi kandidat dikunci dari pencarian seed 42 sebelum konfirmasi; tidak memilih seed terbaik untuk dilaporkan. Evaluasi akhir kelak harus mencakup seluruh seed final yang ditetapkan.

Jika keunggulan terhadap baseline kecil atau tidak konsisten antar-seed, jangan menyebut satu konfigurasi pasti lebih baik. Validation telah digunakan untuk banyak keputusan sehingga estimasi ini bisa optimistis; test independen tetap diperlukan.

## 8. Analisis error validation

![Error validation](06_validation_errors.png)

{markdown_table(tables['candidate_validation_bands'].round(4))}

Residual bertanda adalah prediksi−aktual. Tidak memotong prediksi ke 0..100 demi mempercantik metrik. Kandidat seed 42 menghasilkan {candidate['validation']['outside_0_100_pct']:.3f}% prediksi di luar batas tersebut. Selain MAE per observasi, rerata MAE per baterai kandidat adalah **{candidate['validation']['battery_mean_mae']:.4f}** poin, sehingga baterai dengan banyak baris tidak selalu mendominasi ringkasan kedua.

Rentang SOH rendah lebih jarang menurut EDA. Baca error per rentang bersama n dan jumlah baterai; skor agregat bagus tidak menjamin semua kondisi sama baik. Analisis ini tetap validation; jangan mengubah kebijakan berdasarkan test nanti.

## 9. Pemeriksaan kewajaran terhadap konstanta train

{markdown_table(constants.round(4))}

Konstanta median/rerata train bukan algoritma ML pesaing dan tidak belajar dari validation; ini kontrol kewajaran agar MLP tidak dianggap berhasil hanya karena target dominan SOH tinggi. Model prediksi yang dilatih tetap hanya MLP/backpropagation.

## 10. Overfitting, underfitting, dan yang belum dapat diklaim

Pada checkpoint terpilih, gap validation−train MAE baseline **{baseline['validation']['mae']-baseline['train']['mae']:.4f}** poin dan kandidat **{candidate['validation']['mae']-candidate['train']['mae']:.4f}** poin. Gap perlu dibaca bersama seluruh kurva, bukan satu ambang universal. Bertambahnya epoch dapat memperbaiki train dan memperburuk validation; eksperimen budget dan early stopping memperlihatkan perilaku aktualnya.

Underfitting bukan sekadar “R² kurang dari angka tertentu”. Periksa apakah train loss tetap tinggi, apakah melebihi kontrol konstanta, dan apakah kapasitas/optimasi belum memadai. Kecilnya gap juga tidak membuktikan model pasti bebas masalah, terutama pada data sintetis dan split terbatas.

## 11. Artefak, reproduksi, dan pemeriksaan

- `configs/mlp_study.json`: konfigurasi/protokol yang dipakai.
- `results/experiments/run_summary.csv`: satu baris per training run unik.
- `runs/<run_id>/`: config, history, summary, dan status. Prediksi per-baris validation dibuat lokal dan diabaikan Git.
- `models/mlp/<run_id>/`: best/last serta endpoint checkpoint fixed-epoch; lokal, diabaikan Git.
- `study.json`: kandidat sementara, baseline, seed, fingerprint, versi, dan status test belum dimuat.

Perintah: `python -m src.mlp_experiments`; atau jalankan notebook 03. Cache memakai source/data/config fingerprint, sehingga notebook dapat menampilkan ulang run yang sudah selesai. Ubah source/config/data secara sengaja bila bereksperimen dan catat hasil baru; jangan mengedit skor CSV manual. Determinisme dikonfigurasi pada CPU ini, namun hasil numerik lintas versi/hardware tidak dijamin identik.

Pengujian mencakup gradien manual vs autograd, parameter count, batch satu, eval/dropout, metrik, loader tanpa test, checkpoint epoch minimum, reload prediksi, dan cache. Hasil data aktual juga diverifikasi melalui notebook.

## 12. Langkah setelah notebook ini

Kandidat masih **sementara berdasarkan validation**. Feature engineering domain dapat diuji sebagai tahap lanjutan dengan prosedur yang sama sebelum konfigurasi final dibekukan. Setelah seluruh keputusan disepakati, barulah notebook 04 mengevaluasi test satu tahap akhir pada baseline/kandidat serta semua seed final yang dikunci. Tidak membuat klaim performa dunia nyata dari dataset sintetis ini.

Referensi: [optimasi dan backward PyTorch](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html), [checkpoint state_dict](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html), [reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html). Versi runtime dipin pada PyTorch 2.11.0; dokumentasi stable dapat menunjukkan versi lebih baru.
"""
    (output/"EXPERIMENT_REPORT.md").write_text(source,encoding="utf-8",newline="\n")
    return output/"EXPERIMENT_REPORT.md"


def build_report(root=ROOT):
    figures=build_figures(root)
    report=write_report(root)
    return {"figures":figures,"report":report}


if __name__=="__main__":
    print(build_report()["report"])
