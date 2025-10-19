# 🎬 Reproducible C++ Matrix Factorization for Movie Recommendation  
**ALS, SGD, and ALS→SGD Fine-Tuning on MovieLens 100K / 1M / 10M**

[![C++](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![Eigen](https://img.shields.io/badge/Eigen-3.4-lightgrey.svg)](https://eigen.tuxfamily.org)
[![Datasets](https://img.shields.io/badge/Datasets-MovieLens%20100K%2F1M%2F10M-orange.svg)](https://grouplens.org/datasets/movielens/)

A lean **C++17 / Eigen** implementation of matrix factorization for **top-N recommendation**, including:

- **ALS** (closed-form alternating least squares)
- **SGD** (stochastic gradient descent)
- **HYBRID**: **ALS warm-start + a few SGD epochs** (ranking-oriented fine-tuning)

We evaluate on **MovieLens 100K / 1M / 10M** with user-wise **80/10/10** splits and report **RMSE, MAE, Recall@10, NDCG@10, MAP@10, Coverage**. The pipeline is **deterministic** (fixed seeds) and exports **CSV** for all metrics + optional logs for convergence plots.

---

## 📂 Project Structure

recsys/
├─ include/ # C++ headers (io.h, als.h, sgd.h, metrics.h, utils.h, split.h, ...)
├─ src/ # C++ sources (main.cpp, io.cpp, als.cpp, sgd.cpp, metrics.cpp, utils.cpp, split.cpp)
├─ third_party/
│ └─ eigen/ # Eigen 3.4 (copy Eigen/ and unsupported/ here)
├─ data/
│ ├─ ml100k/ # raw MovieLens 100K
│ ├─ ml1m/ # raw MovieLens 1M
│ └─ ml10m/ # raw MovieLens 10M
├─ results/ # CSV outputs per run (auto-created)
├─ bin/ # compiled binary (recsys / recsys.exe)
├─ plots.py # optional plotting script (matplotlib)
└─ README.md

bash
Copiar código

> **Eigen setup:** download Eigen 3.4 and place **both** folders `Eigen/` and `unsupported/` inside `third_party/eigen/`, ending up with  
> `third_party/eigen/Eigen/...` and `third_party/eigen/unsupported/...`

---

## 🧱 Build

### Windows (MinGW / VS Code Terminal)

```bash
mkdir -p bin
g++ -std=c++17 -O3 -march=native -DEIGEN_NO_DEBUG ^
  -I "third_party/eigen" -I "include" ^
  -o bin/recsys ^
  src/main.cpp src/io.cpp src/split.cpp src/als.cpp src/sgd.cpp src/metrics.cpp src/utils.cpp
Linux / macOS (clang++/g++)
bash
Copiar código
mkdir -p bin
g++ -std=c++17 -O3 -march=native -DEIGEN_NO_DEBUG \
  -I third_party/eigen -I include \
  -o bin/recsys \
  src/main.cpp src/io.cpp src/split.cpp src/als.cpp src/sgd.cpp src/metrics.cpp src/utils.cpp
If the compiler cannot find Eigen, double-check the -I third_party/eigen include path and that the internal folder names are exactly Eigen/ and unsupported/.

📦 Datasets
Download MovieLens from GroupLens and place files like this:

kotlin
Copiar código
data/
├─ ml100k/   (e.g., u.data / u.item or ratings.csv)
├─ ml1m/     (e.g., ratings.dat / movies.dat or ratings.csv)
└─ ml10m/    (e.g., ratings.dat / movies.dat or ratings.csv)
Use the correct --format flag below (ml100k, ml1m, ml10m, or csv).

🧰 Command-Line Interface
General pattern:

bash
Copiar código
bin/recsys \
  --dataset <ml100k|ml1m|ml10m> \
  --data_dir data/<ml100k|ml1m|ml10m> \
  --format <ml100k|ml1m|ml10m|csv> \
  --algo <ALS|SGD|HYBRID> \
  --factors 50 --lambda 0.1 \
  --als_iters 6 \
  --sgd_epochs 2 --lr 0.005 --neg 500 \
  --topk 10 \
  --seed 42 \
  --outdir results/<ml100k|ml1m|ml10m>/<ALGO>_seed42
🔹 ALS
ML-100K

bash
Copiar código
bin/recsys --dataset ml100k --data_dir data/ml100k --format ml100k \
  --algo ALS --factors 50 --lambda 0.1 --als_iters 6 --topk 10 \
  --seed 42 --outdir results/ml100k/ALS_seed42
ML-1M

bash
Copiar código
bin/recsys --dataset ml1m --data_dir data/ml1m --format ml1m \
  --algo ALS --factors 50 --lambda 0.1 --als_iters 6 --topk 10 \
  --seed 42 --outdir results/ml1m/ALS_seed42
ML-10M

bash
Copiar código
bin/recsys --dataset ml10m --data_dir data/ml10m --format ml10m \
  --algo ALS --factors 50 --lambda 0.1 --als_iters 6 --topk 10 \
  --seed 42 --outdir results/ml10m/ALS_seed42
🔹 SGD (from random init)
bash
Copiar código
bin/recsys --dataset ml100k --data_dir data/ml100k --format ml100k \
  --algo SGD --factors 50 --lambda 0.1 --sgd_epochs 30 --lr 0.01 --neg 500 \
  --topk 10 --seed 42 --outdir results/ml100k/SGD_seed42
🔹 HYBRID (ALS warm-start + a few SGD epochs)
ML-100K

bash
Copiar código
bin/recsys --dataset ml100k --data_dir data/ml100k --format ml100k \
  --algo HYBRID --factors 50 --lambda 0.1 --als_iters 6 \
  --sgd_epochs 2 --lr 0.005 --neg 500 \
  --topk 10 --seed 42 --outdir results/ml100k/HYBRID_seed42
ML-1M

bash
Copiar código
bin/recsys --dataset ml1m --data_dir data/ml1m --format ml1m \
  --algo HYBRID --factors 50 --lambda 0.1 --als_iters 6 \
  --sgd_epochs 2 --lr 0.005 --neg 500 \
  --topk 10 --seed 42 --outdir results/ml1m/HYBRID_seed42
ML-10M (more negatives due to scale)

bash
Copiar código
bin/recsys --dataset ml10m --data_dir data/ml10m --format ml10m \
  --algo HYBRID --factors 50 --lambda 0.1 --als_iters 6 \
  --sgd_epochs 2 --lr 0.005 --neg 1000 \
  --topk 10 --seed 42 --outdir results/ml10m/HYBRID_seed42
📝 Outputs (per run)
Each run writes to results/<dataset>/<ALGO>_seedXX/:

error_metrics.csv — RMSE/MAE for train/val/test

topn_metrics.csv — Recall@10, NDCG@10, MAP@10, Coverage

topK.csv (optional) — per-user Top-K recommendations

log.txt (optional) — config/seed; if enabled, per-iter/epoch traces + timings

CSV schemas

error_metrics.csv

bash
Copiar código
split,model,rmse,mae,seed,dataset
test,ALS,0.90,0.72,42,ml1m
...
topn_metrics.csv

python-repl
Copiar código
dataset,model,k,recall,ndcg,map,coverage,seed
ml100k,HYBRID,10,0.21,0.12,0.08,0.64,42
...
📈 Plotting (optional)
Requirements:

bash
Copiar código
pip install pandas matplotlib
Run from repo root:

bash
Copiar código
python plots.py
Outputs (saved under Plots/ or figs_ieee_pub/):

fig_recall_pub.pdf, fig_ndcg_pub.pdf, fig_map_pub.pdf, fig_coverage_pub.pdf

fig_rmse_pub.pdf, fig_mae_pub.pdf

(If you log per-iteration/epoch RMSE on val) convergence curves for ALS/SGD

⚙️ Key Hyperparameters
Flag	Meaning	Typical
--factors	Latent dimensions d	50
--lambda	L2 regularization	0.1–0.2
--als_iters	ALS iterations	6
--sgd_epochs	SGD epochs (fine-tune: 1–3)	2–3
--lr	SGD learning rate	0.001–0.01
--neg	Negatives per user (ranking bias)	500 (100K/1M), 1000 (10M)
--topk	K for ranking metrics	10
--seed	RNG seed (deterministic)	42

🔬 What We Evaluate
Pointwise accuracy: RMSE, MAE (lower is better)

Ranking quality: Recall@10, NDCG@10, MAP@10 (higher is better)

Catalog Coverage: fraction of distinct recommended items (higher is better)

Observed pattern: The HYBRID (ALS→SGD) model improves Recall@10/NDCG@10 over pure ALS while keeping RMSE/MAE comparable. At larger scales, coverage can decrease (popularity concentration); consider re-ranking or diversity-oriented regularizers if needed.

🔁 Reproducibility
Single C++ binary; same CLI across datasets

Fixed seeds; user-wise 80/10/10 splits

All metrics exported to CSV

Plot scripts read CSV and produce publication-ready figures

Tip: Instrument the code to log RMSE(val) after each ALS iteration / SGD epoch to produce convergence plots.

🛠️ Troubleshooting
“cannot open output file bin/recsys: No such file or directory”
Create bin/ first: mkdir -p bin

“Eigen headers not found”
Ensure third_party/eigen/Eigen and third_party/eigen/unsupported exist and you’re passing -I third_party/eigen.

Windows paths with spaces
Quote include paths: -I "third_party/eigen" -I "include"

📚 References (selection)
Koren, Bell, Volinsky. Matrix Factorization Techniques for Recommender Systems. Computer, 2009.

Rendle et al. Neural Collaborative Filtering vs. Matrix Factorization Revisited. RecSys, 2020.

Zangerle et al. Evaluating Recommender Systems: Survey and Framework. ACM TORS, 2022.

Krichene & Rendle. On Sampled Metrics for Item Recommendation. ACM TOIS, 2022.

Rendle et al. iALS++: Subspace Optimization. arXiv, 2021.

(See the paper’s refs.bib for the full list.)

👤 Author
Andres Alexander Basantes Balcazar
School of Mathematical and Computational Sciences — Yachay Tech University (Ecuador)
📧 andres.basantes@yachaytech.edu.ec
