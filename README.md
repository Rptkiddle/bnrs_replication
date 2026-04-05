# BNRS Replication

Replication repository for:

> Kiddle, R., Welbers, K., Kroon, A., & Trilling, D. (2026). Bisociative Pivoting as a Method for Moderating User Response to Text Similarity in Next-item News Recommendation. *ACM Transactions on Recommender Systems*.


## Layout

- `bnrs_algorithm/` — code to produce the recommendation types used in the paper (with a demo dataset: `rjac/all-the-news-2-1-Component-one`).
  - `01_preprocess.ipynb` prepares the article corpus and produces the required embeddings. **Note:** this notebook downloads the full All the News 2.1 dataset (~8.8 GB) from HuggingFace. If you only want to explore the recommendation step, you can skip this notebook and run `02_recommend.ipynb` directly using the pre-computed `data/01_processing_output.csv` already included in the repository.
  - `02_recommend.ipynb` generates recommendation types and provides exploration of top-N recommendations.
  - `utils/` — contains `keybert_return_embeds.py` (a required KeyBERT wrapper that returns `(keyword, score, embedding)` tuples).
  - `data/` — default directory for preprocessed CSV: `01_processing_output.csv`; and for HuggingFace dataset download.

- `bnrs_analysis/` — code to reproduce the statistical analyses reported in the paper (validation, H1a-c, H2a-c).
  - `01_preprocess.ipynb` documents how participant and recommendation-level inputs were prepared for modeling. **Note:** this notebook requires raw news article data and Qualtrics survey exports that cannot be redistributed due to copyright and privacy restrictions. It is included for transparency but cannot be re-run. The model-ready outputs it produces are provided in `data/processed/`.
  - `02_models.ipynb` fits the mixed-effects ordinal and binary logistic regression models and produces the reported results. **This is the notebook to run for replication** — it uses the pre-processed data in `data/processed/` and is fully self-contained.
  - `data/processed/` — model-ready data used by `02_models.ipynb`. Contains anonymized participant-level data restricted to the fields required to run the models (`model_df.csv`, `supplement_df.csv`, `HxRyRz_recs.json`, `transitions.json`).


## Setup

This project uses [uv](https://docs.astral.sh/uv/) for Python dependency management.

### Algorithm notebooks (`bnrs_algorithm/`)

```bash
uv sync --extra algorithm
uv run python -m spacy download en_core_web_lg
uv run python -c "import nltk; nltk.download('stopwords')"
```

### Analysis notebooks (`bnrs_analysis/`)

The analysis notebooks use R via `rpy2`. You need both the Python dependencies and a working R installation with the required packages.

**Python dependencies:**

```bash
uv sync --extra analysis
```

**R dependencies:**

Install R (>= 4.4) and the following packages:

```r
install.packages(c("lme4", "Matrix", "performance", "insight", "ordinal", "MuMIn"))
```

If you prefer conda for the R setup:

```bash
conda install r-base=4.4.3 r-lme4=1.1 r-matrix=1.7 r-performance=0.15 r-insight=1.4 r-ordinal=2023.12 r-mumin=1.48 -c conda-forge
```


## Citation

If you use this code, please cite the paper:

```bibtex
@article{Kiddle2026BisociativePivoting,
  author  = {Kiddle, Rupert and Welbers, Kasper and Kroon, Anne and Trilling, Damian},
  title   = {Bisociative Pivoting as a Method for Moderating User Response to Text Similarity in Next-item News Recommendation},
  journal = {ACM Transactions on Recommender Systems},
  year    = {2026}
}
```

## Contact

Rupert Kiddle — rptkiddle@gmail.com
