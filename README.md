# BNRS replication

This is a replication repository for the paper:

"Bisociative Pivoting as a Method for Moderating User Response to Text Similarity in Next-item News Recommendation"

That is currently under review.


Layout
-----------------
- `bnrs_algorithm/` — code to produce the recommendation types used in the paper (with a demo dataset: `rjac/all-the-news-2-1-Component-one`).
  - `01_preprocess.ipynb` prepares the article corpus and produces the required embeddings. **Note:** this notebook downloads the full All the News 2.1 dataset (~8.8 GB) from HuggingFace. If you only want to explore the recommendation step, you can skip this notebook and run `02_recommend.ipynb` directly using the pre-computed `data/01_processing_output.csv` already included in the repository.
  - `02_recommend.ipynb` generates recommendation types and provides exploration of top-N recommendations.
  - `utils/` — contains `keybert_return_embeds.py` (a required KeyBERT wrapper that returns `(keyword, score, embedding)` tuples).
  - `data/` — default directory for preprocessed CSV: `01_processing_output.csv`; and for HuggingFace dataset download.

- `bnrs_analysis/` — code to reproduce the statistical analyses reported in the paper (validation, H1a-c, H2a-c).
  - `01_preprocess.ipynb` documents how participant and recommendation-level inputs were prepared for modeling. **Note:** this notebook requires raw news article data and Qualtrics survey exports that cannot be redistributed due to copyright and privacy restrictions. It is included for transparency but cannot be re-run. The model-ready outputs it produces are provided in `data/processed/`.
  - `02_models.ipynb` fits the mixed-effects ordinal and binary logistic regression models and produces the reported results. **This is the notebook to run for replication** — it uses the pre-processed data in `data/processed/` and is fully self-contained.
  - `data/processed/` — model-ready data used by `02_models.ipynb`. Contains anonymized participant-level data restricted to the fields required to run the models (`model_df.csv`, `supplement_df.csv`, `HxRyRz_recs.json`, `transitions.json`).

Environment
-------------------------
For running the `bnrs_algorithm/` notebooks, we used the following setup: 

```bash
conda create -n bnrs_algorithm python=3.11 numpy=2.3 pandas=2.3 scikit-learn=1.7 networkx=3.5 tqdm=4.67 nltk=3.9 -c conda-forge
conda activate bnrs_algorithm
conda install pytorch=2.9 -c pytorch
pip install sentence-transformers==5.1 keybert==0.9 keyphrase-vectorizers==0.0.13 spacy==3.8 datasets==4.2
python -m spacy download en_core_web_lg
python -c "import nltk; nltk.download('stopwords')"
```

For running the `bnrs_analysis/` notebooks, we used:

```bash
conda create -n bnrs_analysis \
  python=3.11 \
  r-base=4.4.3 \
  r-essentials=4.4 \
  r-lme4=1.1 \
  r-matrix=1.7 \
  r-performance=0.15 \
  r-insight=1.4 \
  r-ordinal=2023.12 \
  r-mumin=1.48 \
  rpy2=3.6 \
  numpy=2.3 \
  pandas=2.3 \
  matplotlib=3.10 \
  seaborn=0.13 \
  tabulate=0.9 \
  scipy=1.16 \
  scikit-learn=1.7 \
  statsmodels=0.14 \
  -c conda-forge
conda activate bnrs_analysis
```


Citations & contact
--------------------
Citation details forthcoming. 

Contact: rptkiddle@gmail.com


