# BNRS replication

This is a replication repository for the paper:

"Bisociative Pivoting as a Method for Moderating User Response to Text Similarity in Next-item News Recommendation"

That is currently under review.


Layout
-----------------
- `bnrs_algorithm/` — code to produce the recommendation types used in the paper (with a demo dataset: `rjac/all-the-news-2-1-Component-one`).
  - `01_preprocess.ipynb` prepares the article corpus and produces the required embeddings.
  - `02_recommend.ipynb` generates recommendation types and provides exploration of top-N recommendations.
  - `utils/` — contains `keybert_return_embeds.py` (a required KeyBERT wrapper that returns `(keyword, score, embedding)` tuples).
  - `data/` — default directory for preprocessed CSV: `01_processing_output.csv`; and for HuggingFace dataset download. 

- `bnrs_analysis/` — code to reproduce the statistical analyses reported in the paper (validation, H1a-c, H2a-c).
  - `01_preprocess.ipynb` prepares participant and recommendation-level inputs for modeling and returns model-ready outputs.
  - `02_models.ipynb` contains the code that fits the mixed-effects ordinal and binary logistic regression models and produces the reported results.
  - `data/processed/` — model-ready data used by `02_models.ipynb`. For privacy, this contains anonymized participant-level data restricted to the fields required to run the models (includes: `model_df.csv`, `supplement_df.csv`, `HxRyRz_recs.json`, `transitions.json`).

Environment
-------------------------
For running the `bnrs_algorithm/` notebooks, we used the following setup: 

```bash
conda create -n bnrs_algorithm python=3.11 numpy pandas scikit-learn networkx tqdm nltk -c conda-forge
conda activate bnrs_algorithm
conda install pytorch-c pytorch
pip install sentence-transformers keybert keyphrase-vectorizers spacy datasets
python -m spacy download en_core_web_lg
python -c "import nltk; nltk.download('stopwords')"
```

For running the `bnrs_analysis/` notebooks, we used:

```bash
conda create -n bnrs_analysis \
  python=3.11 \
  r-base=4.4.3 \
  r-essentials \
  r-lme4 \
  r-matrix \
  r-performance \
  r-insight \
  r-ordinal \
  r-mumin \
  rpy2 \
  numpy \
  pandas \
  matplotlib \
  seaborn \
  tabulate \
  scipy \
  scikit-learn \
  statsmodels \
  -c conda-forge
conda activate bnrs_analysis
```


Citations & contact
--------------------
Citation details forthcoming. 

Contact: rptkiddle@gmail.com


