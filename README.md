# Context
This is a personal project to improve my machine learning, data visualisation and various other skills on the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

# How to execute

You can run the Python scripts directly using `uv`, which handles dependencies automatically. To run the classification pipeline:

```bash
uv run classification.py
```

*Note: The script automatically downloads the Kaggle dataset using `kagglehub`, so you don't need to download it manually.*

### Files & Notebooks

* **classification.py**: A LightGBM pipeline that handles extreme class imbalance (578:1). It features automated hyperparameter tuning via **Optuna** (using early-stopping pruning mapped to a custom **PR-AUC** metric).
* **Archives/**: Contains previous attempts and experiments.
    * **1. data-exploration.ipynb**: First analysis and EDA.
    * **2. ml-pipeline.ipynb**: Machine learning approach with scikit-learn.
    * **3. neural-network.ipynb**: Neural network approach with TensorFlow.