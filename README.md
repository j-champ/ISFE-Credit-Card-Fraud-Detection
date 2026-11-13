# ISFE: Credit Card Fraud Detection

## Project 26

**Project Title:** Incremental Slimming–Fattening Ensemble for Imbalanced Classification (ISFE)

**Domain:** Machine Learning / Imbalanced Classification

## 1. Overview

This project investigates machine learning techniques for highly imbalanced credit-card fraud detection.

It compares conventional classification algorithms with engineering adaptations inspired by the Incremental Slimming–Fattening Ensemble (ISFE) approach.

The implementation evaluates feature-space refinement, experimental feature views, clustering-based undersampling and ensemble classification.

**Important:** The SFR, ISFR and ISFE implementations in this project are engineering adaptations based on the available reference paper abstract, not verified reproductions of the complete published algorithms.

## 2. Dataset

The project uses the Credit Card Fraud Detection dataset.

| Attribute | Value |
|---|---:|
| Total transactions | 284,807 |
| Input features | 30 |
| Legitimate transactions | 284,315 |
| Fraudulent transactions | 492 |
| Fraud prevalence | 0.1727% |
| Imbalance ratio | Approximately 577.88:1 |

The dataset contains anonymized PCA-transformed features V1–V28, along with Time and Amount.

## 3. Methodology

The experimental workflow consists of:

1. Dataset validation and exploratory data analysis.
2. Duplicate-aware stratified data splitting.
3. Baseline classification using Decision Tree, Random Forest, Linear SVM and hard voting.
4. Feature slimming and fattening refinement adaptation using scaling, PCA and pairwise interactions.
5. ISFR adaptation using three experimental feature views.
6. ISFE adaptation using clustering-based majority-class undersampling and three Random Forest classifiers.
7. Hyperparameter tuning.
8. Final evaluation on an untouched test partition.

## 4. Experimental Results

The dataset was divided into the following partitions:

| Partition | Samples | Fraud cases |
|---|---:|---:|
| Training | 182,276 | 315 |
| Validation | 45,570 | 79 |
| Testing | 56,961 | 98 |

### Final Test Results

| Model | Macro F1 | Balanced Accuracy | Fraud Precision | Fraud Recall | Fraud F1 |
|---|---:|---:|---:|---:|---:|
| Baseline Random Forest | 0.9293 | 0.8877 | 0.9620 | 0.7755 | 0.8588 |
| Tuned Random Forest | 0.9245 | 0.8877 | 0.9383 | 0.7755 | 0.8492 |
| ISFR Adaptation | 0.8748 | 0.8061 | 0.9677 | 0.6122 | 0.7500 |
| ISFE Adaptation | 0.6102 | 0.9538 | 0.1288 | 0.9184 | 0.2258 |

The baseline Random Forest achieved the strongest overall Macro F1 and Fraud F1.

The ISFE adaptation achieved the highest fraud recall, but its precision was considerably lower due to a large number of false-positive predictions.

## 5. Project Structure

```text
isfe_creditcard_project/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_dataset_validation.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_baseline_models.ipynb
│   ├── 05_sfr_implementation.ipynb
│   ├── 06_isfr_implementation.ipynb
│   ├── 07_isfe_ensemble.ipynb
│   ├── 08_hyperparameter_tuning.ipynb
│   └── 09_final_evaluation.ipynb
│
├── src/
│   └── isfe_pipeline.py
│
├── models/
├── reports/
├── requirements.txt
└── README.md
```

## 6. Installation

Clone or download the project.

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 7. Running the Pipeline

From the project root, execute:

```bash
python src/isfe_pipeline.py
```

The script loads the saved models and preprocessing objects, generates predictions and evaluates all four approaches using the saved test partition.

## 8. Generated Outputs

The pipeline generates:

- `reports/pipeline_test_predictions.csv`
- `reports/pipeline_test_results.csv`

Additional experiment results, figures and evaluation outputs are stored in the `reports/` directory.

## 9. Technologies Used

- Python
- NumPy
- pandas
- scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Jupyter Notebook

## 10. Limitations

- The SFR, ISFR and ISFE implementations are engineering adaptations based on the available abstract.
- The implemented ISFR view comparison does not constitute a verified incremental update mechanism.
- The ISFE adaptation has a substantial false-positive rate.
- Evaluation was conducted using one credit-card fraud dataset.
- Further experiments on additional imbalanced datasets are needed.

## 11. Conclusion

This project investigates conventional machine learning classifiers and feature-refinement and clustering-based ensemble adaptations for imbalanced credit-card fraud detection.

The baseline Random Forest achieved the best overall classification performance, while the adapted ISFE approach achieved the highest fraud recall.

The findings highlight the importance of considering minority-class precision, recall, Macro F1 and balanced accuracy when evaluating highly imbalanced classification problems.

## 12. References

Xu, Yuhong, Wang, Zaibo, and Huang, Peijie. *Incremental slimming–fattening ensemble for imbalanced classification*. DOI: https://doi.org/10.1016/j.engappai.2026.114201.
