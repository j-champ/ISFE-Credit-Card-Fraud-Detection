# Implementation sequence

1. Dataset integrity and duplicate analysis.
2. EDA with fraud-focused visualizations.
3. Stratified train/validation/test design; fit preprocessing on training data only.
4. Decision Tree, Random Forest, SVM, and simple ensemble baselines.
5. Implement and test the SFR components from the paper.
6. Implement ISFR's iterative view repository and selection logic.
7. Implement final ClusterUS-based subsets and ISFE ensemble.
8. Tune using training-only cross-validation.
9. Evaluate once on untouched test data; perform ablation and error analysis.
10. Build reproducible prediction script, report, README, presentation, and demo.
