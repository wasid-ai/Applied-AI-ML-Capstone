## Decision Threshold Sensitivity

The Logistic Regression model generated prediction probabilities using `predict_proba()`. Decision thresholds from **0.30 to 0.70** were evaluated to study their impact on Precision, Recall, and F1-score.

Lower thresholds generally increase Recall but may reduce Precision because more samples are classified as positive. Higher thresholds increase Precision but may reduce Recall by missing positive samples.

The threshold that produced the highest F1-score was selected as the best balance between Precision and Recall for this dataset.

---

## Regularization Experiment

A second Logistic Regression model was trained using **C = 0.01** to apply stronger L2 regularization.

The baseline model (**C = 1.0**) and the strongly regularized model (**C = 0.01**) were compared using:

* Precision
* Recall
* AUC Score

The parameter **C** controls the inverse strength of regularization.

* Smaller C → Stronger Regularization
* Larger C → Weaker Regularization

Strong regularization reduces model complexity and can reduce overfitting, but if it is too strong it may also reduce predictive performance.

---

## Bootstrap Confidence Interval

A bootstrap analysis with **500 resamples** was performed to estimate the reliability of the AUC difference between the two Logistic Regression models.

For each bootstrap sample:

* Test samples were drawn with replacement.
* The AUC of both models was computed.
* The AUC difference (C=1.0 − C=0.01) was recorded.

Finally, the following statistics were calculated:

* Mean AUC Difference
* Lower 95% Confidence Interval
* Upper 95% Confidence Interval

If the confidence interval excludes zero, the baseline model consistently performs better. If the interval includes zero, the observed difference may not be statistically reliable.

---

## Results Summary

The project successfully implemented:

* Data preprocessing
* One-Hot Encoding
* Train-Test Split
* Standard Scaling
* Linear Regression
* Ridge Regression
* Logistic Regression
* ROC Curve and AUC
* Threshold Analysis
* Logistic Regularization Comparison
* Bootstrap Confidence Interval

All models were successfully trained and evaluated using the cleaned Adult Census dataset.

---

## Repository Contents

* Question_2_Supervised_Machine_Learning.ipynb
* cleaned_adult.csv
* README.md
* roc_curve.png
* ridge_vs_linear.csv
* logistic_comparison.csv
* threshold_analysis.csv
* bootstrap_auc_results.csv
* classification_report.csv
* confusion_matrix.csv
* top3_coefficients.csv
* model_results.json

---

## Conclusion

This project demonstrates a complete supervised machine learning pipeline for both regression and binary classification. Proper preprocessing, feature encoding, scaling, model training, evaluation, regularization, threshold analysis, and bootstrap validation were performed following machine learning best practices. The results show that careful preprocessing and appropriate evaluation techniques improve the reliability and interpretability of predictive models.

