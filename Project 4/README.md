# Project 4 – LLM-Powered Income Prediction Explanation

## Overview

This project combines a Machine Learning model with a Large Language Model (LLM) to generate human-readable explanations for income predictions on the Adult Income Dataset.

The workflow predicts whether a person's income is greater than or less than \$50K and then uses an LLM to explain the prediction in structured JSON format.

---

## Dataset

- Dataset: Adult Income Dataset
- File Used: `cleaned_adult.csv`

Target Variable:
- income

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- OpenRouter/OpenAI API
- JSON
- jsonschema

---

## Machine Learning Pipeline

The trained pipeline includes:

- SimpleImputer (Median Strategy)
- StandardScaler
- RandomForestClassifier

The trained pipeline is saved as:

`best_model.pkl`

---

## Project Workflow

1. Load Adult Income Dataset
2. Load trained ML pipeline
3. Predict income class
4. Predict confidence probability
5. Generate structured LLM explanation
6. Validate JSON output
7. Apply PII Guardrail
8. Save prediction outputs

---

## LLM Prompt

The LLM receives:

- Person's features
- Prediction result
- Prediction confidence

It returns JSON containing:

- prediction_label
- confidence_level
- top_reason
- second_reason
- next_step

---

## JSON Validation

The LLM response is validated using a predefined JSON Schema to ensure all required fields are present.

---

## PII Guardrail

The notebook detects personally identifiable information (PII) such as:

- Email addresses
- Phone numbers

Requests containing PII are blocked before sending them to the LLM.

---

## Temperature Comparison

Two responses were generated using different temperature values:

- Temperature = 0
- Temperature = 0.7

This demonstrates the effect of deterministic versus creative generation.

---

## Output Files

The project generates:

- best_model.pkl
- llm_output.json
- prediction_results.csv
- single_prediction.csv
- final_response.txt

---

## Project Structure

```
Project4/
│
├── Question4.ipynb
├── cleaned_adult.csv
├── best_model.pkl
├── llm_output.json
├── prediction_results.csv
├── single_prediction.csv
├── final_response.txt
└── README.md
```

---

## Conclusion

This project successfully integrates a machine learning pipeline with a Large Language Model to provide interpretable income predictions. The model produces structured explanations, validates JSON outputs, applies PII protection, and demonstrates responsible AI practices for deployment.

---

## Author

**Wasid Khan**

AI / Machine Learning Student
