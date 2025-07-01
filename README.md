# Predicting Supplier Risk with a KNN Algorithm
![hero_image](https://github.com/DillonGelman/KNN-supplier-risk-classifier/blob/main/hero_image.png)

This project provides a simple K-Nearest Neighbors (KNN) implementation for classifying supplier risk levels. The dataset is generated synthetically and risk categories are assigned based on configurable thresholds.

## Usage

Run the classifier from the project root:

```bash
python src/knn_risk_classifier.py
```

The script outputs basic evaluation metrics and a classification report for the model.

## Streamlit App

You can also explore the model interactively using [Streamlit](https://streamlit.io/).
Launch the app with:

```bash
streamlit run src/streamlit_app.py
```

Use the sidebar to tweak dataset generation parameters and the number of
neighbors used in the KNN classifier. Results are updated automatically when
parameters change and a confusion matrix visualization is displayed alongside the
metrics and classification report.
