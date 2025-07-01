# Predicting Supplier Risk with a KNN Algorithm
![hero_image](https://github.com/DillonGelman/KNN-supplier-risk-classifier/blob/main/hero_image.png)

This project provides a simple K-Nearest Neighbors (KNN) implementation for classifying supplier risk levels. Risk categories are assigned automatically based on configurable thresholds. You can now upload your own CSV data instead of relying on randomly generated samples.

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

Use the sidebar to upload a CSV file containing your supplier data and choose
the number of neighbors used in the KNN classifier. If no file is uploaded, the
app generates sample data based on slider settings for quick testing. After the
classification report, a live Seaborn pairplot of the predicted risk levels is
displayed for interactive exploration.

## Example Output

Running the script prints evaluation metrics and a classification report to the
console. The interactive Streamlit app shows a pairplot generated from the
current settings directly in the browser.
