import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from knn_risk_classifier import generate_dataset, train_knn, preprocess_data

# --- Constants ---
TARGET_COLUMN = "Risk Classification"
RISK_LABEL_MAPPING = {1: "Low", 2: "Medium", 3: "High"}
RISK_COLOR_MAPPING = {"Low": "green", "Medium": "yellow", "High": "orange", "New Supplier": "red"}

# --- Main App Logic ---
def main():
    st.title("Supplier Risk Classifier")

    # Initialize session state
    if 'data_source_choice' not in st.session_state:
        st.session_state.data_source_choice = None
    if 'new_supplier_data' not in st.session_state:
        st.session_state.new_supplier_data = None
    if 'form_inputs' not in st.session_state:
        st.session_state.form_inputs = {}

    # --- Initial Modal ---
    if st.session_state.data_source_choice is None:
        with st.expander("Choose Your Data Source", expanded=True):
            st.write("Would you like to upload your own data or use the interactive demo data?")
            col1, col2 = st.columns(2)
            if col1.button("⬆️ Upload File", use_container_width=True):
                st.session_state.data_source_choice = 'upload'
                st.rerun()
            if col2.button("📊 Use Demo Data", use_container_width=True):
                st.session_state.data_source_choice = 'demo'
                st.rerun()
        return

    # --- Sidebar Configuration ---
    st.sidebar.header("Configuration")
    df = None

    if st.session_state.data_source_choice == 'upload':
        st.info("Coming soon! Please reload the page and try out the demo.")
        # Upload logic commented out as per original file
        return

    elif st.session_state.data_source_choice == 'demo':
        st.info("Using demo data, configured in the sidebar.")
        overlap = st.sidebar.slider(
            "Data Overlap", min_value=0.1, max_value=1.0, value=0.4, step=0.05,
            help="Controls the 'fuzziness' of the data clusters. Lower values create more distinct groups."
        )
        n_samples = st.sidebar.number_input("Number of Samples", 50, 1000, 350, 50)
        df = generate_dataset(n_samples=int(n_samples), overlap_multiplier=overlap)

    n_neighbors = st.sidebar.number_input("KNN Neighbors", 1, 20, 3, 1)

    if df is not None:
        try:
            processed_df, dropped_rows, categorical_cols = preprocess_data(df)
            
            if dropped_rows > 0:
                st.warning(f"Warning: {dropped_rows} rows were dropped due to missing data.")
            if categorical_cols:
                st.info(f"The following categorical columns were identified and encoded: {', '.join(categorical_cols)}")

            st.subheader("Data Preview (After Processing)")
            st.dataframe(processed_df.head())

            metrics, report, cm_df, model, scaler = train_knn(processed_df, n_neighbors=int(n_neighbors))
            st.subheader("Model Metrics")
            st.json(metrics)

            feature_cols = processed_df.drop(columns=[TARGET_COLUMN]).columns
            st.subheader("Predict a New Supplier")
            with st.form("new_supplier_form"):
                new_supplier_inputs = {}
                
                for col in feature_cols:
                    stats = processed_df[col].describe()
                    min_val = float(stats['min'])
                    max_val = float(stats['max'])
                    mean_val = float(stats['mean'])
                    st.caption(f"Stats for '{col}': Min: {min_val:.1f} | Mean: {mean_val:.1f} | Max: {max_val:.1f}")
                    default_value = float(st.session_state.form_inputs.get(col, mean_val))
                    new_supplier_inputs[col] = st.number_input(
                        label=f"Enter value for '{col}'",
                        min_value=min_val,
                        max_value=max_val,
                        value=default_value
                    )
                
                submitted = st.form_submit_button("Predict Risk")
                if submitted:
                    st.session_state.form_inputs = new_supplier_inputs
                    st.session_state.new_supplier_data = pd.DataFrame([new_supplier_inputs])
                    st.rerun()

            # --- Plotting Logic ---
            st.subheader("Pairplot of Features")
            with st.spinner("Building plot..."):
                plot_df = processed_df.copy()

                if st.session_state.new_supplier_data is not None:
                    new_supplier_row = st.session_state.new_supplier_data.copy()
                    new_supplier_scaled = scaler.transform(new_supplier_row[feature_cols])
                    prediction = model.predict(new_supplier_scaled)
                    prediction_label = RISK_LABEL_MAPPING.get(prediction[0], "Unknown")
                    st.success(f"Predicted Risk Category: **{prediction_label}**")
                    new_supplier_row[TARGET_COLUMN] = prediction[0]
                    new_supplier_row['Risk Label'] = "New Supplier"
                    plot_df['Risk Label'] = plot_df[TARGET_COLUMN].map(RISK_LABEL_MAPPING)
                    plot_df = pd.concat([plot_df, new_supplier_row], ignore_index=True)
                else:
                    plot_df['Risk Label'] = plot_df[TARGET_COLUMN].map(RISK_LABEL_MAPPING)

                SAMPLES_FOR_PLOT = 200
                if len(plot_df) > SAMPLES_FOR_PLOT:
                    plot_sample_df = plot_df.sample(SAMPLES_FOR_PLOT)
                    if st.session_state.new_supplier_data is not None and "New Supplier" not in plot_sample_df['Risk Label'].values:
                         plot_sample_df = pd.concat([plot_sample_df.iloc[:-1], plot_df[plot_df['Risk Label'] == "New Supplier"]])
                else:
                    plot_sample_df = plot_df

                # --- FIX ---
                # 1. Create a new column for marker size. This is the idiomatic way for pairplot.
                plot_sample_df['dot_size'] = plot_sample_df['Risk Label'].apply(lambda x: 150 if x == "New Supplier" else 40)
                
                # 2. Remove the old dot_sizes list and use the `size` and `sizes` parameters.
                g = sns.pairplot(
                    plot_sample_df,
                    vars=feature_cols,
                    hue='Risk Label',
                    palette=RISK_COLOR_MAPPING,
                    diag_kind='hist',
                    size='dot_size', # Tell seaborn to use this column for sizing
                    sizes=(40, 150)  # Define the min and max size of the markers
                )
                # --- END FIX ---
                
                st.pyplot(g.fig)

        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            st.exception(e)

if __name__ == "__main__":
    main()
