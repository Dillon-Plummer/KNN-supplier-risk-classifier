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
        st.info("Coming soon!  Please reload the page and try out the demo.")
        # uploaded_file = st.sidebar.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])
        # if uploaded_file:
        #     try:
        #         df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        #     except Exception as e:
        #         st.error(f"Error reading file: {e}")
        #         return
        # else:
        #     st.info("Please upload a file to continue.")
        #     return

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
                    # Use .get() for safety and robustly create the input
                    mean_val = stats.get('mean', 0.0)
                    st.caption(f"Stats for '{col}': Min: {stats.get('min', 0.0):.1f} | Mean: {mean_val:.1f} | Max: {stats.get('max', 0.0):.1f}")
                    default_value = st.session_state.form_inputs.get(col, float(mean_val))
                    new_supplier_inputs[col] = st.number_input(f"Enter value for '{col}'", value=default_value)
                
                submitted = st.form_submit_button("Predict Risk")
                if submitted:
                    st.session_state.form_inputs = new_supplier_inputs
                    st.session_state.new_supplier_data = pd.DataFrame([new_supplier_inputs])

            # --- Final Prediction and Plotting Logic ---
            all_features = processed_df.drop(columns=[TARGET_COLUMN])
            base_predictions = model.predict(scaler.transform(all_features))
            plot_df = all_features.copy()
            plot_df[TARGET_COLUMN] = base_predictions

            if st.session_state.new_supplier_data is not None:
                new_data_scaled = scaler.transform(st.session_state.new_supplier_data)
                prediction_array = model.predict(new_data_scaled)
                prediction_scalar = prediction_array.item()
                prediction_label = RISK_LABEL_MAPPING.get(prediction_scalar, "Unknown")
                st.success(f"Predicted Risk Category: **{prediction_label}**")

                new_row = st.session_state.new_supplier_data.copy()
                new_row[TARGET_COLUMN] = prediction_scalar
                plot_df = pd.concat([plot_df, new_row], ignore_index=True)
            
            plot_df['Risk Label'] = plot_df[TARGET_COLUMN].map(RISK_LABEL_MAPPING)
            if st.session_state.new_supplier_data is not None:
                plot_df.iloc[-1, plot_df.columns.get_loc('Risk Label')] = "New Supplier"

            st.subheader("Pairplot of Features")
            st.info("Generating plot from a sample of the data...")
            with st.spinner("Building plot..."):
                SAMPLES_FOR_PLOT = 200
                plot_sample_df = plot_df.sample(min(len(plot_df), SAMPLES_FOR_PLOT))
                
                if st.session_state.new_supplier_data is not None:
                    if "New Supplier" not in plot_sample_df['Risk Label'].values:
                         plot_sample_df = pd.concat([plot_sample_df, plot_df[plot_df['Risk Label'] == "New Supplier"]])

                dot_sizes = [100 if cat == "New Supplier" else 40 for cat in plot_sample_df['Risk Label']]

                g = sns.pairplot(
                    plot_sample_df,
                    vars=feature_cols,
                    hue='Risk Label',
                    palette=RISK_COLOR_MAPPING,
                    diag_kind='hist',
                    plot_kws={'s': dot_sizes}
                )
                st.pyplot(g.fig)

        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
