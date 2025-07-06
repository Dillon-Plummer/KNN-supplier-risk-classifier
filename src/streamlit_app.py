import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from knn_risk_classifier import generate_dataset, train_knn, preprocess_data

# --- Constants ---
TARGET_COLUMN = "Risk Classification"

# --- Main App Logic ---
def main():
    st.title("Supplier Risk Classifier")

    # Initialize session state for user choices
    if 'data_source_choice' not in st.session_state:
        st.session_state.data_source_choice = None
    if 'new_supplier_data' not in st.session_state:
        st.session_state.new_supplier_data = None

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
        uploaded_file = st.sidebar.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            except Exception as e:
                st.error(f"Error reading file: {e}")
                return
        else:
            st.info("Please upload a file to continue.")
            return

    elif st.session_state.data_source_choice == 'demo':
        st.info("Using demo data, configured in the sidebar.")
        overlap = st.sidebar.slider(
            "Data Overlap", 
            min_value=0.1, max_value=1.0, value=0.4, step=0.05,
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

            # --- Train model and get base results ---
            metrics, report, cm_df, model, scaler = train_knn(processed_df, n_neighbors=int(n_neighbors))
            st.subheader("Model Metrics")
            st.json(metrics)

            # --- Predict New Supplier Section ---
            feature_cols = processed_df.drop(columns=[TARGET_COLUMN]).columns
            st.subheader("Predict a New Supplier")
            with st.form("new_supplier_form"):
                new_supplier_inputs = {}
                for col in feature_cols:
                    new_supplier_inputs[col] = st.number_input(f"Enter value for '{col}'", value=0)
                
                submitted = st.form_submit_button("Predict Risk")
                if submitted:
                    # Store new supplier data in session state for replotting
                    st.session_state.new_supplier_data = pd.DataFrame([new_supplier_inputs])

            # --- Perform prediction and display if new data exists ---
            if st.session_state.new_supplier_data is not None:
                new_data_scaled = scaler.transform(st.session_state.new_supplier_data)
                prediction = model.predict(new_data_scaled)
                st.session_state.new_supplier_data[TARGET_COLUMN] = "New Supplier" # Special value for plotting
                st.success(f"Predicted Risk Category: **{prediction[0]}**")


            # --- Plotting Section ---
            st.subheader("Pairplot of Features")
            all_features = processed_df.drop(columns=[TARGET_COLUMN])
            preds = model.predict(scaler.transform(all_features))
            pairplot_df = all_features.copy()
            pairplot_df[TARGET_COLUMN] = preds
            
            st.info("Generating plot from a sample of the data...")
            with st.spinner("Building plot..."):
                SAMPLES_FOR_PLOT = 200
                plot_df = pairplot_df.sample(min(len(pairplot_df), SAMPLES_FOR_PLOT))
                
                # Define a color palette
                unique_risks = sorted(plot_df[TARGET_COLUMN].unique())
                palette = dict(zip(unique_risks, sns.color_palette(n_colors=len(unique_risks))))

                # If there's a new supplier, add it to the plot data and update the palette
                if st.session_state.new_supplier_data is not None:
                    plot_df = pd.concat([plot_df, st.session_state.new_supplier_data], ignore_index=True)
                    palette["New Supplier"] = "red" # Assign red color

                g = sns.pairplot(plot_df, hue=TARGET_COLUMN, palette=palette, diag_kind='hist')
                st.pyplot(g.fig)

        except ValueError as e:
            st.error(e)
        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")

if __name__ == "__main__":
    main()
