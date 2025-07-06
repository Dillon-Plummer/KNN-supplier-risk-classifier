import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from knn_risk_classifier import generate_dataset, train_knn, preprocess_data

# --- Constants ---
TARGET_COLUMN = "Risk Classification"
RISK_CATEGORY_MAPPING = {1: "Low", 2: "Medium", 3: "High", "New Supplier": "New Supplier"}
RISK_COLOR_MAPPING = {1: "green", 2: "yellow", 3: "orange", "New Supplier": "red"}

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
                    stats = processed_df.describe().loc[['min', 'mean', 'max'], col].to_dict()
                    st.caption(f"Stats for '{col}': Min: {stats['min']:.1f} | Mean: {stats['mean']:.1f} | Max: {stats['max']:.1f}")
                    new_supplier_inputs_key = f"new_{col}" # Unique key for each input
                    new_supplier_inputs_value = st.number_input(
                        f"Enter value for '{col}'",
                        value=float(stats['mean']),
                        key=new_supplier_inputs_key
                    )
                    new_supplier_inputs['_' + col] = new_supplier_inputs_value # Prefix to avoid conflict with df columns

                submitted = st.form_submit_button("Predict Risk")
                if submitted:
                    new_supplier_df = pd.DataFrame([new_supplier_inputs])
                    # Rename columns to match the training data (removing the underscore)
                    new_supplier_df.columns = [col.replace('_', '') for col in new_supplier_df.columns]
                    st.session_state.new_supplier_data = new_supplier_df

            # --- Perform prediction and display if new data exists ---
            if st.session_state.new_supplier_data is not None:
                new_data_scaled = scaler.transform(st.session_state.new_supplier_data)
                prediction = model.predict(new_data_scaled)
                predicted_category_num = prediction.item()
                predicted_category_label = RISK_CATEGORY_MAPPING.get(predicted_category_num, f"Category {predicted_category_num}")
                st.success(f"Predicted Risk Category: **{predicted_category_label}**")

                # Add the new supplier data with a special 'Risk Classification' for plotting
                temp_new_supplier_plot = st.session_state.new_supplier_data.copy()
                temp_new_supplier_plot["Risk Classification"] = "New Supplier"
                plot_supplier_data = temp_new_supplier_plot
            else:
                plot_supplier_data = None

            # --- Plotting Section ---
            st.subheader("Pairplot of Features")
            all_features = processed_df.drop(columns=[TARGET_COLUMN])
            preds = model.predict(scaler.transform(all_features))
            pairplot_df = all_features.copy()
            # Map numerical predictions to labels for consistent hue
            pairplot_df["Risk Classification"] = [RISK_CATEGORY_MAPPING.get(p, f"Category {p}") for p in preds]

            st.info("Generating plot from a sample of the data...")
            with st.spinner("Building plot..."):
                SAMPLES_FOR_PLOT = 200
                plot_df_sample = pairplot_df.sample(min(len(pairplot_df), SAMPLES_FOR_PLOT), random_state=42)

                # Add new supplier data to the plot dataframe
                if plot_supplier_data is not None:
                    plot_df_combined = pd.concat([plot_df_sample, plot_supplier_data], ignore_index=True)
                else:
                    plot_df_combined = plot_df_sample

                # Use the mapping for colors
                palette = RISK_COLOR_MAPPING

                g = sns.pairplot(
                    plot_df_combined,
                    hue=TARGET_COLUMN,
                    palette=palette,
                    diag_kind='hist',
                    hue_order=["Low", "Medium", "High", "New Supplier"], # Ensure correct order
                    plot_kws={'s': [50 if cat != "New Supplier" else 100 for cat in plot_df_combined['Risk Classification']]} # Adjust dot size
                )

                # Manually update legend labels
                for ax in g.axes.flat:
                    if ax is not None and ax.get_legend() is not None:
                        old_labels = ax.get_legend().get_texts()
                        new_labels = [RISK_CATEGORY_MAPPING.get(int(t.get_text()) if t.get_text().isdigit() else t.get_text(), t.get_text()) for t in old_labels]
                        for old_text, new_text in zip(old_labels, new_labels):
                            old_text.set_text(new_text)
                        break # Only need to do this once

                st.pyplot(g.fig)

        except ValueError as e:
            st.error(e)
        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")

if __name__ == "__main__":
    main()
