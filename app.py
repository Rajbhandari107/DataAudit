
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from analysis import (
    analyze_tabular,
    analyze_image,
    evaluate_classifier,
    evaluate_defects
)

from reports import (
    create_pdf_report,
    create_json_report
)


st.set_page_config(
    page_title="DataAudit",
    page_icon="🔍",
    layout="wide"
)

st.title("DataAudit")
st.caption(
    "Multi-Modal Data Quality Assessment and Anomaly Detection"
)

if "report" not in st.session_state:
    st.session_state.report = None


def show_evaluation(result):

    st.subheader("Evaluation Results")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Accuracy", f"{result['accuracy']:.3f}")
    c2.metric("Precision", f"{result['precision']:.3f}")
    c3.metric("Recall", f"{result['recall']:.3f}")
    c4.metric("F1 Score", f"{result['f1_score']:.3f}")

    cm = np.array(result["confusion_matrix"])
    labels = result["labels"]

    st.subheader("Confusion Matrix")

    matrix = pd.DataFrame(
        cm,
        index=[f"Actual: {x}" for x in labels],
        columns=[f"Predicted: {x}" for x in labels]
    )

    st.dataframe(matrix)

    fig, ax = plt.subplots()

    ax.imshow(cm)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30)
    ax.set_yticklabels(labels)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    for i in range(len(cm)):
        for j in range(len(cm[i])):
            ax.text(
                j, i, str(cm[i][j]),
                ha="center", va="center"
            )

    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)


def show_downloads():

    if st.session_state.report is None:
        return

    st.subheader("Download Report")

    report = st.session_state.report

    st.download_button(
        "Download PDF Report",
        data=create_pdf_report(report),
        file_name="DataAudit_Report.pdf",
        mime="application/pdf"
    )

    st.download_button(
        "Download JSON Report",
        data=create_json_report(report),
        file_name="DataAudit_Report.json",
        mime="application/json"
    )


page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Tabular Analysis",
        "Image Analysis",
        "Defect Evaluation"
    ]
)


# ==================================
# HOME
# ==================================

if page == "Home":

    st.subheader("Welcome to DataAudit")

    st.write("""
    DataAudit is a multi-modal dataset quality assessment framework.

    **Tabular Analysis**
    - Missing value detection
    - Duplicate detection
    - Statistical outlier detection

    **Image Analysis**
    - Image validation
    - Blur detection
    - Brightness and contrast
    - Resolution assessment

    **Evaluation**
    - Classification performance
    - Confusion matrix
    - Accuracy, precision, recall and F1-score

    **Reporting**
    - Downloadable PDF reports
    - JSON reports
    """)


# ==================================
# TABULAR ANALYSIS
# ==================================

elif page == "Tabular Analysis":

    st.header("Tabular Dataset Analysis")

    file = st.file_uploader(
        "Upload CSV or Excel",
        type=["csv", "xlsx"]
    )

    if file:

        try:

            if file.name.endswith(".csv"):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)

            st.subheader("Dataset Preview")
            st.dataframe(df.head())

            results = analyze_tabular(df)

            c1, c2, c3 = st.columns(3)

            c1.metric("Rows", results["rows"])
            c2.metric("Columns", results["columns"])
            c3.metric(
                "Missing Values",
                results["missing_values"]
            )

            st.subheader("Missing Values by Column")

            st.dataframe(pd.DataFrame(
                list(results["missing_by_column"].items()),
                columns=["Column", "Missing Values"]
            ))

            st.subheader("Duplicate Rows")

            st.metric(
                "Duplicate Rows",
                results["duplicate_rows"]
            )

            st.subheader("Outlier Detection")

            st.dataframe(pd.DataFrame(
                list(results["outliers"].items()),
                columns=["Column", "IQR Outliers"]
            ))

            report = {
                "dataset_name": file.name,
                "modality": "Tabular",
                "sample_count": results["rows"],
                "quality_metrics": results
            }

            st.session_state.report = report

            st.subheader("Classification Evaluation")

            target = st.selectbox(
                "Select target column",
                ["None"] + df.columns.tolist()
            )

            if st.button("Run Classification"):

                if target == "None":
                    st.warning("Please select a target column.")

                else:

                    try:

                        with st.spinner("Training model..."):

                            evaluation = evaluate_classifier(
                                df, target
                            )

                        report["evaluation"] = evaluation
                        report["target_column"] = target

                        st.session_state.report = report

                        show_evaluation(evaluation)

                    except Exception as error:
                        st.error(str(error))

            show_downloads()

        except Exception as error:
            st.error(f"Error: {error}")


# ==================================
# IMAGE ANALYSIS
# ==================================

elif page == "Image Analysis":

    st.header("Image Dataset Analysis")

    files = st.file_uploader(
        "Upload images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if files:

        results = []

        for file in files:

            try:
                results.append(analyze_image(file))

            except Exception as error:
                results.append({
                    "image": file.name,
                    "error": str(error)
                })

        results_df = pd.DataFrame(results)

        st.subheader("Image Quality Results")
        st.dataframe(results_df)

        valid = [
            x for x in results if "error" not in x
        ]

        report = {
            "dataset_name": "Image Dataset",
            "modality": "Image",
            "sample_count": len(files),
            "quality_metrics": {
                "total_images": len(files),
                "valid_images": len(valid),
                "possible_blur": sum(
                    x["possible_blur"] for x in valid
                ),
                "low_resolution": sum(
                    x["low_resolution"] for x in valid
                )
            }
        }

        st.session_state.report = report

        show_downloads()


# ==================================
# DEFECT EVALUATION
# ==================================

elif page == "Defect Evaluation":

    st.header("Defect Detection Evaluation")

    st.write("""
    Upload a CSV with two columns:
    - Actual label
    - Predicted label

    0 = Normal
    1 = Defect
    """)

    file = st.file_uploader(
        "Upload evaluation CSV",
        type=["csv"],
        key="defect_file"
    )

    if file:

        df = pd.read_csv(file)

        st.dataframe(df.head())

        if len(df.columns) >= 2:

            actual_col = st.selectbox(
                "Actual labels",
                df.columns
            )

            predicted_col = st.selectbox(
                "Predicted labels",
                [x for x in df.columns if x != actual_col]
            )

            if st.button("Evaluate Defects"):

                try:

                    result = evaluate_defects(
                        df[actual_col],
                        df[predicted_col]
                    )

                    show_evaluation(result)

                    st.session_state.report = {
                        "dataset_name": file.name,
                        "modality": "Defect Detection",
                        "sample_count": len(df),
                        "quality_metrics": {},
                        "evaluation": result
                    }

                except Exception as error:
                    st.error(str(error))

        show_downloads()
