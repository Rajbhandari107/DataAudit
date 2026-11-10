
import io
import json
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Table, TableStyle, Image as PDFImage
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch


def create_json_report(data):

    return json.dumps(
        data,
        indent=4,
        default=str
    )


def create_pdf_report(data):

    buffer = io.BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4
    )

    styles = getSampleStyleSheet()

    story = []

    story.append(
        Paragraph("DataAudit Analysis Report", styles["Title"])
    )

    story.append(Spacer(1, 12))

    story.append(Paragraph(
        "Generated: " + datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        styles["Normal"]
    ))

    story.append(Spacer(1, 15))

    story.append(Paragraph(
        "Dataset Information",
        styles["Heading2"]
    ))

    info = [
        ["Property", "Value"],
        ["Dataset", str(data.get("dataset_name", "Unknown"))],
        ["Modality", str(data.get("modality", "Unknown"))],
        ["Samples", str(data.get("sample_count", "N/A"))]
    ]

    table = Table(info, colWidths=[150, 300])

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 7)
    ]))

    story.append(table)

    story.append(Spacer(1, 15))

    story.append(Paragraph(
        "Quality Assessment",
        styles["Heading2"]
    ))

    quality = data.get("quality_metrics", {})

    if quality:

        rows = [["Metric", "Value"]]

        for key, value in quality.items():
            rows.append([str(key), str(value)])

        table = Table(rows, colWidths=[200, 250])

        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("PADDING", (0, 0), (-1, -1), 6)
        ]))

        story.append(table)

    else:
        story.append(Paragraph(
            "No quality metrics available.",
            styles["Normal"]
        ))

    evaluation = data.get("evaluation")

    if evaluation:

        story.append(Spacer(1, 15))

        story.append(Paragraph(
            "Model Evaluation",
            styles["Heading2"]
        ))

        rows = [["Metric", "Score"]]

        for key in [
            "accuracy", "precision", "recall", "f1_score"
        ]:
            rows.append([
                key.replace("_", " ").title(),
                f"{evaluation[key]:.4f}"
            ])

        table = Table(rows, colWidths=[200, 250])

        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("PADDING", (0, 0), (-1, -1), 6)
        ]))

        story.append(table)

        cm = evaluation["confusion_matrix"]
        labels = evaluation["labels"]

        fig, ax = plt.subplots(figsize=(5, 4))

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

        image_buffer = io.BytesIO()

        fig.savefig(
            image_buffer,
            format="png",
            dpi=150
        )

        plt.close(fig)

        image_buffer.seek(0)

        story.append(Spacer(1, 12))

        story.append(Paragraph(
            "Confusion Matrix",
            styles["Heading2"]
        ))

        story.append(PDFImage(
            image_buffer,
            width=4 * inch,
            height=3 * inch
        ))

    story.append(Spacer(1, 15))

    story.append(Paragraph(
        "Note: Results depend on the dataset, selected labels, "
        "preprocessing and evaluation configuration.",
        styles["Italic"]
    ))

    document.build(story)

    buffer.seek(0)

    return buffer
