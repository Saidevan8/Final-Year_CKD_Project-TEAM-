import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
from PIL import Image
import os

# Paths for inputs and outputs
metrics_path = "B:/CKC/data/evaluation_metrics.csv"
confusion_matrices_dir = "B:/CKC/evolution"
report_output_path = "B:/CKC/data/model_performance_report.pdf"

# PDF report generation function
def generate_pdf_report():
    # Initialize the PDF
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=10)
    
    # Load evaluation metrics
    metrics_df = pd.read_csv(metrics_path)

    # Cover page
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Model Performance Report", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(10)

    # Section for metrics
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "Evaluation Metrics", new_x="LMARGIN", new_y="NEXT", align="L")
    pdf.set_font("Helvetica", "", 10)
    pdf.ln(5)

    # Insert evaluation metrics as a table
    col_width = pdf.w / (len(metrics_df.columns) + 1)
    row_height = pdf.font_size * 1.5

    # Header
    for column in metrics_df.columns:
        pdf.cell(col_width, row_height, str(column), border=1)
    pdf.ln(row_height)

    # Rows
    for _, row in metrics_df.iterrows():
        for item in row:
            pdf.cell(col_width, row_height, str(item), border=1)
        pdf.ln(row_height)

    pdf.ln(10)

    # Section for Confusion Matrices
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "Confusion Matrices", new_x="LMARGIN", new_y="NEXT", align="L")
    pdf.set_font("Helvetica", "", 10)
    pdf.ln(5)

    # Insert each confusion matrix image
    for model_name in metrics_df["Model"]:
        image_path = os.path.join(confusion_matrices_dir, f"{model_name}_confusion_matrix.png")
        if os.path.exists(image_path):
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 10, f"{model_name} Confusion Matrix", new_x="LMARGIN", new_y="NEXT", align="C")
            pdf.ln(10)

            # Add the confusion matrix image
            pdf.image(image_path, x=10, w=180)

    # Save the PDF report
    pdf.output(report_output_path)
    print("Model performance report generated and saved successfully.")

if __name__ == "__main__":
    generate_pdf_report()
