from fpdf import FPDF

def generate_report(seq, prob):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Off-Target Probability: {prob:.3f}")
    pdf.output("report.pdf")
