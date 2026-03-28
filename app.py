import pickle
import gradio as gr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import os

# PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# =========================
# LOAD DATASET
# =========================
try:
    df = pd.read_excel("Dataset.xlsx")
    locations = df["Location"].unique().tolist()
except:
    df = None
    locations = ["IFE-T"]

# =========================
# LOAD MODEL
# =========================
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# =========================
# PDF GENERATION
# =========================
def generate_pdf(best, good, medium, bad):
    doc = SimpleDocTemplate("report.pdf")
    styles = getSampleStyleSheet()

    content = []
    content.append(Paragraph("AI Antibiotic Recommendation Report", styles["Title"]))
    content.append(Paragraph(f"Best Antibiotic: {best}", styles["Normal"]))
    content.append(Paragraph(f"Safe: {good}", styles["Normal"]))
    content.append(Paragraph(f"Intermediate: {medium}", styles["Normal"]))
    content.append(Paragraph(f"Resistant: {bad}", styles["Normal"]))

    doc.build(content)
    return "report.pdf"

# =========================
# AUTO-FILL
# =========================
def autofill(location):
    if df is not None:
        row = df[df["Location"] == location]
        if not row.empty:
            row = row.iloc[0]
            return (
                float(row["IMIPENEM"]),
                float(row["CEFTAZIDIME"]),
                float(row["GENTAMICIN"]),
                float(row["AUGMENTIN"]),
                float(row["CIPROFLOXACIN"])
            )
    return 20, 20, 20, 20, 20

# =========================
# PREDICT
# =========================
def predict(location, imipenem, ceftazidime, gentamicin, augmentin, ciprofloxacin):

    location_map = {loc: i for i, loc in enumerate(locations)}
    loc = location_map.get(location, 0)

    data = np.array([[loc, imipenem, ceftazidime, gentamicin, augmentin, ciprofloxacin]])

    preds = model.predict(data)[0]

    antibiotics = ["IMIPENEM", "CEFTAZIDIME", "GENTAMICIN", "AUGMENTIN", "CIPROFLOXACIN"]
    results = dict(zip(antibiotics, preds))

    best = min(results, key=results.get)

    good = [k for k, v in results.items() if v == 0]
    medium = [k for k, v in results.items() if v == 1]
    bad = [k for k, v in results.items() if v == 2]

    # =========================
    # OUTPUT TEXT
    # =========================
    result_text = f"""
    <h2> AI Recommendation System</h2>

    <h3>✅ Best Antibiotic: <span style='color:green'>{best}</span></h3>

    <h4>🟢 Safe:</h4>
    {good}

    <h4>🟡 Intermediate:</h4>
    {medium}

    <h4>🔴 Resistant:</h4>
    {bad}
    """

    # =========================
    # GRAPH 1
    # =========================
    fig1, ax1 = plt.subplots()
    ax1.barh(antibiotics, preds)
    ax1.set_title("Resistance Levels (0=Safe, 2=Resistant)")

    # =========================
    # GRAPH 2
    # =========================
    G = nx.Graph()
    for i in range(len(antibiotics)):
        for j in range(i + 1, len(antibiotics)):
            G.add_edge(antibiotics[i], antibiotics[j])

    fig2, ax2 = plt.subplots()
    nx.draw(G, with_labels=True, node_size=2000, ax=ax2)

    # =========================
    # PDF REPORT
    # =========================
    pdf_file = generate_pdf(best, good, medium, bad)

    return result_text, fig1, fig2, pdf_file

# =========================
# UI
# =========================
with gr.Blocks(theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
#  AI Antibiotic Recommendation System
### Smart prediction + recommendation + reporting
""")

    with gr.Row():

        with gr.Column():

            location = gr.Dropdown(choices=locations, label="Location")

            imipenem = gr.Slider(0, 40, value=20, label="IMIPENEM")
            ceftazidime = gr.Slider(0, 40, value=20, label="CEFTAZIDIME")
            gentamicin = gr.Slider(0, 40, value=20, label="GENTAMICIN")
            augmentin = gr.Slider(0, 40, value=20, label="AUGMENTIN")
            ciprofloxacin = gr.Slider(0, 40, value=20, label="CIPROFLOXACIN")

            with gr.Row():
                btn = gr.Button(" Predict", variant="primary")
                clear = gr.Button(" Clear")

        with gr.Column():

            output = gr.HTML(label="Result")

            plot1 = gr.Plot(label="Resistance Graph")
            plot2 = gr.Plot(label="Network Graph")

            pdf_output = gr.File(label="Download Report")

    # EVENTS
    location.change(
        autofill,
        inputs=location,
        outputs=[imipenem, ceftazidime, gentamicin, augmentin, ciprofloxacin]
    )

    btn.click(
        predict,
        inputs=[location, imipenem, ceftazidime, gentamicin, augmentin, ciprofloxacin],
        outputs=[output, plot1, plot2, pdf_output]
    )

    clear.click(
        lambda: (20, 20, 20, 20, 20, "", None, None, None),
        outputs=[imipenem, ceftazidime, gentamicin, augmentin, ciprofloxacin, output, plot1, plot2, pdf_output]
    )

# =========================
# RUN
# =========================
port = int(os.environ.get("PORT", 10000))
demo.launch(server_name="0.0.0.0", server_port=port)
