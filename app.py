import pickle
import gradio as gr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import os

# =========================
# LOAD DATASET (for UI)
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

features = ["Location", "IMIPENEM", "CEFTAZIDIME", "GENTAMICIN", "AUGMENTIN", "CIPROFLOXACIN"]

# =========================
# AUTO-FILL
# =========================
def autofill(location):
    if df is not None:
        row = df[df["Location"] == location]
        if not row.empty:
            row = row.iloc[0]
            return (
                float(row.get("IMIPENEM", 20)),
                float(row.get("CEFTAZIDIME", 20)),
                float(row.get("GENTAMICIN", 20)),
                float(row.get("AUGMENTIN", 20)),
                float(row.get("CIPROFLOXACIN", 20))
            )
    return 20, 20, 20, 20, 20

# =========================
# PREDICT + GRAPHS
# =========================
def predict(location, imipenem, ceftazidime, gentamicin, augmentin, ciprofloxacin):

    location_map = {loc: i for i, loc in enumerate(locations)}
    loc = location_map.get(location, 0)

    #  6 FEATURES FIX
    data = np.array([[loc, imipenem, ceftazidime, gentamicin, augmentin, ciprofloxacin]])

    pred = model.predict(data)[0]
    probs = model.predict_proba(data)[0]
    confidence = round(max(probs) * 100, 2)

    # Result mapping
    if pred == 0:
        result = "🟢 Susceptible"
        recommendation = "Use this antibiotic"
    elif pred == 1:
        result = "🟡 Intermediate"
        recommendation = "Use with caution"
    else:
        result = "🔴 Resistant"
        recommendation = "Avoid this antibiotic"

    #  BETTER OUTPUT
    result_text = f"""
 Antibiotic Resistance Analysis

Prediction (Target Antibiotic): CIPROFLOXACIN

Result: {result}
Confidence: {confidence}%

Input Values:
IMIPENEM: {imipenem}
CEFTAZIDIME: {ceftazidime}
GENTAMICIN: {gentamicin}
AUGMENTIN: {augmentin}
CIPROFLOXACIN: {ciprofloxacin}

 Recommendation:
{recommendation}
"""

    # =========================
    #  IMPROVED GRAPH (CLEAR LABELS)
    # =========================
    labels = ["IMIPENEM", "CEFTAZIDIME", "GENTAMICIN", "AUGMENTIN", "CIPROFLOXACIN"]
    values = [imipenem, ceftazidime, gentamicin, augmentin, ciprofloxacin]

    fig1, ax1 = plt.subplots(figsize=(7,4)) 
    palette = sns.color_palette("crest", len(labels))
    bars = ax1.barh(labels, values, color=palette)
    ax1.set_title("Antibiotic Values")
    ax1.set_xlabel("Value")

    #  SHOW VALUES ON BARS
    for bar in bars:
        width = bar.get_width()
        ax1.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                 f'{width}', va='center')

    #  FIX LABEL CUTTING
    plt.subplots_adjust(left=0.35)

    # =========================
    #  IMPROVED NETWORK GRAPH
    # =========================
    G = nx.Graph()
    antibiotics = labels

    for i in range(len(antibiotics)):
        for j in range(i + 1, len(antibiotics)):
            G.add_edge(antibiotics[i], antibiotics[j])

    fig2, ax2 = plt.subplots(figsize=(6, 6))

    pos = nx.spring_layout(G, seed=42)  # better layout

    nx.draw(
        G, pos,
        with_labels=True,
        node_size=2500,
        node_color="skyblue",
        font_size=10,
        ax=ax2
    )

    ax2.set_title("Antibiotic Relationship Network")

    return result_text, fig1, fig2

# =========================
# UI DESIGN
# =========================
with gr.Blocks(theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
#  AI-Based Antibiotic Resistance Prediction System
### Predict antibiotic effectiveness using AI
""")

    with gr.Row():

        with gr.Column():

            gr.Markdown("###  Select Location")

            location = gr.Dropdown(
                choices=locations,
                label="Location",
                interactive=True
            )

            gr.Markdown("###  Input Antibiotics")

            imipenem = gr.Slider(0, 40, value=20, label="IMIPENEM")
            ceftazidime = gr.Slider(0, 40, value=20, label="CEFTAZIDIME")
            gentamicin = gr.Slider(0, 40, value=20, label="GENTAMICIN")
            augmentin = gr.Slider(0, 40, value=20, label="AUGMENTIN")
            ciprofloxacin = gr.Slider(0, 40, value=20, label="CIPROFLOXACIN")  # ✅ NEW

            with gr.Row():
                btn = gr.Button("Predict", variant="primary")
                clear = gr.Button("Clear")

        with gr.Column():

            gr.Markdown("###  Prediction Output")

            output = gr.Textbox(label="Result", lines=10)

            plot1 = gr.Plot(label="Antibiotic Values")
            plot2 = gr.Plot(label="Resistance Network")

    # =========================
    # EVENTS
    # =========================
    location.change(
        autofill,
        inputs=location,
        outputs=[imipenem, ceftazidime, gentamicin, augmentin, ciprofloxacin]
    )

    btn.click(
        predict,
        inputs=[location, imipenem, ceftazidime, gentamicin, augmentin, ciprofloxacin],
        outputs=[output, plot1, plot2]
    )

    clear.click(
        lambda: (20, 20, 20, 20, 20, "", None, None),
        outputs=[imipenem, ceftazidime, gentamicin, augmentin, ciprofloxacin, output, plot1, plot2]
    )

# =========================
# RUN
# =========================
port = int(os.environ.get("PORT", 10000))
demo.launch(server_name="0.0.0.0", server_port=port)
