import pickle
import gradio as gr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

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
# PREDICT FUNCTION
# =========================
def predict(location, imipenem, ceftazidime, gentamicin, augmentin, ciprofloxacin):

    location_map = {loc: i for i, loc in enumerate(locations)}
    loc = location_map.get(location, 0)

    # 6 features
    data = np.array([[loc, imipenem, ceftazidime, gentamicin, augmentin, ciprofloxacin]])

    pred = model.predict(data)[0]
    probs = model.predict_proba(data)[0]
    confidence = round(max(probs) * 100, 2)

    if pred == 0:
        result = "Susceptible"
        recommendation = "Use antibiotic"
    elif pred == 1:
        result = "Intermediate"
        recommendation = "Use with caution"
    else:
        result = "Resistant"
        recommendation = "Avoid antibiotic"

    result_text = f"""
Antibiotic Resistance Result

Prediction: {result}
Confidence: {confidence}%

Recommendation: {recommendation}
"""

    # =========================
    # GRAPH 1 (FIXED LABELS)
    # =========================
    labels = ["IMIPENEM", "CEFTAZIDIME", "GENTAMICIN", "AUGMENTIN", "CIPROFLOXACIN"]
    values = [imipenem, ceftazidime, gentamicin, augmentin, ciprofloxacin]

    fig1, ax1 = plt.subplots(figsize=(12, 6))
    bars = ax1.barh(labels, values)

    for bar in bars:
        width = bar.get_width()
        ax1.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                 f"{width}", va='center')

    ax1.set_title("Resistance Levels")
    ax1.set_xlabel("Value")

    ax1.tick_params(axis='y', labelsize=12)
    plt.subplots_adjust(left=0.35)
    plt.tight_layout()

    # =========================
    # GRAPH 2 (NETWORK)
    # =========================
    G = nx.Graph()

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            G.add_edge(labels[i], labels[j])

    fig2, ax2 = plt.subplots(figsize=(6, 6))
    pos = nx.spring_layout(G, seed=42)

    nx.draw(
        G, pos,
        with_labels=True,
        node_size=2500,
        node_color="lightblue",
        font_size=10,
        ax=ax2
    )

    ax2.set_title("Antibiotic Network")

    return result_text, fig1, fig2

# =========================
# UI
# =========================
with gr.Blocks() as demo:

    gr.Markdown("# Antibiotic Resistance Predictor")

    location = gr.Dropdown(choices=locations, label="Location")

    imipenem = gr.Slider(0, 40, value=20, label="IMIPENEM")
    ceftazidime = gr.Slider(0, 40, value=20, label="CEFTAZIDIME")
    gentamicin = gr.Slider(0, 40, value=20, label="GENTAMICIN")
    augmentin = gr.Slider(0, 40, value=20, label="AUGMENTIN")
    ciprofloxacin = gr.Slider(0, 40, value=20, label="CIPROFLOXACIN")

    btn = gr.Button("Predict")
    clear = gr.Button("Clear")

    output = gr.Textbox(lines=8)
    plot1 = gr.Plot()
    plot2 = gr.Plot()

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
import os
port = int(os.environ.get("PORT", 10000))
demo.launch(server_name="0.0.0.0", server_port=port)