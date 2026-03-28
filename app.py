import pickle
import gradio as gr
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # Fix for Render

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

# =========================
# AUTO-FILL FUNCTION
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
# PREDICTION FUNCTION
# =========================
def predict(location, imipenem, ceftazidime, gentamicin, augmentin, ciprofloxacin):
    try:
        location_map = {loc: i for i, loc in enumerate(locations)}
        loc = location_map.get(location, 0)

        #  6 FEATURES
        data = np.array([[loc, imipenem, ceftazidime, gentamicin, augmentin, ciprofloxacin]])

        pred = model.predict(data)[0]
        probs = model.predict_proba(data)[0]
        confidence = round(max(probs) * 100, 2)

        if pred == 0:
            result = "🟢 Susceptible"
            recommendation = "Use this antibiotic"
        elif pred == 1:
            result = "🟡 Intermediate"
            recommendation = "Use with caution"
        else:
            result = "🔴 Resistant"
            recommendation = "Avoid this antibiotic"

        result_text = f"""
🔬 Prediction Result

Result: {result}
Confidence: {confidence}%

Recommendation:
{recommendation}
"""

        # =========================
        #  BAR GRAPH (CLEAR NAMES)
        # =========================
        antibiotics = ["IMIPENEM", "CEFTAZIDIME", "GENTAMICIN", "AUGMENTIN", "CIPROFLOXACIN"]
        values = [imipenem, ceftazidime, gentamicin, augmentin, ciprofloxacin]

        fig1, ax1 = plt.subplots(figsize=(8, 5))
        ax1.barh(antibiotics, values)
        ax1.set_title("Antibiotic Resistance Levels")
        ax1.set_xlabel("Value")
        plt.tight_layout()

        # =========================
        #  NETWORK GRAPH (CLEAN)
        # =========================
        G = nx.Graph()

        for ab in antibiotics:
            G.add_node(ab)

        for i in range(len(antibiotics)):
            for j in range(i + 1, len(antibiotics)):
                G.add_edge(antibiotics[i], antibiotics[j])

        fig2, ax2 = plt.subplots(figsize=(6, 6))
        nx.draw(G, with_labels=True, node_size=2000, font_size=10, ax=ax2)
        ax2.set_title("Resistance Network")

        return result_text, fig1, fig2

    except Exception as e:
        return f"Error: {str(e)}", None, None

# =========================
# UI DESIGN
# =========================
with gr.Blocks(theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
#  AI-Based Antibiotic Resistance Prediction System
### Predict antibiotic effectiveness using AI
""")

    with gr.Row():

        # LEFT SIDE INPUT
        with gr.Column():

            location = gr.Dropdown(choices=locations, label="Location")

            imipenem = gr.Slider(0, 40, value=20, label="IMIPENEM")
            ceftazidime = gr.Slider(0, 40, value=20, label="CEFTAZIDIME")
            gentamicin = gr.Slider(0, 40, value=20, label="GENTAMICIN")
            augmentin = gr.Slider(0, 40, value=20, label="AUGMENTIN")
            ciprofloxacin = gr.Slider(0, 40, value=20, label="CIPROFLOXACIN")

            with gr.Row():
                btn = gr.Button("Predict")
                clear = gr.Button("Clear")

        # RIGHT SIDE OUTPUT
        with gr.Column():

            output = gr.Textbox(label="Result", lines=8)
            plot1 = gr.Plot(label="Resistance Graph")
            plot2 = gr.Plot(label="Network Graph")

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
# RUN (RENDER FIX)
# =========================
port = int(os.environ.get("PORT", 10000))
demo.launch(server_name="0.0.0.0", server_port=port)

