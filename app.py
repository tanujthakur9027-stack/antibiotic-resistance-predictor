import pickle
import gradio as gr
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import os

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
# LOAD MODEL (optional)
# =========================
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
except:
    model = None

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
# CLASSIFICATION LOGIC
# =========================
def classify(value):
    if value < 15:
        return "Resistant"
    elif value <= 20:
        return "Intermediate"
    else:
        return "Safe"

# =========================
# PREDICT FUNCTION
# =========================
def predict(location, imipenem, ceftazidime, gentamicin, augmentin, ciprofloxacin):
    try:
        data_dict = {
            "IMIPENEM": classify(imipenem),
            "CEFTAZIDIME": classify(ceftazidime),
            "GENTAMICIN": classify(gentamicin),
            "AUGMENTIN": classify(augmentin),
            "CIPROFLOXACIN": classify(ciprofloxacin)
        }

        safe, intermediate, resistant = [], [], []

        for k, v in data_dict.items():
            if v == "Safe":
                safe.append(k)
            elif v == "Intermediate":
                intermediate.append(k)
            else:
                resistant.append(k)

        # BEST ANTIBIOTIC
        if safe:
            best = safe[0]
        elif intermediate:
            best = intermediate[0]
        else:
            best = "None"

        # CLEAN TEXT
        safe_text = ", ".join(safe) if safe else "None"
        inter_text = ", ".join(intermediate) if intermediate else "None"
        res_text = ", ".join(resistant) if resistant else "None"

        result_text = f"""
 AI Recommendation System

 Best Antibiotic: {best}

🟢 Safe:
{safe_text}

🟡 Intermediate:
{inter_text}

🔴 Resistant:
{res_text}
"""

        # =========================
        # GRAPH (FIXED LABELS)
        # =========================
        antibiotics = list(data_dict.keys())
        values = [imipenem, ceftazidime, gentamicin, augmentin, ciprofloxacin]

        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.barh(antibiotics, values)
        ax1.set_title("Resistance Graph (0=Safe, 2=Resistant)")
        ax1.set_xlabel("Values")
        ax1.tick_params(axis='y', labelsize=10)
        plt.tight_layout()

        # =========================
        # NETWORK GRAPH (CLEAN)
        # =========================
        G = nx.Graph()

        for ab in antibiotics:
            G.add_node(ab)

        for i in range(len(antibiotics)):
            for j in range(i + 1, len(antibiotics)):
                G.add_edge(antibiotics[i], antibiotics[j])

        fig2, ax2 = plt.subplots(figsize=(7, 7))
        pos = nx.spring_layout(G, seed=42)
        nx.draw(
            G, pos,
            with_labels=True,
            node_size=2500,
            node_color="skyblue",
            font_size=9,
            ax=ax2
        )
        ax2.set_title("Resistance Network")

        return result_text, fig1, fig2

    except Exception as e:
        return f"Error: {str(e)}", None, None

# =========================
# UI
# =========================
with gr.Blocks(theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
#  AI-Based Antibiotic Resistance Prediction System
### Smart Antibiotic Recommendation
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
                btn = gr.Button("Predict")
                clear = gr.Button("Clear")

        with gr.Column():

            output = gr.Textbox(label="AI Recommendation", lines=10)
            plot1 = gr.Plot(label="Resistance Graph")
            plot2 = gr.Plot(label="Network Graph")

    # EVENTS
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
