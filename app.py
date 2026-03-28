import pickle
import gradio as gr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import os

# =========================
# LOAD DATASET (SAFE)
# =========================
try:
    df = pd.read_excel("Dataset.xlsx")
    locations = df["Location"].dropna().unique().tolist()
except:
    df = None
    locations = ["IFE-T"]

# =========================
# LOAD MODEL (SAFE)
# =========================
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
except:
    model = None

features = ["Location", "IMIPENEM", "CEFTAZIDIME", "GENTAMICIN", "AUGMENTIN"]

# =========================
# AUTO-FILL FUNCTION
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
                float(row.get("AUGMENTIN", 20))
            )
    return 20, 20, 20, 20

# =========================
# PREDICT FUNCTION
# =========================
def predict(location, imipenem, ceftazidime, gentamicin, augmentin):

    if model is None:
        return " Model not loaded", None, None

    # Encode location
    location_map = {loc: i for i, loc in enumerate(locations)}
    loc = location_map.get(location, 0)

    data = np.array([[loc, imipenem, ceftazidime, gentamicin, augmentin]])

    # Prediction
    pred = model.predict(data)[0]

    # Confidence
    try:
        probs = model.predict_proba(data)
        probs = probs[0][0] if isinstance(probs, list) else probs[0]
        confidence = round(max(probs) * 100, 2)
    except:
        confidence = 0

    # Result
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
 Prediction for CIPROFLOXACIN

Result: {result}
Confidence: {confidence}%

 Recommendation:
{recommendation}
"""

    # =========================
    #  RESISTANCE GRAPH (FIXED LABEL ISSUE)
    # =========================
    labels = ["CIPROFLOXACIN", "AUGMENTIN", "GENTAMICIN", "CEFTAZIDIME", "IMIPENEM"]
    values = [pred, augmentin/20, gentamicin/20, ceftazidime/20, imipenem/20]

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.barh(labels, values)

    ax1.set_title("Resistance Levels (0=Safe, 2=Resistant)")
    ax1.set_xlabel("Resistance Score")

    # FIX CUT LABELS
    plt.subplots_adjust(left=0.35)

    # =========================
    #  NETWORK GRAPH
    # =========================
    G = nx.Graph()
    antibiotics = ["IMIPENEM", "CEFTAZIDIME", "GENTAMICIN", "AUGMENTIN", "CIPROFLOXACIN"]

    for i in range(len(antibiotics)):
        for j in range(i + 1, len(antibiotics)):
            G.add_edge(antibiotics[i], antibiotics[j])

    fig2, ax2 = plt.subplots()
    nx.draw(G, with_labels=True, node_size=2000, ax=ax2)
    ax2.set_title("Antibiotic Network")

    return result_text, fig1, fig2

# =========================
# UI
# =========================
with gr.Blocks(theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
# ResistAI – Antibiotic Resistance Predictor  
### Smarter Antibiotics. Better Decisions.
""")

    with gr.Row():

        with gr.Column():

            location = gr.Dropdown(choices=locations, label="Location")

            imipenem = gr.Slider(0, 40, value=20, label="IMIPENEM")
            ceftazidime = gr.Slider(0, 40, value=20, label="CEFTAZIDIME")
            gentamicin = gr.Slider(0, 40, value=20, label="GENTAMICIN")
            augmentin = gr.Slider(0, 40, value=20, label="AUGMENTIN")

            with gr.Row():
                btn = gr.Button("Predict", variant="primary")
                clear = gr.Button("Clear")

        with gr.Column():

            output = gr.Textbox(label="Result", lines=8)
            plot1 = gr.Plot(label="Resistance Graph")
            plot2 = gr.Plot(label="Network Graph")

    # EVENTS
    location.change(
        autofill,
        inputs=location,
        outputs=[imipenem, ceftazidime, gentamicin, augmentin]
    )

    btn.click(
        predict,
        inputs=[location, imipenem, ceftazidime, gentamicin, augmentin],
        outputs=[output, plot1, plot2]
    )

    clear.click(
        lambda: (20, 20, 20, 20, "", None, None),
        outputs=[imipenem, ceftazidime, gentamicin, augmentin, output, plot1, plot2]
    )

# =========================
# RUN (RENDER SAFE)
# =========================
port = int(os.environ.get("PORT", 10000))
demo.launch(server_name="0.0.0.0", server_port=port)
