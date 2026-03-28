import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pickle

# =========================
# STEP 1: LOAD DATA
# =========================
df1 = pd.read_excel("Data/Dataset.xlsx")
df2 = pd.read_csv("Data/Bacteria_dataset_Multiresictance.csv")

# =========================
# STEP 2: CLEAN SECOND DATASET
# =========================
df2 = df2[["IPM", "GEN", "CIP"]]

df2 = df2.rename(columns={
    "IPM": "IMIPENEM",
    "GEN": "GENTAMICIN",
    "CIP": "CIPROFLOXACIN"
})

df2["CEFTAZIDIME"] = 20
df2["AUGMENTIN"] = 20
df2["Location"] = "External"

df2 = df2[df1.columns]

# =========================
# STEP 3: COMBINE DATA
# =========================
df = pd.concat([df1, df2], ignore_index=True)

print("Combined dataset:", df.shape)

# =========================
# STEP 4: CLEAN DATA
# =========================
df["Location"] = df["Location"].astype("category").cat.codes

antibiotics = [
    "IMIPENEM",
    "CEFTAZIDIME",
    "GENTAMICIN",
    "AUGMENTIN",
    "CIPROFLOXACIN"
]

for col in antibiotics:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.fillna(20)

# =========================
# STEP 5: CONVERT TO CLASSES
# =========================
def convert(value):
    if value < 15:
        return 2   # Resistant
    elif value <= 20:
        return 1   # Intermediate
    else:
        return 0   # Susceptible

for col in antibiotics:
    df[col] = df[col].apply(convert)

# =========================
# STEP 6: FEATURES & TARGET (MULTI OUTPUT)
# =========================
X = df[["Location"] + antibiotics]
y = df[antibiotics]

# =========================
# STEP 7: TRAIN MODEL
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# =========================
# STEP 8: EVALUATE MODEL
# =========================
y_pred = model.predict(X_test)

# Accuracy (average across all antibiotics)
accuracy = np.mean(y_pred == y_test.values)
print("Model Accuracy:", accuracy)

# =========================
# STEP 9: CONFUSION MATRIX GRAPH
# =========================
# Flatten multi-output into single array
y_test_flat = y_test.values.flatten()
y_pred_flat = y_pred.flatten()

cm = confusion_matrix(y_test_flat, y_pred_flat)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

plt.title("Model Accuracy (Confusion Matrix)")
plt.savefig("accuracy_graph.png")  # ✅ IMPORTANT FILE

# =========================
# STEP 10: SAVE MODEL
# =========================
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved!")
print("Accuracy graph saved as accuracy_graph.png")
