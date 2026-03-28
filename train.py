import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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

# Add missing columns
df2["CEFTAZIDIME"] = 20
df2["AUGMENTIN"] = 20
df2["Location"] = "External"

# Match columns with df1
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
# STEP 6: FEATURES & TARGET
# =========================
X = df[["Location"] + antibiotics]   # 6 features
y = df[antibiotics]                 # multi-output targets

# =========================
# STEP 7: TRAIN MODEL
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# =========================
# STEP 8: SAVE MODEL
# =========================
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved successfully!")