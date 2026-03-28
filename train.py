import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# LOAD DATA
df1 = pd.read_excel("Data/Dataset.xlsx")
df2 = pd.read_csv("Data/Bacteria_dataset_Multiresictance.csv")

# CLEAN SECONDARY DATA
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

# COMBINE
df = pd.concat([df1, df2], ignore_index=True)

# CLEAN
df["Location"] = df["Location"].astype("category").cat.codes

features = ["Location", "IMIPENEM", "CEFTAZIDIME", "GENTAMICIN", "AUGMENTIN", "CIPROFLOXACIN"]

for col in features:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.fillna(20)

# CONVERT TO CLASSES
def convert(value):
    if value < 15:
        return 2
    elif value <= 20:
        return 1
    else:
        return 0

for col in features:
    df[col] = df[col].apply(convert)

# ✅ 6 FEATURES INPUT
X = df[features]

# ✅ TARGET (same for now)
y = df["CIPROFLOXACIN"]

# TRAIN
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# EVALUATE
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# SAVE
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved!")