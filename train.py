import re
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score

LABELS = ['admiration','anger','disgust','fear','hope','joy','love','pride','sadness']

def clean(t):
    t = str(t).lower()
    t = re.sub(r"http\S+|www\S+", "", t)
    t = re.sub(r"@\w+", "", t)
    t = re.sub(r"#", "", t)
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\d+", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


df = pd.read_csv(r"C:\Users\user\Desktop\DevDay\dataset.csv")

texts = df["Tweets (text)"].apply(clean)

labels_raw = df["Emotions (Multi-labeled)"].apply(
    lambda x: [i.strip() for i in str(x).split(",")]
)

mlb = MultiLabelBinarizer(classes=LABELS)
Y = mlb.fit_transform(labels_raw)

X_train, X_val, y_train, y_val = train_test_split(
    texts, Y, test_size=0.2, random_state=42
)

# 🔥 SIZE OPTIMIZED FEATURES (IMPORTANT CHANGE)
vectorizer = FeatureUnion([
    ("word", TfidfVectorizer(
        ngram_range=(1,2),
        max_features=25000,   # reduced
        min_df=2,
        sublinear_tf=True
    )),
    ("char", TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3,4),
        max_features=10000   # reduced
    ))
])

X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)

# 🔥 SIMPLE & STRONG MODEL (NO CALIBRATION)
model = OneVsRestClassifier(
    LogisticRegression(
        max_iter=2000,
        C=1.0,
        class_weight="balanced",
        solver="lbfgs"
    )
)

print("Training...")
model.fit(X_train_vec, y_train)

probs = model.predict_proba(X_val_vec)

# thresholds
thresholds = np.zeros(9)

for i in range(9):
    best_f1 = 0
    best_t = 0.3

    for t in np.linspace(0.2, 0.6, 30):
        pred = (probs[:, i] >= t).astype(int)
        f1 = f1_score(y_val[:, i], pred, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    thresholds[i] = best_t

final = np.zeros_like(probs)

for i in range(9):
    final[:, i] = (probs[:, i] >= thresholds[i]).astype(int)

def enforce(preds, probs):
    for i in range(len(preds)):
        c = preds[i].sum()

        if c < 2:
            top = np.argsort(probs[i])[-2:]
            preds[i][top] = 1

        elif c > 5:
            top = np.argsort(probs[i])[-5:]
            new = np.zeros_like(preds[i])
            new[top] = 1
            preds[i] = new

    return preds

final = enforce(final, probs)

print("MICRO F1:", f1_score(y_val, final, average="micro"))

# 🔥 SMALL PKL SAVE (IMPORTANT FIX)
joblib.dump({
    "vectorizer": vectorizer,
    "classifier": model,
    "thresholds": thresholds,
    "labels": LABELS
}, "MH.pkl", compress=3)

print("MODEL SAVED")