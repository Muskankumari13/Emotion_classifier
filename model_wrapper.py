import re
import joblib
import numpy as np

LABELS = ['admiration','anger','disgust','fear','hope','joy','love','pride','sadness']

def _preprocess(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class MyModel:
    def __init__(self):
        bundle = joblib.load("model.pkl")
        self.vectorizer = bundle["vectorizer"]
        self.lr = bundle["lr"]
        self.svm = bundle["svm"]
        self.nb = bundle["nb"]
        self.thresholds = np.array(bundle["thresholds"]).reshape(-1)

    def predict(self, texts):
        texts = [_preprocess(t) for t in texts]
        X = self.vectorizer.transform(texts)

        p1 = self.lr.predict_proba(X)
        p2 = self.svm.predict_proba(X)
        p3 = self.nb.predict_proba(X)

        probs = (0.4 * p1) + (0.4 * p2) + (0.2 * p3)
        probs = np.array(probs)

        preds = (probs >= self.thresholds).astype(int)

        for i in range(len(preds)):
            c = int(np.sum(preds[i]))

            if c < 2:
                top = np.argsort(probs[i])[-2:]
                preds[i][top] = 1

            elif c > 5:
                top = np.argsort(probs[i])[-5:]
                new = np.zeros(9, dtype=int)
                new[top] = 1
                preds[i] = new

        return preds.astype(np.int32)