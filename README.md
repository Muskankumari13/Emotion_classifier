```md
# Emotion_classifier

# Multi-Label Emotion Detection Model

A machine learning based NLP system for detecting multiple emotions from text/tweets using TF-IDF features and Logistic Regression.

---

## Features

- Multi-label emotion classification  
- Text preprocessing (cleaning, normalization)  
- TF-IDF word and character n-grams  
- Logistic Regression with One-vs-Rest strategy  
- Threshold tuning using F1 score  
- Lightweight model saved using Joblib  

---

## Supported Emotions

- admiration  
- anger  
- disgust  
- fear  
- hope  
- joy  
- love  
- pride  
- sadness  

---

## Project Structure

```

Emotion-Detection-Model/
│
├── train.py
├── model_wrapper.py
├── MH.pkl
├── dataset.csv
└── README.md

````

---

## Installation

```bash
git clone https://github.com/your-username/emotion-detection-model.git
cd emotion-detection-model
pip install -r requirements.txt
````

---

## Training

```bash
python train.py
```

This will:

* Train the model
* Optimize thresholds
* Save trained model as `MH.pkl`

---

## Inference

```python
from model_wrapper import MyModel

model = MyModel()

texts = [
    "I am very happy today",
    "This is terrible"
]

print(model.predict(texts))
```

---

## Model Info

* TF-IDF Word: (1,2) n-grams
* TF-IDF Char: (3,4) n-grams
* Classifier: Logistic Regression (One-vs-Rest)
* Metric: Micro F1 Score
* Label-wise threshold tuning

---

## Requirements

```txt
numpy
pandas
scikit-learn
joblib
```

`
