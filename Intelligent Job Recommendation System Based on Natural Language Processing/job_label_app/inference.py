#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import json
import joblib
from transformers import AutoTokenizer
from model_arch import BertForMultiLabelClassification

def load_inference_components(model_path, tokenizer_path, mlb_path, device="cpu"):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    mlb = joblib.load(mlb_path)

    model = BertForMultiLabelClassification(
        model_name="bert-base-multilingual-cased",
        num_labels=len(mlb.classes_)
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return tokenizer, model, mlb

def predict_single(text, tokenizer, model, mlb, device="cpu", threshold=0.5):
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=128
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        logits = model(**inputs)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
        preds = (probs > threshold).astype(int)

    labels = mlb.classes_[preds == 1].tolist()
    label_probs = {label: float(prob) for label, prob in zip(mlb.classes_, probs)}
    return labels, label_probs

def save_prediction(text, labels, label_probs, path="prediction_output.json"):
    result = {
        "input": text,
        "predicted_labels": labels,
        "label_probabilities": label_probs
    }
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Prediction saved to {path}")

if __name__ == "__main__":
    example_text = "5 years of experience in sales and marketing, fluent in English and Chinese."
    tokenizer, model, mlb = load_inference_components(
        model_path="saved_model/bert_model.pt",
        tokenizer_path="saved_model/tokenizer_dir",
        mlb_path="saved_model/mlb.pkl"
    )
    labels, label_probs = predict_single(example_text, tokenizer, model, mlb)
    print("Predicted Labels:", labels)
    save_prediction(example_text, labels, label_probs)

