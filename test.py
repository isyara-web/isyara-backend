import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt

def get_embedding(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()  # Mengambil representasi dari [CLS] token

def evaluate_model(true_texts, predicted_texts, model_name="bert-base-multilingual-cased"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    true_embeddings = np.array([get_embedding(text, model, tokenizer) for text in true_texts])
    predicted_embeddings = np.array([get_embedding(text, model, tokenizer) for text in predicted_texts])
    
    similarities = [cosine_similarity(true_embeddings[i], predicted_embeddings[i])[0][0] for i in range(len(true_texts))]
    accuracy = np.mean(np.array(similarities) > 0.8)  # Ambang batas kesamaan
    
    print(f"Accuracy: {accuracy * 100:.2f}%")
    
    # Confusion Matrix
    y_true = np.array(true_texts)
    y_pred = np.array(predicted_texts)
    
    labels = list(set(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    
    plt.figure(figsize=(10, 7))
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.show()
    
    # Classification Report
    print(classification_report(y_true, y_pred, target_names=labels))

# Contoh data uji
true_texts = ["Halo", "Terima Kasih", "Selamat Pagi"]
predicted_texts = ["Halo", "Terima kasih", "Selamat siang"]

evaluate_model(true_texts, predicted_texts)
