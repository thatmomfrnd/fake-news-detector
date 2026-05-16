from transformers import DistilBertForSequenceClassification
from transformers import DistilBertTokenizer
import torch
import torch.nn.functional as F

model=DistilBertForSequenceClassification.from_pretrained("../models/fake_news_model")
tokenizer=DistilBertTokenizer.from_pretrained("../models/fake_news_model")

model.eval()

def predict(text):
    inputs=tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs=model(**inputs)

    logits=outputs.logits
    probs=F.softmax(logits, dim=1)
    
    predicted_class=torch.argmax(logits).item()
    confidence=probs[0][predicted_class].item()
    label= "REAL" if predicted_class==1 else "FAKE"

    return label, round(confidence*100, 2)

text="Breaking: Study shows vaccines cause autism worldwide"
print(predict(text))