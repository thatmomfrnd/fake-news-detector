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

print(predict("The Indian government announced a new policy to improve digital education infrastructure across rural areas."))
print(predict("Scientists confirm that humans can now breathe underwater without any equipment after a new discovery."))
print(predict("Experts warn that excessive social media use may be linked to declining attention spans in teenagers."))