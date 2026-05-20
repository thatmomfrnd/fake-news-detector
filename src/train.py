from preprocessing import load_data, prep_data
import torch
from transformers import DistilBertTokenizer
from transformers import DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments 

df=load_data("/content/driver/MyDrive/news.csv")
x_train, x_test, y_train, y_test=prep_data(df)
x_train=x_train[:10000]
y_train=y_train[:10000]
x_test=x_test[:2000]
y_test=y_test[:2000]

tokenizer=DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
train_encodings=tokenizer(
    x_train,
    truncation=True,
    padding=True,
    max_length=128
)

test_encodings=tokenizer(
    x_test,
    truncation=True,
    padding=True,
    max_length=128
)

class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings=encodings
        self.labels=labels

    def __getitem__(self, idx):
        item={}
        for key, val in self.encodings.items():
            item[key]=torch.tensor(val[idx])
        item["labels"]=torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    
train_dataset=NewsDataset(train_encodings, y_train)
test_dataset=NewsDataset(test_encodings, y_test)

from collections import Counter
print(Counter(y_train))

model=DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased",num_labels=2)

counts = Counter(y_train)
total = sum(counts.values())

weight_0 = total / counts[0]
weight_1 = total / counts[1]

class_weights = torch.tensor([weight_0, weight_1]).to("cuda")

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }

training_args=TrainingArguments(
    output_dir="../models/results",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    load_best_model_at_end=True,
)

from torch.nn import CrossEntropyLoss

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss

trainer=CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

trainer.train() 

logs = trainer.state.log_history

train_loss = []
train_epochs = []

eval_loss = []
eval_epochs = []

for log in logs:
    if "loss" in log and "epoch" in log:
        train_loss.append(log["loss"])
        train_epochs.append(log["epoch"])

    if "eval_loss" in log and "epoch" in log:
        eval_loss.append(log["eval_loss"])
        eval_epochs.append(log["epoch"])

   import matplotlib.pyplot as plt

plt.plot(train_epochs, train_loss, label="Train Loss")
plt.plot(eval_epochs, eval_loss, label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

eval_acc = [log["eval_accuracy"] for log in logs if "eval_accuracy" in log]

plt.plot(eval_acc)
plt.title("Validation Accuracy")
plt.show()


model.save_pretrained("../models/fake_news_model")
tokenizer.save_pretrained("../models/fake_news_model")