from preprocessing import load_data, prep_data
import torch
from transformers import DistilBertTokenizer
from transformers import DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments 

df=load_data("../data/news.csv")
x_train, x_test, y_train, y_test=prep_data(df)
x_train=x_train[:5000]
y_train=y_train[:5000]
x_test=x_test[:1000]
y_test=y_test[:1000]

tokenizer=DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
train_encodings=tokenizer(
    x_train,
    truncation=True,
    padding=True,
    max_length=512
)

test_encodings=tokenizer(
    x_test,
    truncation=True,
    padding=True,
    max_length=512
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
training_args=TrainingArguments(
    output_dir="../models/results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=8,
    logging_dir="../models/logs",
)
trainer=Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train() 
model.save_pretrained("../models/fake_news_model")
tokenizer.save_pretrained("../models/fake_news_model")