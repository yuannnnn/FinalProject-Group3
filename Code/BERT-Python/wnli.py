# -*- coding: utf-8 -*-
#%%
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from transformers import BertModel
from transformers import AutoModelForSequenceClassification
from transformers import AdamW
from transformers import get_scheduler
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from datasets import load_metric

GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]

num_epochs = 30
task = "wnli"
checkpoint = "bert-base-uncased"

actual_task = "mnli" if task == "mnli-mm" else task
dataset = load_dataset("glue", actual_task)
metric = load_metric('glue', actual_task)
tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}
sentence1_key, sentence2_key = task_to_keys[task]

def tokenize_function(example):
    if sentence2_key is None:
        return tokenizer(example[sentence1_key], truncation=True)
    return tokenizer(example[sentence1_key], example[sentence2_key], truncation=True)

num_labels = 3 if task.startswith("mnli") else 1 if task=="stsb" else 2
tokenized_datasets = dataset.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")
tokenized_datasets["train"].column_names

train_dataloader = DataLoader(tokenized_datasets["train"],
                              shuffle=True, batch_size=4, collate_fn=data_collator)
eval_dataloader = DataLoader(tokenized_datasets["validation"],
                             batch_size=4, collate_fn=data_collator)

for batch in train_dataloader:
    break
{k: v.shape for k, v in batch.items()}
#%%
class CustomBERTModel(nn.Module):
    def __init__(self, num_labels):
          super(CustomBERTModel, self).__init__()
          self.bert = BertModel.from_pretrained("bert-base-uncased")
          self.hidden_size = self.bert.config.hidden_size
          self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True,bidirectional=True)
          self.clf = nn.Linear(self.hidden_size*2, num_labels)

    def forward(self, **batch):
          sequence_output, pooled_output = self.bert(batch['input_ids'],batch['attention_mask'])[:2]
          # sequence_output has the following shape: (batch_size, sequence_length, self.hidden_size)
          lstm_output, (h,c) = self.lstm(sequence_output) ## extract the 1st token's embeddings
          hidden = torch.cat((lstm_output[:,-1, :self.hidden_size],lstm_output[:,0, self.hidden_size:]),dim=-1)
          hidden = F.dropout(hidden,0.1)
          linear_output = self.clf(hidden.view(-1,self.hidden_size*2)) ### only using the output of the last LSTM cell to perform classification
          return linear_output

model = CustomBERTModel(num_labels)

model

optimizer = AdamW(model.parameters(), lr=5e-5)
loss_fct = nn.CrossEntropyLoss()

num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler( "linear", optimizer=optimizer,
                              num_warmup_steps=0, num_training_steps=num_training_steps)
print(num_training_steps)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
print(device)
#%%
progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = loss_fct(outputs, batch['labels'])
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
# 100%|███████████████████████████████████████| 4770/4770 [04:25<00:00, 17.83it/s]
#%%
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
    if actual_task != "stsb":
        predictions = torch.argmax(outputs, dim=-1)
    else:
        predictions = outputs[:, 0]
    metric.add_batch(predictions=predictions, references=batch["labels"])
#%%
metric.compute()
# {'accuracy': 0.5492957746478874}
#%%
"""### Codes borrowed and adjusted from

1.   [NLP lecture codes](https://github.com/amir-jafari/NLP/blob/master/Lecture_09/Lecture%20Code/12-training.py)
2.   [Hugging Face transformers notebooks](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/text_classification.ipynb#scrollTo=7k8ge1L1IrJk)
3.   [Ashwin Geet D'Sa's answer on Stackoverflow](https://stackoverflow.com/questions/65205582/how-can-i-add-a-bi-lstm-layer-on-top-of-bert-model?rq=1)


"""