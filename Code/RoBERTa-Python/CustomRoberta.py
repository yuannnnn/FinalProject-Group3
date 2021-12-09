#------------------------Imports----------------------------
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from transformers import RobertaModel, RobertaTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import AdamW
from transformers import get_scheduler
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from datasets import load_metric


GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]

num_epochs = 5
# Name of task
task = "cola"
checkpoint = "roberta-base"

actual_task = "mnli" if task == "mnli-mm" else task
dataset = load_dataset("glue", actual_task)
metric = load_metric('glue', actual_task)


tokenizer = AutoTokenizer.from_pretrained('roberta-base')

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

if task == "cola" or task == "sst2":
    tokenized_datasets = tokenized_datasets.remove_columns(["sentence", "idx"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

if task == "mnli" or task == "mnli-mm":
    tokenized_datasets = tokenized_datasets.remove_columns(["premise", "hypothesis", "idx"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

if task == "mrpc" or task == "stsb" or task == "rte" or task == "wnli":
    tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

if task == "qnli":
    tokenized_datasets = tokenized_datasets.remove_columns(["question", "sentence", "idx"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

if task == "qqp":
    tokenized_datasets = tokenized_datasets.remove_columns(["question1", "question2", "idx"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

tokenized_datasets["train"].column_names

if task == "mnli-mm":
    train_dataloader = DataLoader(tokenized_datasets["train"],
                              shuffle=True, batch_size=8, collate_fn=data_collator)
    eval_dataloader = DataLoader(tokenized_datasets["validation_matched"],
                             batch_size=8, collate_fn=data_collator)
else:
    train_dataloader = DataLoader(tokenized_datasets["train"],
                                  shuffle=True, batch_size=8, collate_fn=data_collator)
    eval_dataloader = DataLoader(tokenized_datasets["validation"],
                                 batch_size=8, collate_fn=data_collator)


for batch in train_dataloader:
    break
{k: v.shape for k, v in batch.items()}

class CustomROBERTAModel(nn.Module):
    def __init__(self, num_labels):
          super(CustomROBERTAModel, self).__init__()
          self.roberta = RobertaModel.from_pretrained("roberta-base")
          self.hidden_size = self.roberta.config.hidden_size
          self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True,bidirectional=True)
          self.clf = nn.Linear(self.hidden_size*2, num_labels)
          self.num_labels = num_labels

    def forward(self, **batch):
          sequence_output, pooled_output = self.roberta(batch['input_ids'],batch['attention_mask'])[:2]
          # sequence_output has the following shape: (batch_size, sequence_length, self.hidden_size)
          lstm_output, (h,c) = self.lstm(sequence_output) ## extract the 1st token's embeddings
          hidden = torch.cat((lstm_output[:,-1, :self.hidden_size],lstm_output[:,0, self.hidden_size:]),dim=-1)
          hidden = F.dropout(hidden,0.1)
          linear_output = self.clf(hidden.view(-1,self.hidden_size*2)) ### only using the output of the last LSTM cell to perform classification
          return linear_output

model = CustomROBERTAModel(num_labels)
optimizer = AdamW(model.parameters(), lr=5e-5)
if task== "stsb":
    loss_fct = nn.MSELoss()
else:
    loss_fct= nn.CrossEntropyLoss()
print(model)
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler( "linear", optimizer=optimizer,
                              num_warmup_steps=0, num_training_steps=num_training_steps)
print(num_training_steps)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
print(device)

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


print(metric.compute())