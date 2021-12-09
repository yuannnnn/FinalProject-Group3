# -*- coding: utf-8 -*-
# The following code I borrowed and adjusted from
# Ashwin Geet D'Sa's answer on Stackoverflow,
# https://stackoverflow.com/questions/65205582/how-can-i-add-a-bi-lstm-layer-on-top-of-bert-model?rq=1


class CustomBERTModel(nn.Module):
    def __init__(self, num_labels):
          super(CustomBERTModel, self).__init__()
          self.bert = BertModel.from_pretrained("bert-base-uncased")
          self.hidden_size = self.bert.config.hidden_size
          self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True,bidirectional=True)
          self.clf = nn.Linear(self.hidden_size*2, num_labels)

    def forward(self, **batch):
          sequence_output, pooled_output = self.bert(batch['input_ids'],batch['attention_mask'])[:2]
          lstm_output, (h,c) = self.lstm(sequence_output)
          hidden = torch.cat((lstm_output[:,-1, :self.hidden_size],lstm_output[:,0, self.hidden_size:]),dim=-1)
          hidden = F.dropout(hidden,0.1)
          linear_output = self.clf(hidden.view(-1,self.hidden_size*2)) 
          return linear_output



# The lines I added and adjusted for the stsb dataset:

loss_fct = nn.MSELoss()


loss = loss_fct(outputs, batch['labels'].unsqueeze(1))
