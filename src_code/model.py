from transformers import BertTokenizer, AdamW, BertModel, BertPreTrainedModel
import torch.nn as nn
import torch


class BertForNER(BertPreTrainedModel):
  def __init__(self, config):
    super(BertForNER, self).__init__(config)
    self.bert = BertModel(config)
    self.num_labels=config.num_labels
    self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)  # start/end
    self.dropout = nn.Dropout(config.hidden_dropout_prob)
    self.init_weights()

  def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, labels=None):
    outputs = self.bert(input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        position_ids=position_ids,
                        head_mask=head_mask)

    sequence_output = outputs[0]

    sequence_output = self.dropout(sequence_output)
    per_pos_label_logits = self.qa_outputs(sequence_output)  # batch*seq_len*9
    if labels is not None:
      if True:
        weight_tensor = torch.Tensor([0.1, 1, 2, 5, 3, 1, 2, 5, 3]).cuda()
        start_loss = nn.CrossEntropyLoss(weight=weight_tensor,reduction='none')(per_pos_label_logits.view(-1,self.num_labels), labels.view(-1))

        top_num = int(start_loss.shape[0]//2)
        start_loss = torch.mean(torch.topk(start_loss, min(top_num, start_loss.shape[0]))[0])
      else:
        start_loss = nn.CrossEntropyLoss()(per_pos_label_logits.view(-1, self.num_labels), labels.view(-1))
      outputs = start_loss
    else:
      outputs = per_pos_label_logits

    return outputs

