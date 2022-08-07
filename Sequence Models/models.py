"""
defines model
"""

import numpy as np
import torch
import torch.nn as nn
import torchmetrics
import torch.optim as optim
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig, BertForTokenClassification, \
     RobertaForTokenClassification
import pytorch_lightning as pl


class BERTCustomModel(pl.LightningModule):
    # fetch model from hugging hub using checkpoint weights
    # define BERT with softmax activation
    def __init__(self, checkpoint, num_classes, steps_per_epoch=None, n_epochs=None):
        super().__init__()
        # define metrics
        self.trainAccuracy = torchmetrics.Accuracy(average='macro', num_classes=2)
        self.valAccuracy = torchmetrics.Accuracy(average='macro', num_classes=2)
        self.trainPrecision = torchmetrics.Precision(average='macro', num_classes=2)
        self.valPrecision = torchmetrics.Precision(average='macro', num_classes=2)
        self.trainRecall = torchmetrics.Recall(average='macro', num_classes=2)
        self.valRecall = torchmetrics.Recall(average='macro', num_classes=2)
        self.trainF1 = torchmetrics.F1Score(average='macro', num_classes=2)
        self.valF1 = torchmetrics.F1Score(average='macro', num_classes=2)
        # Load Model with given checkpoint and extract its body
        self.bert = BertModel.from_pretrained(checkpoint, return_dict=True)
        # add linear layer
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes, bias=True)
        self.steps_per_epoch = steps_per_epoch
        self.n_epochs = n_epochs
        self.criterion = nn.CrossEntropyLoss(weight=torch.tensor([1, 3], dtype=torch.float))

    def forward(self, input_ids, attention_mask=None, label=None):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        output = self.classifier(output.pooler_output)
        output = torch.softmax(output, dim=-1)

        # calculate losses
        loss = None
        if label is not None:
            loss = self.criterion(output, torch.max(label, 1)[0].long())

        return loss, output

    def training_step(self, batch, batch_idx):
        input_ids = batch["inputs_ids"]
        attention_mask = batch["attention_mask"]
        label = batch["label"]
        loss, outputs = self(input_ids, attention_mask, label)
        self.trainAccuracy(torch.max(outputs, 1)[1], torch.max(label, 1)[0].int())
        self.trainF1(torch.max(outputs, 1)[1], torch.max(label, 1)[0].int())
        # print a much of metrics through out training to monitor learning
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_accuracy", self.trainAccuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_F1", self.trainF1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs, "label": label}

    def validation_step(self, batch, batch_idx):
        input_ids = batch["inputs_ids"]
        attention_mask = batch["attention_mask"]
        label = batch["label"]
        loss, outputs = self(input_ids, attention_mask, label)
        self.valAccuracy(torch.max(outputs, 1)[1], torch.max(label, 1)[0].int())
        self.valF1(torch.max(outputs, 1)[1], torch.max(label, 1)[0].int())
        self.valPrecision(torch.max(outputs, 1)[1], torch.max(label, 1)[0].int())
        self.valRecall(torch.max(outputs, 1)[1], torch.max(label, 1)[0].int())
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("vaL_accuracy", self.valAccuracy, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_precision", self.valPrecision, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_Recall", self.valRecall, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_F1", self.valF1, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["inputs_ids"]
        attention_mask = batch["attention_mask"]
        label = batch["label"]
        loss, outputs = self(input_ids, attention_mask, label)
        return loss

    def configure_optimizers(self):
        # set up optimizer
        optimizer = AdamW(self.parameters(), lr=5e-5)

        warmup_steps = self.steps_per_epoch // 3
        total_steps = self.steps_per_epoch * self.n_epochs - warmup_steps

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            warmup_steps,
            total_steps
        )

        return [optimizer], [scheduler]


class RoBERTaCustomModel(pl.LightningModule):
    # fetch model from hugging hub using checkpoint weights
    # define RoBERTa with softmax activation
    def __init__(self, checkpoint, num_classes, steps_per_epoch=None, n_epochs=None):
        super().__init__()
        # define metrics
        self.trainAccuracy = torchmetrics.Accuracy(average='macro', num_classes=2)
        self.valAccuracy = torchmetrics.Accuracy(average='macro', num_classes=2)
        self.trainPrecision = torchmetrics.Precision(average='macro', num_classes=2)
        self.valPrecision = torchmetrics.Precision(average='macro', num_classes=2)
        self.trainRecall = torchmetrics.Recall(average='macro', num_classes=2)
        self.valRecall = torchmetrics.Recall(average='macro', num_classes=2)
        self.trainF1 = torchmetrics.F1Score(average='macro', num_classes=2)
        self.valF1 = torchmetrics.F1Score(average='macro', num_classes=2)
        # Load Model with given checkpoint and extract its body
        self.roberta = RobertaModel.from_pretrained(checkpoint, return_dict=True)
        # add linear layer
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_classes, bias=True)
        self.steps_per_epoch = steps_per_epoch
        self.n_epochs = n_epochs
        self.criterion = nn.CrossEntropyLoss(weight=torch.tensor([3, 1], dtype=torch.float))

    def forward(self, input_ids, attention_mask=None, label=None):
        output = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        output = self.classifier(output.pooler_output)
        output = torch.softmax(output, dim=-1)

        # calculate losses
        loss = None
        if label is not None:
            loss = self.criterion(output, torch.max(label, 1)[0].long())

        return loss, output

    def training_step(self, batch, batch_idx):
        input_ids = batch["inputs_ids"]
        attention_mask = batch["attention_mask"]
        label = batch["label"]
        loss, outputs = self(input_ids, attention_mask, label)
        self.trainAccuracy(torch.max(outputs, 1)[1], torch.max(label, 1)[0].int())
        self.trainF1(torch.max(outputs, 1)[1], torch.max(label, 1)[0].int())
        # print a much of metrics through out training to monitor learning
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_accuracy", self.trainAccuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_F1", self.trainF1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs, "label": label}

    def validation_step(self, batch, batch_idx):
        input_ids = batch["inputs_ids"]
        attention_mask = batch["attention_mask"]
        label = batch["label"]
        loss, outputs = self(input_ids, attention_mask, label)
        self.valAccuracy(torch.max(outputs, 1)[1], torch.max(label, 1)[0].int())
        self.valF1(torch.max(outputs, 1)[1], torch.max(label, 1)[0].int())
        self.valPrecision(torch.max(outputs, 1)[1], torch.max(label, 1)[0].int())
        self.valRecall(torch.max(outputs, 1)[1], torch.max(label, 1)[0].int())
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("vaL_accuracy", self.valAccuracy, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_precision", self.valPrecision, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_Recall", self.valRecall, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_F1", self.valF1, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["inputs_ids"]
        attention_mask = batch["attention_mask"]
        label = batch["label"]
        loss, outputs = self(input_ids, attention_mask, label)
        return loss

    def configure_optimizers(self):
        # set up optimizer
        optimizer = AdamW(self.parameters(), lr=5e-5)

        warmup_steps = self.steps_per_epoch // 3
        total_steps = self.steps_per_epoch * self.n_epochs - warmup_steps

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            warmup_steps,
            total_steps
        )

        return [optimizer], [scheduler]


class MultiTaskBERTCustomModel(pl.LightningModule):
    # fetch model from hugging hub using checkpoint weights
    # define BERT with softmax activation
    def __init__(self, checkpoint, num_classes, steps_per_epoch=None, n_epochs=None):
        super().__init__()
        # define metrics
        self.trainAccuracy = torchmetrics.Accuracy(average='macro', num_classes=2)
        self.valAccuracy = torchmetrics.Accuracy(average='macro', num_classes=2)
        self.trainPrecision = torchmetrics.Precision(average='macro', num_classes=2)
        self.valPrecision = torchmetrics.Precision(average='macro', num_classes=2)
        self.trainRecall = torchmetrics.Recall(average='macro', num_classes=2)
        self.valRecall = torchmetrics.Recall(average='macro', num_classes=2)
        self.trainF1 = torchmetrics.F1Score(average='macro', num_classes=2)
        self.valF1 = torchmetrics.F1Score(average='macro', num_classes=2)
        # Load Model with given checkpoint and extract its body
        self.bert = BertModel.from_pretrained(checkpoint, return_dict=True)
        # add linear layer to reduce dimensions
        self.linear = nn.Linear(self.bert.config.hidden_size, 248, bias=True)
        # add classifiers for both tasks
        self.classiferMetaphor = nn.Linear(248, num_classes, bias=True)
        self.classiferNonLiteral = nn.Linear(248, num_classes, bias=True)

        self.steps_per_epoch = steps_per_epoch
        self.n_epochs = n_epochs
        # define loss function
        # scale positives by 3 as the dataset is unbalanced
        self.criterion = nn.CrossEntropyLoss(weight=torch.tensor([3, 1], dtype=torch.float))

    def forward(self, input_ids, attention_mask=None, label=None):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        output = self.linear(output.pooler_output)
        outputMetaphor = self.classiferMetaphor(output)
        outputNonLiteral = self.classiferNonLiteral(output)
        outputMetaphor = torch.softmax(outputMetaphor, dim=-1)
        outputNonLiteral = torch.softmax(outputMetaphor, dim=-1)

        # calculate losses
        loss = None
        if label is not None:
            # scale aux task loss so that it doesnt dominate the training
            loss = (self.criterion(outputMetaphor, torch.max(label, 1)[0].long()) +
                    self.criterion(outputNonLiteral, torch.max(label, 1)[0].long())*0.1)

        return loss, outputMetaphor

    def training_step(self, batch, batch_idx):
        input_ids = batch["inputs_ids"]
        attention_mask = batch["attention_mask"]
        label = batch["label"]
        loss, outputs = self(input_ids, attention_mask, label)
        # select the metaphor output as this is the main task
        self.trainAccuracy(torch.max(outputs, 1)[1], torch.max(label, 1)[0].int())
        self.trainF1(torch.max(outputs, 1)[1], torch.max(label, 1)[0].int())
        # print a much of metrics through out training to monitor learning
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_accuracy", self.trainAccuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_F1", self.trainF1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs, "label": label}

    def validation_step(self, batch, batch_idx):
        input_ids = batch["inputs_ids"]
        attention_mask = batch["attention_mask"]
        label = batch["label"]
        loss, outputs = self(input_ids, attention_mask, label)
        self.valAccuracy(torch.max(outputs, 1)[1], torch.max(label, 1)[0].int())
        self.valF1(torch.max(outputs, 1)[1], torch.max(label, 1)[0].int())
        self.valPrecision(torch.max(outputs, 1)[1], torch.max(label, 1)[0].int())
        self.valRecall(torch.max(outputs, 1)[1], torch.max(label, 1)[0].int())
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("vaL_accuracy", self.valAccuracy, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_precision", self.valPrecision, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_Recall", self.valRecall, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_F1", self.valF1, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["inputs_ids"]
        attention_mask = batch["attention_mask"]
        label = batch["label"]
        loss, outputs = self(input_ids, attention_mask, label)
        return loss

    def configure_optimizers(self):
        # set up optimizer
        optimizer = AdamW(self.parameters(), lr=1e-5)

        warmup_steps = self.steps_per_epoch // 3
        total_steps = self.steps_per_epoch * self.n_epochs - warmup_steps

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            warmup_steps,
            total_steps
        )

        return [optimizer], [scheduler]


# # token classification using BERT
# (Depreciated)
# class BERTTokenClassificationModel(torch.nn.Module):
#     def __init__(self, num_labels):
#         super().__init__()
#         self.bertLayer = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)
#         # has to be [1, 3, 0] so that the padding doesnt effect the loss function
#         self.criterion = nn.CrossEntropyLoss(weight=torch.Tensor([3, 1]), reduction='sum', ignore_index=-100)
#         self.num_labels = num_labels
#
#     def forward(self, batch_input_ids, token_type_ids, attention_mask, labels):
#         pooled_output = self.bertLayer(batch_input_ids,
#                                        token_type_ids=None,
#                                        attention_mask=attention_mask,
#                                        labels=None)
#
#         # SOFTMAX LAYER
#         softmax_output = torch.softmax(pooled_output[0], dim=2)
#
#         reshapeParams = [int(softmax_output.shape[0]), int(softmax_output.shape[2]), int(softmax_output.shape[1])]
#
#         #print(torch.max(softmax_output, 2)[1][0])
#         #print(labels[0])
#
#         # calculate losses
#         loss = None
#         if labels is not None:
#             loss = self.criterion(softmax_output.view(reshapeParams[0], reshapeParams[1], reshapeParams[2]), labels)
#
#         return loss, softmax_output

