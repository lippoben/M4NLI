"""

Create a multi tasking model for both datasets VUA18 All POS and TroFi
Should result in better performance

Datasets:
    - VUA18
    - TroFi

Inputs:
    - Tokenized Sentence
    - Attention Mask

Model:
    - BERT-base-uncased/RoBERTa-base
    - Linear
    - Softmax

Outputs:
    - [prob1, prob2] [non-metaphor, metaphor]
    - [prob1, prob2] [non-literal, literal]


"""

import pandas as pd
import math

import torch
from torch.utils.data.sampler import RandomSampler
from pytorch_lightning.callbacks import RichProgressBar
from models import *
from data import *

pl.seed_everything(42, workers=True)

# select model to use
BERT_MODEL_NAME = "bert-base-uncased"
ROBERTA_MODEL_NAME = "roberta-base"

N_EPOCHS = 5
BATCH_SIZE = 16

train_VUA_df = pd.read_csv("C:/Users/lipb1/Documents/Year 4 Bristol/Technical Project/Datasets/VUA18/train.tsv",
                           sep='\t')

val_VUA_df = pd.read_csv("C:/Users/lipb1/Documents/Year 4 Bristol/Technical Project/Datasets/VUA18/dev.tsv",
                         sep='\t')

train_TroFi_df = pd.read_csv("C:/Users/lipb1/Documents/Year 4 Bristol/Technical Project/Datasets/TroFi/CLS/train0.tsv",
                             sep='\t')

for i in range(1, 10):
    train_TroFi_df = train_TroFi_df.append(pd.read_csv(
        "C:/Users/lipb1/Documents/Year 4 Bristol/Technical Project/Datasets/TroFi/CLS/train" + str(i) + ".tsv",
        sep='\t'))


# getting tokenizer from checkpoint
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

VUA_dataset = DatasetModule(train_VUA_df, tokenizer)
TroFi_dataset = DatasetModule(train_TroFi_df, tokenizer)

concat_dataset = ConcatDataset([VUA_dataset, TroFi_dataset])


class BatchSchedulerSampler(torch.utils.data.sampler.Sampler):
    """
    iterate over tasks and provide 3 random batches from VUA to every 1 random batch of TroFi
    """

    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.number_of_datasets = len(dataset.datasets)
        self.largest_dataset_size = max([len(cur_dataset) for cur_dataset in dataset.datasets])
        self.smallest_dataset_size = min([len(cur_dataset) for cur_dataset in dataset.datasets])

    def __len__(self):
        return self.batch_size * math.ceil((self.largest_dataset_size + self.smallest_dataset_size) / self.batch_size)

    def __iter__(self):
        samplers_list = []
        sampler_iterators = []
        for dataset_idx in range(self.number_of_datasets):
            cur_dataset = self.dataset.datasets[dataset_idx]
            sampler = RandomSampler(cur_dataset)
            samplers_list.append(sampler)
            cur_sampler_iterator = sampler.__iter__()
            sampler_iterators.append(cur_sampler_iterator)

        push_index_val = [0] + self.dataset.cumulative_sizes[:-1]
        step = self.batch_size * 4
        samples_to_grab = self.batch_size
        # In my case I don't want over or sub sampling. Since TroFi is 3 times smaller than VUA
        # I want 3 mini-batches of VUA to every 1 of TroFi
        epoch_samples = self.largest_dataset_size + self.smallest_dataset_size
        final_samples_list = []  # this is a list of indexes from the combined dataset
        for _ in range(0, epoch_samples, step):
            # grab 3 mini batchs from VUA
            for i in range(0, 3):
                cur_batch_sampler = sampler_iterators[0]
                cur_samples = []
                for _ in range(samples_to_grab):
                    try:
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[0]
                        cur_samples.append(cur_sample)
                    except StopIteration:
                        # got to the end of iterator - restart the iterator and continue to get samples
                        # until reaching "epoch_samples"
                        sampler_iterators[0] = samplers_list[0].__iter__()
                        cur_batch_sampler = sampler_iterators[0]
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[0]
                        cur_samples.append(cur_sample)
                final_samples_list.extend(cur_samples)

            # grab 1 mini batch from TroFi
            cur_batch_sampler = sampler_iterators[1]
            cur_samples = []
            for _ in range(samples_to_grab):
                try:
                    cur_sample_org = cur_batch_sampler.__next__()
                    cur_sample = cur_sample_org + push_index_val[1]
                    cur_samples.append(cur_sample)
                except StopIteration:
                    # got to the end of iterator - restart the iterator and continue to get samples
                    # until reaching "epoch_samples"
                    sampler_iterators[1] = samplers_list[1].__iter__()
                    cur_batch_sampler = sampler_iterators[1]
                    cur_sample_org = cur_batch_sampler.__next__()
                    cur_sample = cur_sample_org + push_index_val[1]
                    cur_samples.append(cur_sample)
            final_samples_list.extend(cur_samples)

        return iter(final_samples_list)


def main():
    dataloader_train = torch.utils.data.DataLoader(dataset=concat_dataset,
                                                   sampler=BatchSchedulerSampler(dataset=concat_dataset,
                                                                                 batch_size=BATCH_SIZE),
                                                   batch_size=BATCH_SIZE,
                                                   shuffle=False,
                                                   num_workers=0)

    data_module = DataModule(None, val_VUA_df, None, tokenizer, batch_size=BATCH_SIZE)
    data_module.setup()

    model = MultiTaskBERTCustomModel(
        checkpoint=BERT_MODEL_NAME,
        num_classes=2,
        steps_per_epoch=len(dataloader_train),
        n_epochs=N_EPOCHS
    )

    trainer = pl.Trainer(
        max_epochs=N_EPOCHS,
        gpus=1,
        enable_progress_bar=True,
        callbacks=[RichProgressBar(refresh_rate=30)],
        accelerator="gpu",
        devices=1,
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
        amp_backend='apex',
        limit_train_batches=len(dataloader_train),
        limit_val_batches=len(data_module.val_dataloader())
    )

    #trainer.fit(
    #           model,
    #           train_dataloaders=dataloader_train,
    #           val_dataloaders=data_module.val_dataloader(),
    #            )

    checkpoint = torch.load('C:/Users/lipb1/Documents/Year 4 Bristol/Technical Project/PythonProjects/'
                            'HuggingFaceLearning/models/multi-task-model-VUA/checkpoints/epoch=4-step=40110.ckpt')

    model.load_state_dict(checkpoint['state_dict'])

    trainer.validate(model, data_module.val_dataloader())


if __name__ == '__main__':
    main()
