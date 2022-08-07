import pandas as pd
from pytorch_lightning.callbacks import RichProgressBar
from models import *
from data import *

pl.seed_everything(42, workers=True)
train_df = pd.read_csv("C:/Users/lipb1/Documents/Year 4 Bristol/Technical Project/Datasets/VUA18/train.tsv", sep='\t')


val_df = pd.read_csv("C:/Users/lipb1/Documents/Year 4 Bristol/Technical Project/Datasets/VUA18/dev.tsv", sep='\t')

"""
train_df = pd.read_csv("C:/Users/lipb1/Documents/Year 4 Bristol/Technical Project/Datasets/TroFi/CLS/train0.tsv",
                        sep='\t')
val_df = pd.read_csv("C:/Users/lipb1/Documents/Year 4 Bristol/Technical Project/Datasets/TroFi/CLS/test0.tsv",
                     sep='\t')
for i in range(1, 10):
    train_df = train_df.append(pd.read_csv(
        "C:/Users/lipb1/Documents/Year 4 Bristol/Technical Project/Datasets/TroFi/CLS/train" + str(i) + ".tsv",
        sep='\t'))
    val_df = val_df.append(pd.read_csv("C:/Users/lipb1/Documents/Year 4 Bristol/Technical Project/"
                                       "Datasets/TroFi/CLS/test"+str(i)+".tsv", sep='\t'))
"""
# select model to use
BERT_MODEL_NAME = "bert-base-uncased"
ROBERTA_MODEL_NAME = "roberta-base"

N_EPOCHS = 5
BATCH_SIZE = 16


def main():

    # getting tokenizer from checkpoint
    tokenizer = RobertaTokenizer.from_pretrained(ROBERTA_MODEL_NAME)
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

    data_module = DataModule(train_df, val_df, None, tokenizer, batch_size=BATCH_SIZE)
    data_module.setup()

    model = BERTCustomModel(
        checkpoint=BERT_MODEL_NAME,
        num_classes=2,
        steps_per_epoch=len(data_module.train_dataloader()),
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
                        limit_train_batches=len(data_module.train_dataloader()),
                        limit_val_batches=len(data_module.val_dataloader()),
                        )

    trainer.fit(
                model,
                train_dataloaders=data_module.train_dataloader(),
                val_dataloaders=data_module.val_dataloader(),
                )

    #checkpoint = torch.load('C:/Users/lipb1/Documents/Year 4 Bristol/Technical Project/PythonProjects/'
    #                        'HuggingFaceLearning/lightning_logs/version_27/checkpoints/epoch=4-step=30520.ckpt')

    # optimizer = AdamW(model.parameters(), lr=5e-5)

    # model.load_state_dict(checkpoint['state_dict'])

    trainer.validate(model, data_module.val_dataloader())


if __name__ == '__main__':
    main()
