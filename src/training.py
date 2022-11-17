from pathlib import Path

import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from dataset import labels_to_id
from helpers import read_jsonl

class ModelTrainer(pl.LightningModule):

    def __init__(self, model, f1, config):

        super().__init__()

        self.model = model
        self.config = config

        self.save_hyperparameters()
        
        self.f1 = f1

    def training_step(self, batch, batch_idx):

        loss = self.model(batch, compute_loss=True)['loss']

        self.log('train_loss', loss)

        return loss

    def validation_step(self, val_batch, batch_idx):

        output = self.model.predict(val_batch)

        # TODO : This should work for all model (by put the evaluation method in model or on config ?)
        # We should use nereval here for consistency
        f1_micro_base = self.f1(output['true'], output['pred'].cpu().numpy(), average="micro")

        self.log('f1_score', f1_micro_base, prog_bar=True)

    def train_dataloader(self):
        return self.model.data_processor.create_dataloader(
            read_jsonl(self.config.train_path), is_train=True, 
            batch_size=self.config.train_batch_size, 
            num_workers=self.config.num_workers, 
            shuffle=True)

    def val_dataloader(self):
        return self.model.data_processor.create_dataloader(
            read_jsonl(self.config.dev_path), 
            batch_size=self.config.val_batch_size, 
            num_workers=self.config.num_workers, 
            shuffle=False)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.config.learning_rate)
        return optimizer


def train_model(model, f1, config):

    ckpt_path = Path(f'{config.dirpath}{config.name}.ckpt')


    lightning_model = ModelTrainer(model, f1, config)

    model_ckp = ModelCheckpoint(dirpath=config.dirpath, save_top_k=1,
                                monitor='f1_score', mode='max', filename=config.name)

    early_stop = EarlyStopping(patience=5, monitor='f1_score', mode='max')

    trainer = pl.Trainer(accelerator='gpu', devices=1, precision=32, max_epochs=config.max_epoch, callbacks=[
                        model_ckp, early_stop], check_val_every_n_epoch=config.check_val_every_n_epoch, 
                        limit_val_batches=config.limit_val_batches, val_check_interval=config.val_check_interval)

    if ckpt_path.is_file():
        print(f'Find a checkpoint at {ckpt_path} : resume training ...')
        trainer.fit(lightning_model, ckpt_path=ckpt_path)

    else:  
        print(f'No checkpoint at {ckpt_path} : start training ...')
        trainer.fit(lightning_model)


# TODO : Specify tag_format is only for NER

class ConfigClass(object):
    def __init__(self, **kwargs):
        self.__dict__ = dict(kwargs)


def create_config(name, dirpath, train_path, dev_path, model_name, word_encoder="transformer", mode="word", 
    device="cuda", train_batch_size=8, val_batch_size=8, num_workers=1, learning_rate=2e-5, 
    max_epoch=5, tag_format='BIO', check_val_every_n_epoch=1, limit_train_batches=1.0, 
    limit_val_batches=1.0, val_check_interval=1.0):

    tag_to_id = labels_to_id(read_jsonl(train_path))

    # TODO : ATM it's hner and coref is same but as the end they should be different

    hner = ConfigClass(
        tag_to_id=tag_to_id, name=name, dirpath=dirpath, train_path=train_path, dev_path=dev_path, word_encoder=word_encoder, mode=mode, 
        model_name=model_name, device=device, train_batch_size=train_batch_size, val_batch_size=val_batch_size, 
        num_workers=num_workers, learning_rate=learning_rate, max_epoch=max_epoch, tag_format=tag_format,
        check_val_every_n_epoch=check_val_every_n_epoch, limit_train_batches=limit_train_batches, 
        limit_val_batches=limit_val_batches, val_check_interval=val_check_interval)

    coref = ConfigClass(
        tag_to_id=tag_to_id, name=name, dirpath=dirpath, train_path=train_path, dev_path=dev_path, word_encoder=word_encoder, mode=mode, 
        model_name=model_name, device=device, train_batch_size=train_batch_size, val_batch_size=val_batch_size, 
        num_workers=num_workers, learning_rate=learning_rate, max_epoch=max_epoch, tag_format=tag_format,
        check_val_every_n_epoch=check_val_every_n_epoch, limit_train_batches=limit_train_batches, 
        limit_val_batches=limit_val_batches, val_check_interval=val_check_interval)

    rel = ConfigClass(name=name, dirpath=dirpath, train_path=train_path, dev_path=dev_path, word_encoder=word_encoder, mode=mode, 
        model_name=model_name, device=device, train_batch_size=train_batch_size, val_batch_size=val_batch_size, 
        num_workers=num_workers, learning_rate=learning_rate, max_epoch=max_epoch, tag_format=tag_format,
        check_val_every_n_epoch=check_val_every_n_epoch, limit_train_batches=limit_train_batches, 
        limit_val_batches=limit_val_batches, val_check_interval=val_check_interval)

    cfg = ConfigClass(hner=hner, coref=coref, rel=rel)

    return cfg
