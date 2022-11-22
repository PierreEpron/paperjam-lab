from training import create_config, train_model
from model import BertRel

if __name__ == '__main__': 
    name = "rel"
    dirpath= "output/rel/"
    train_path = "data/train.jsonl"
    dev_path = "data/dev.jsonl"

    config = create_config(
        name, dirpath=dirpath, train_path=train_path, dev_path=dev_path, max_epoch=10, num_workers=1, 
        model_name='allenai/scibert_scivocab_uncased', train_batch_size=1, val_batch_size=1).rel
    
    model = BertRel(model_name=config.model_name)

    train_model(model, config)