from sklearn.metrics import f1_score

from training import create_config, train_model
from model import BertCoref

def f1(true, pred, average="micro"):
    return f1_score(true.cpu().numpy(), pred.cpu().numpy(), average=average)

if __name__ == '__main__': 
    name = "coref"
    dirpath= "output/coref/"
    train_path = "data/train.jsonl"
    dev_path = "data/dev.jsonl"

    config = create_config(
        name, dirpath=dirpath, train_path=train_path, dev_path=dev_path, max_epoch=10, num_workers=1, 
        model_name='allenai/scibert_scivocab_uncased', train_batch_size=64, val_batch_size=64).coref
    
    model = BertCoref(model_name=config.model_name)

    train_model(model, f1, config)