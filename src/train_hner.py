from training import create_config, train_model
from model import BertWordCRF

if __name__ == '__main__': 
    name = "tdmm_slvl"
    dirpath= "output/tdmm_slvl/"
    train_path = "data/train.jsonl"
    dev_path = "data/dev.jsonl"

    config = create_config(
        name, dirpath=dirpath, train_path=train_path, dev_path=dev_path, max_epoch=5, num_workers=1, 
        model_name='allenai/scibert_scivocab_uncased').hner
    
    model = BertWordCRF(
            tag_to_id=config.tag_to_id, model_name=config.model_name, tag_format=config.tag_format, 
            word_encoder=config.word_encoder, mode=config.mode)

    train_model(model, config)