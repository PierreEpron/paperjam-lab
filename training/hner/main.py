from pathlib import Path
import pickle

from train_model import create_config, train_model

if __name__ == '__main__': 
    name = "keyphrase_extractor"
    dirpath= "training/hner/output/"

    data_path = "training/corpus/tdmm.pk" 
    # the dataset file (pickle) should be a dictionnary containing 'train' and 'dev' splits and a 'tag_to_id' dict
    # for instance:

    # data['train'] = [
    #     {'tokens': ['NER', 'is', 'an'], 'tags': ['B-ENT', 'O', 'O']}
    #     , ....]
    # data['dev'] = [
    #     {'tokens': ['NER', 'is', 'an'], 'tags': ['B-ENT', 'O', 'O']},
    #      ....]
    # data['tag_to_id'] = {'O' 0, 'B-ENT': 1, 'I-ENT': 2}
    # create config

    config = create_config(name, dirpath=dirpath, data_path=data_path, max_epoch=5, model_name='allenai/scibert_scivocab_uncased')

    # # train model
    train_model(config)