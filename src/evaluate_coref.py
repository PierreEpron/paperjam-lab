from sklearn.metrics import classification_report, confusion_matrix

from inference import load_model
from pathlib import Path
import json

from tqdm import tqdm
from helpers import read_jsonl

def evaluate_file(file_path, model_path, **loader_kwgs):
    
    model = load_model(model_path)

    data = read_jsonl(file_path)
    loader = model.data_processor.create_dataloader(data, **loader_kwgs)
    
    trues = []
    preds = []

    for x in tqdm(loader):

        preds.extend(model.predict(x))
        trues.extend(x['true'])


    Path(model_path).with_name('report.json').write_text(json.dumps(classification_report(trues, preds, output_dict=True)))
    Path(model_path).with_name('cmatrix.json').write_text(json.dumps(confusion_matrix(trues, preds).tolist()))

if __name__ == '__main__': 

    model_path = "output/coref/coref.ckpt"
    train_path = "data/train.jsonl"
    dev_path = "data/dev.jsonl"
    test_path = "data/test.jsonl"
    
    f1=None

    evaluate_file(test_path, model_path, batch_size=64)