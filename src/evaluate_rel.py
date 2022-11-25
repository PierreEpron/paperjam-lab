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
    
    golds = []
    preds = []
    probs = []

    for x in tqdm(loader):
        outputs = model.predict(x)
        golds.extend(list(outputs['golds']))
        preds.extend(list(outputs['preds']))
        probs.extend(list(outputs['probs']))


    Path(model_path).with_name('values.json').write_text(json.dumps({
        'golds':[int(gold) for gold in golds], 
        'preds':[int(pred) for pred in preds],
        'probs':[int(prob) for prob in probs]
    }))
    Path(model_path).with_name('report.json').write_text(json.dumps(classification_report(golds, preds, output_dict=True)))
    Path(model_path).with_name('cmatrix.json').write_text(json.dumps(confusion_matrix(golds, preds).tolist()))

if __name__ == '__main__': 

    model_path = "output/rel/rel.ckpt"
    train_path = "data/train.jsonl"
    dev_path = "data/dev.jsonl"
    test_path = "data/test.jsonl"
    
    evaluate_file(test_path, model_path, batch_size=1)