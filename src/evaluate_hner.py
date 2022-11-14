from nervaluate import Evaluator
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
    
    true = []
    pred = []

    for x in tqdm(loader):

        pred.extend(v)
        true.extend(x['true'])


    evaluator = Evaluator(true, pred, tags=['Task', 'Material', 'Metric', 'Method'], loader="list")
    results, results_by_tag = evaluator.evaluate()

    print(confusion_matrix(true, pred))

    Path(model_path).with_name('full.json').write_text(json.dumps(results))
    for k, v in results_by_tag.items():
        Path(model_path).with_name(f'{k.lower()}.json').write_text(json.dumps(v))

if __name__ == '__main__': 

    model_path = "output/tdmm_slvl/tdmm_slvl.ckpt"
    train_path = "data/train.jsonl"
    dev_path = "data/dev.jsonl"
    test_path = "data/test.jsonl"
    
    evaluate_file(test_path, model_path)