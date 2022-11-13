from nervaluate import Evaluator
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

        pred = model.predict(x)
        true = x['true']

        preds.extend(pred)
        trues.extend(true)


    evaluator = Evaluator(trues, preds, tags=['Task', 'Material', 'Metric', 'Method'], loader="list")
    results, results_by_tag = evaluator.evaluate()

    Path('output/tdmm_slvl/full.json').write_text(json.dumps(results))
    for k, v in results_by_tag.items():
        Path(f'output/tdmm_slvl/{k}.json').write_text(json.dumps(v))

if __name__ == '__main__': 

    model_path = "output/tdmm_slvl/tdmm_slvl.ckpt"
    train_path = "data/train.jsonl"
    dev_path = "data/dev.jsonl"
    test_path = "data/test.jsonl"
    
    evaluate_file(test_path, model_path)


    # load test data
    # data = pickle.loads(Path(data_path).read_bytes())['test']
    # test_tokens = [item['tokens'] for item in data]
    # test_tags = [item['tags'] for item in data]


    # load model
    # model = load_model(model_path)
    # predictions = model.extract_entities(test_tokens)


    # prediction
    # preds = []
    # for tokens, item in zip(test_tokens, predictions):
    #     pred = ['O'] * len(tokens)
    #     for ent in item:
    #         start, end = ent['span']
    #         pred[start] = f"B-{ent['type']}"
    #         for i in range(start+1, end+1):
    #             pred[i] = f"I-{ent['type']}"
    #     preds.append(pred)

    # for token, tag in zip(test_tokens[3], preds[3]):
    #     print(token, tag)


    # evaluation
    # evaluator = Evaluator(test_tags, preds, tags=['Task', 'Material', 'Metric', 'Method'], loader="list")
    # results, results_by_tag = evaluator.evaluate()