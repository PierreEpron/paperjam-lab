from sklearn.metrics import classification_report, confusion_matrix

from inference import load_model
from pathlib import Path
import json

from tqdm import tqdm
from helpers import read_jsonl

def evaluate_file(file_path, model_path, **loader_kwgs):
    
    model = load_model(model_path)

    data = read_jsonl(file_path)
    loader = model.data_processor.create_dataloader(data, is_train=True, **loader_kwgs)
    
    trues = []
    preds = []

    print()


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

    evaluate_file(test_path, model_path, batch_size=64,num_workers=8)


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