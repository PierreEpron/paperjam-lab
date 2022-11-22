from pathlib import Path
import tqdm
from dataset import NERDataLoader
from helpers import read_jsonl, write_jsonl
from training import create_config, train_model
from model import BertWordCRF

if __name__ == '__main__': 
    name = "mi_alpha"
    dirpath= Path(f"outputs/{name}/")
    train_path = Path("data/train.jsonl")
    dev_path = Path("data/dev.jsonl")
    test_path = Path("data/test.jsonl")

    if not dirpath.is_dir():
        dirpath.mkdir()

    name = f"{name}_{len([path for path in dirpath.glob('*') if path.is_dir()])}"
    dirpath /= name

    dirpath.mkdir()

    config = create_config(
        name, dirpath=dirpath, train_path=train_path, dev_path=dev_path, max_epoch=10, num_workers=1, 
        model_name='allenai/scibert_scivocab_uncased').hner
    
    model = BertWordCRF(
            tag_to_id=config.tag_to_id, model_name=config.model_name, tag_format=config.tag_format, 
            word_encoder=config.word_encoder, mode=config.mode)

    train_model(model, config)
    
    test_loader = model.model.data_processor.create_dataloader(
        read_jsonl(test_path), 
        batch_size=1, 
        num_workers=1, 
        shuffle=False
    )

    test_results = []
    for x in tqdm(test_loader):
        outputs = model.predict(x)
        test_results.append({
            'doc_id':outputs['doc_id'],
            'golds':outputs['golds'],
            'preds':outputs['preds'],
        })

    write_jsonl(dirpath / 'test_results.jsonl', test_results)