from pathlib import Path
from tqdm import tqdm

from helpers import read_jsonl
from inference import load_model

import numpy as np
import pickle 

if __name__ == '__main__': 

    model_path = "outputs/me_alpha/me_alpha_0/me_alpha_0.ckpt"
    test_path = "data/test.jsonl"
    
    model = load_model(model_path)
    
    test_loader = model.data_processor.create_dataloader(
        read_jsonl(test_path), 
        batch_size=1, 
        num_workers=1, 
        shuffle=False
    )

    tags_count = len(model.tag_to_id)
    ranks = np.zeros((tags_count, tags_count))
    lengths = np.zeros(tags_count)

    test_results = []
    for x in tqdm(test_loader):
        outputs = model.predict_ranks(x)
        for rank, tag in zip(outputs['ranks'][0], x['word_label'][0]):
            ranks[tag] += rank.numpy()
            lengths[tag] += 1
    
    Path(model_path).with_name('ranks.pickle').write_bytes(
        pickle.dumps({
            'lengths':lengths,
            'ranks':ranks
    }))