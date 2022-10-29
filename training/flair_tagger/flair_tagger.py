from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import TokenEmbeddings, TransformerWordEmbeddings, WordEmbeddings, StackedEmbeddings, FlairEmbeddings, CharacterEmbeddings, BertEmbeddings
from typing import List


# define columns
columns = {0: 'text', 1: 'ner'} 

# this is the folder in which train, test and dev files reside
data_folder = 'training/corpus/fixed/'

# init a corpus using column format, data folder and the names of the train, dev and test files
corpus: Corpus = ColumnCorpus(data_folder, columns,
                              train_file='Final_SCIREX_train.txt',
                              test_file='Final_SCIREX_test.txt',
                              dev_file='Final_SCIREX_dev.txt')

# 2. what tag do we want to predict?
label_type = 'ner'

# 3. make the tag dictionary from the corpus
label_dict = corpus.make_label_dictionary(label_type=label_type)


# 4. initialize embeddings
embedding_types: List[TokenEmbeddings] = [
    TransformerWordEmbeddings('bert-base-cased'), 	
]
embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

# # 5. initialize sequence tagger
from flair.models import SequenceTagger
tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                        embeddings=embeddings,
                                        tag_dictionary=label_dict,
                                        tag_type=label_type,
                                        use_crf=True)

# # 6. initialize trainer
from flair.trainers import ModelTrainer
trainer: ModelTrainer = ModelTrainer(tagger, corpus)

# # 7. start training
trainer.train('../model/flair/tdm-bert',
              learning_rate=0.1,
              mini_batch_size=16,
              max_epochs=150)

