from pathlib import Path
import logging
import json
import os


def get_logger():
    '''
        Shorcut for get logger
        
        Returns
        -------
        logger : logging.Logger, logger.
    '''
    if not Path('logs').is_dir():
        os.mkdir('logs')

    # Get or create logger
    logger = logging.getLogger('parser')
    logger.setLevel(logging.DEBUG)
    # create console handler and set level to debug
    # Add file handler
    f_handler = logging.FileHandler('logs/test.log', encoding='utf-8')
    ch_handler = logging.StreamHandler()
    logger.addHandler(f_handler)
    logger.addHandler(ch_handler)
    # Set handlers formatter
    formatter = logging.Formatter('%(asctime)s :: %(name)s :: %(levelname)s :: %(threadName)s :: %(message)s')
    # logging.StreamHandler().setFormatter(formatter)
    f_handler.setFormatter(formatter)
    ch_handler.setFormatter(formatter)
    
    return logger

def read_jsonl(path, encoding='utf-8'):
    """
        Shortcut for read jsonl file

        Parameters
        ----------
        path : str or Path, path of data to read
        encoding : str, default='utf-8', encoding format to read.
    """
    path = Path(path) if isinstance(path, str) else path
    return [json.loads(line) for line in path.read_text(encoding=encoding).strip().split('\n')]

def write_jsonl(path, data, encoding='utf-8'):
    """
        Shortcut for write jsonl file

        Parameters
        ----------
        path : str or Path, path of data to read
        encoding : str, default='utf-8', encoding format to write.
    """
    path = Path(path) if isinstance(path, str) else path
    path.write_text('\n'.join([json.dumps(item) for item in data]), encoding=encoding)