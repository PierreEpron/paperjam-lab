from training import ModelTrainer


def load_model(path):
    model = ModelTrainer.load_from_checkpoint(path, f1=None).model
    model = model.eval()
    return model