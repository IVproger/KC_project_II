import pickle
import os

def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":
        MODEL_PATH = '/workdir/user_input/model'
    else:
        MODEL_PATH = path
    return MODEL_PATH

def save_model(model, path: str):
    model_path = get_model_path(path)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

def load_models():
    model_path = get_model_path("model_pipe.pkl")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

if __name__ == "__main__":
    model = load_models()
    print(model)