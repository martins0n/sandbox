import torch
import json
from unixcoder import UniXcoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_model():
    model = UniXcoder("microsoft/unixcoder-base")
    model.to(DEVICE)
    model.eval()
    return model


def get_embeddings(model, code, max_length=512):
    tokens_ids = model.tokenize([code], max_length=max_length, mode="<encoder-only>")
    source_ids = torch.tensor(tokens_ids).to(DEVICE)
    _, code_embedding = model(source_ids)

    norm = torch.nn.functional.normalize(code_embedding, p=2, dim=1)
    return list(norm.squeeze().numpy().astype(float))


if __name__ == "__main__":

    with open("tmp/index.json", "r") as f:
        index = json.load(f)

    with torch.inference_mode():
        model = make_model()

        for path, data in index.items():
            for name, _data in data.items():
                code = _data["source_code"]
                embedding = get_embeddings(model, code)
                _data["embedding"] = embedding

    with open("index.json", "w") as f:
        json.dump(index, f)
