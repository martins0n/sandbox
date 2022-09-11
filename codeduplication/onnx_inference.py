import onnxruntime
import numpy as np


PAD_TOKEN_ID = 1


def create_session(path: str = "tmp/model.onnx"):
    session = onnxruntime.InferenceSession(path)
    return session


def onnx_inference(input_ids: list, session: onnxruntime.InferenceSession):
    # input_ids (batch_size, dynamic sequence_length)
    max_len = max([len(i) for i in input_ids])
    input_ids: np.ndarray = np.array([i + [PAD_TOKEN_ID] * (max_len - len(i)) for i in input_ids]).astype(np.int64)
    _att_mask = input_ids != PAD_TOKEN_ID
    att_mask = np.einsum("ij,ik->ijk", _att_mask, _att_mask)
    
    token_embeddings = session.run(
        ['last_hidden_state'], {
            "input_ids": input_ids,
            "attention_mask": att_mask
        }
    )[0]
    sentence_embeddings = np.einsum("ijk,ij->ik", token_embeddings,  _att_mask) / _att_mask.sum(axis=1, keepdims=True)
    return token_embeddings, sentence_embeddings


if __name__ == "__main__":
    session = create_session()
    print(onnx_inference([[0, 6, 2, 1040, 126, 2508, 127, 2]], session)[1])