import torch
from unixcoder import UniXcoder
from create_embeddings import *
from onnx_inference import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PATH = "tmp/model.onnx"

max_length = 512

model = UniXcoder("microsoft/unixcoder-base")
model.eval()

source_ids = torch.tensor([[0, 6, 2, 1040, 126, 2508, 127, 2]])
mask = source_ids.ne(model.config.pad_token_id)
att = (mask.unsqueeze(1) * mask.unsqueeze(2))

torch.onnx.export(
    model.model, 
    (source_ids, att),
    f=PATH,  
    input_names=['input_ids', 'attention_mask'], 
    output_names=['last_hidden_state', 'pooler_output'], 
    dynamic_axes={'input_ids': {0: 'batch_size', 1: 'sequence'}, 
                  'attention_mask': {0: 'batch_size', 1: 'sequence', 2: 'sequence'}, 
                  'last_hidden_state': {0: 'batch_size', 1: 'sequence', 2: 'hidden'},
                  'pooler_output': {0: 'batch_size', 1: 'hidden'},
                }, 
    do_constant_folding=True, 
    opset_version=13, 
)

session = create_session(PATH)
_, onnx_code_embedding = onnx_inference(source_ids.numpy().tolist(), session)
with torch.inference_mode():
   _, code_embedding = model(source_ids)


np.testing.assert_allclose(code_embedding, onnx_code_embedding, rtol=1e-03, atol=1e-05)
