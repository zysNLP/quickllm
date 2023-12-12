#! -*- coding: utf-8 -*-
# 基础测试：mlm预测


from quickllm.models import build_transformer_model
from quickllm.tokenizers import Tokenizer
import torch

# 加载模型，请更换成自己的路径
root_model_path = "E:/pretrain_ckpt/albert/brightmart@albert_base_zh"
vocab_path = root_model_path + "/vocab.txt"
config_path = root_model_path + "/config.json"
checkpoint_path = root_model_path + '/pytorch_model.bin'

# 建立分词器
tokenizer = Tokenizer(vocab_path, do_lower_case=True)
model = build_transformer_model(config_path, checkpoint_path, model='albert', with_mlm='softmax')  # 建立模型，加载权重

token_ids, segments_ids = tokenizer.encode("今天[MASK]情很好")
print(''.join(tokenizer.ids_to_tokens(token_ids)))

tokens_ids_tensor = torch.tensor([token_ids])
segment_ids_tensor = torch.tensor([segments_ids])

print('====quickllm output====')
model.eval()
with torch.no_grad():
    _, probas = model([tokens_ids_tensor, segment_ids_tensor])
    result = torch.argmax(probas[0, 3:4], dim=-1).numpy()
    print(tokenizer.decode(result))


# ==========================transformer调用==========================
from transformers import AutoTokenizer, AlbertForMaskedLM
import torch
from torch.nn.functional import softmax

tokenizer = AutoTokenizer.from_pretrained(root_model_path)
model = AlbertForMaskedLM.from_pretrained(root_model_path)

inputtext = "今天[MASK]情很好"

maskpos = tokenizer.encode(inputtext, add_special_tokens=True).index(103)

input_ids = torch.tensor(tokenizer.encode(inputtext, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
outputs = model(input_ids, labels=input_ids)
loss, prediction_scores = outputs[:2]
logit_prob = softmax(prediction_scores[0, maskpos],dim=-1).data.tolist()
predicted_index = torch.argmax(prediction_scores[0, maskpos]).item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
print('====transformers output====')
print(predicted_token, logit_prob[predicted_index])
