#! -*- coding: utf-8 -*-
# 基础测试：ERNIE模型测试

from quickllm.models import build_transformer_model
from quickllm.tokenizers import Tokenizer
import torch

# 加载模型，请更换成自己的路径
root_model_path = "E:/pretrain_ckpt/ernie/baidu@ernie-1-base-zh"
# root_model_path = "E:/pretrain_ckpt/ernie/baidu@ernie-3-base-zh"

vocab_path = root_model_path + "/vocab.txt"
config_path = root_model_path + "/config.json"
checkpoint_path = root_model_path + '/pytorch_model.bin'


# 建立分词器
tokenizer = Tokenizer(vocab_path, do_lower_case=True)
model = build_transformer_model(config_path, checkpoint_path, model='ERNIE', with_mlm='softmax')  # 建立模型，加载权重

token_ids, segments_ids = tokenizer.encode("科学[MASK][MASK]是第一生产力")
print(''.join(tokenizer.ids_to_tokens(token_ids)))

tokens_ids_tensor = torch.tensor([token_ids])
segment_ids_tensor = torch.tensor([segments_ids])

# 需要传入参数
model.eval()
with torch.no_grad():
    _, probas = model([tokens_ids_tensor, segment_ids_tensor])
    result = torch.argmax(probas[0, 3:5], dim=-1).numpy()
    print(tokenizer.decode(result))
