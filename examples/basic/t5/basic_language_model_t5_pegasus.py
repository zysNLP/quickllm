#! -*- coding: utf-8 -*-
# 调用T5 PEGASUS, 使用到是BertTokenizer

import torch
from quickllm.models import build_transformer_model
from quickllm.tokenizers import Tokenizer, load_vocab
from quickllm.generation import AutoRegressiveDecoder, Seq2SeqGeneration
import jieba
jieba.initialize()

# bert配置
# pretrain_model = 'E:/pretrain_ckpt/t5/sushen@chinese_t5_pegasus_small_torch/'
pretrain_model = 'E:/pretrain_ckpt/t5/sushen@chinese_t5_pegasus_base_torch/'
config_path = pretrain_model + 'quickllm_config.json'
checkpoint_path = pretrain_model + 'pytorch_model.bin'
dict_path = pretrain_model + 'vocab.txt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 加载并精简词表，建立分词器
tokenizer = Tokenizer(
    dict_path,
    do_lower_case=True,
    pre_tokenize=lambda s: jieba.cut(s, HMM=False)
)

model = build_transformer_model(config_path, checkpoint_path).to(device)


# 第一种自定义方式
class AutoTitle(AutoRegressiveDecoder):
    """seq2seq解码器
    """
    @AutoRegressiveDecoder.wraps(default_rtype='logits')
    def predict(self, inputs, output_ids, states):
        # inputs中包含了[decoder_ids, encoder_hidden_state, encoder_attention_mask]
        res = model.decoder.predict([output_ids] + inputs)
        return res[-1][:, -1, :] if isinstance(res, list) else res[:, -1, :]  # 保留最后一位

    def generate(self, text, topk=1):
        token_ids, _ = tokenizer.encode(text, maxlen=256)
        token_ids = torch.tensor([token_ids], device=device)
        encoder_output = model.encoder.predict([token_ids])
        output_ids = self.beam_search(encoder_output, topk=topk)[0]  # 基于beam search
        return tokenizer.decode([int(i) for i in output_ids.cpu().numpy()])
autotitle = AutoTitle(start_id=tokenizer._token_start_id, end_id=tokenizer._token_end_id, maxlen=32, device=device)  # 这里end_id可以设置为tokenizer._token_end_id这样结果更短

# 第二种方式
# autotitle = Seq2SeqGeneration(model, tokenizer, start_id=tokenizer._token_start_id, end_id=tokenizer._token_end_id, 
#                               maxlen=32, default_rtype='logits', mode='beam_search')

if __name__ == '__main__':
    print(autotitle.generate('今天天气不错啊', topk=1))

# small版输出：我是个女的，我想知道我是怎么想的
# base版输出：请问明天的天气怎么样啊？