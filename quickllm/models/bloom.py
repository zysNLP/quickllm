from .transformer import Decoder
from ..snippets import delete_arguments
import torch


class Bloom(Decoder):
    '''Bloom: https://arxiv.org/abs/2211.05100
    主要区别就是alibi编码，其他和bert结构一致
    '''
    @delete_arguments('with_pool', 'with_mlm', 'with_nsp')
    def __init__(self, *args, p_bias='alibi', **kwargs):
        kwargs.update({'p_bias': p_bias, 'weight': True, 'bias': True, 'is_decoder': True, 'final_layernorm': True})
        super().__init__(*args, **kwargs)
        self.prefix = 'bloom'

    def load_trans_ckpt(self, checkpoint):
        state_dict = torch.load(checkpoint, map_location='cpu')
        for i in range(self.num_hidden_layers):
            mapping = {
                f'h.{i}.self_attention.query_key_value.weight': 'decoderLayer.{}.multiHeadAttention.{}.weight',
                f'h.{i}.self_attention.query_key_value.bias': 'decoderLayer.{}.multiHeadAttention.{}.bias'
            }
            for old_key, new_key in mapping.items():
                # 如果当前ckpt不存在该key，则跳过
                if (qkv := state_dict.get(old_key)) is None:
                    continue
                tensor_list = torch.split(qkv, self.attention_head_size, 0)
                q, k, v = tensor_list[0::3], tensor_list[1::3], tensor_list[2::3]
                q, k, v = torch.cat(q), torch.cat(k), torch.cat(v)
                for i_k, i_v in {'q':q, 'k':k, 'v':v}.items():
                    state_dict[new_key.format(i, i_k)] = i_v
                state_dict.pop(old_key)
        return state_dict
    
    def variable_mapping(self):
        """权重映射字典，格式为{new_key: old_key}"""
        mapping = {
            'embeddings.word_embeddings.weight': 'word_embeddings.weight',
            'embeddings.layerNorm.weight': 'word_embeddings_layernorm.weight',
            'embeddings.layerNorm.bias': 'word_embeddings_layernorm.bias',
            'lm_head.weight': 'word_embeddings.weight',
            'LayerNormFinal.weight': 'ln_f.weight',
            'LayerNormFinal.bias': 'ln_f.bias'
            }
        for i in range(self.num_hidden_layers):
            mapping.update( 
            {
            f'decoderLayer.{i}.multiHeadAttention.o.weight': f'h.{i}.self_attention.dense.weight',
            f'decoderLayer.{i}.multiHeadAttention.o.bias': f'h.{i}.self_attention.dense.bias',
            f'decoderLayer.{i}.attnLayerNorm.weight': f'h.{i}.input_layernorm.weight',
            f'decoderLayer.{i}.attnLayerNorm.bias': f'h.{i}.input_layernorm.bias',
            f'decoderLayer.{i}.feedForward.intermediateDense.weight': f'h.{i}.mlp.dense_h_to_4h.weight',
            f'decoderLayer.{i}.feedForward.intermediateDense.bias': f'h.{i}.mlp.dense_h_to_4h.bias',
            f'decoderLayer.{i}.feedForward.outputDense.weight': f'h.{i}.mlp.dense_4h_to_h.weight',
            f'decoderLayer.{i}.feedForward.outputDense.bias': f'h.{i}.mlp.dense_4h_to_h.bias',
            f'decoderLayer.{i}.ffnLayerNorm.weight': f'h.{i}.post_attention_layernorm.weight',
            f'decoderLayer.{i}.ffnLayerNorm.bias': f'h.{i}.post_attention_layernorm.bias'
            })
        return mapping
