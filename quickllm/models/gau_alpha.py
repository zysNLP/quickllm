from .roformer import RoFormerV2
from torch import nn
import copy
from ..layers import LayerNorm, BlockIdentity, GatedAttentionUnit
import torch.nn.functional as F
import torch

class GAU_alpha(RoFormerV2):
    def __init__(self, *args, **kwargs):
        kwargs.update({'p_bias': 'rotary', 'weight': False, 'bias': False, 'norm_mode': 'rmsnorm', 'normalization': 'softmax_plus'})
        super().__init__(*args, **kwargs)

        layer = self.GAU_Layer(**kwargs)
        self.encoderLayer = nn.ModuleList([copy.deepcopy(layer) if layer_id in self.keep_hidden_layers else BlockIdentity() for layer_id in range(self.num_hidden_layers)])
    
    def load_trans_ckpt(self, checkpoint):
        return torch.load(checkpoint, map_location='cpu')
    
    def load_variable(self, state_dict, name):
        variable = state_dict[name]
        if name in {'embeddings.word_embeddings.weight', 'mlmDecoder.weight'}:
            return self.load_embeddings(variable)
        else:
            return variable

    def variable_mapping(self):
        '''在convert脚本里已经把key转成quickllm可用的
        '''
        return {k: k for k, _ in self.named_parameters()}

    class GAU_Layer(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.gau = GatedAttentionUnit(**kwargs)
            self.dropout_rate = kwargs.get('dropout_rate')
            self.attnLayerNorm = LayerNorm(**kwargs)
        def forward(self, hidden_states=None, attention_mask=None, conditional_emb=None, **model_kwargs):
            gau_hidden_states = self.gau(hidden_states, attention_mask)
            hidden_states = hidden_states + F.dropout(gau_hidden_states, p=self.dropout_rate, training=self.training)
            hidden_states = self.attnLayerNorm(hidden_states, conditional_emb)
            model_kwargs['hidden_states'] = hidden_states
            return model_kwargs