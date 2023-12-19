## 一. 简介

LLM学习资源库。一个使用pytorch和部分Tensorflow2实现的项目，可以 **<u>*本地运行和调试*</u>** 的大模型LLM相关的应用和代码

再次强调：强调本地调试习惯，强调代码跳转，强调快速掌握LLM相关工作

```python
# 下次更新：1.基于PyTorch的Mistral-AI的MOE模型微调；2.各类TensorRT-LLM和推理加速的相关工作
```

### 最新更新：

- 20231219：🐯feat(basic_language_model_moe*):MOE新增2维和3维数据的训练过程
- 20231218：🐯feat(quickllm/clients/vllm.py):添加vllm的api请求模块
- 20231214：🐯feat(README): 增加Qwen-TensorRT-LLM的链接和说明

- 20231213：🐯feat(quickllm/clients/triton_client*):添加ChatGLM系列triton服务基础代码
- 20231213：🐯feat(basic_language_model_moe.py):添加PyTorch实现MOE模型初始调试代码
- 20231212：🐯feat(quickllm/layers/lora.py):添加TF2实现的LORA层（初测版）
- 20231201：🐯feat: 增加Tensorflow2的ROPE函数实现
- 

**调试方式：**

​		**格式1. 只有一个脚本文件，将examples中的py脚本复制到quickllm文件夹的同级目录（当前项目根目录）**

​		**格式2. 如果调试脚本是个文件夹，推荐将依赖改成相对路径或者格式1的形式；或者在需要使用from quickllm导包调用的脚本代码中添加父级目录**

```python
import sys
sys.path.append('/path/to/directory of quickllm')  # quickllm文件夹的父级目录
```

**核心功能**： chatglm、chatglm2、llama、llama2、 baichuan、ziya、bloom等开源大模型权重进行推理和微调、prompt应用



### 二、快速启动（以MOE核心代码原理为例）

```python
# -*- coding: utf-8 -*- 
# @Time : 2023/12/13 02:09 
# @Author : ys 
# @File : basic_language_model_moe.py

import torch
from torch import nn
from quickllm.layers.moe import MoE


if __name__ == "__main__":

    moe = MoE(
        dim=512,  # 输入张量的维度
        num_experts=16,  # 专家数量，可以增加该参数而不增加计算量
        hidden_dim=512 * 4,  # 每个专家网络中的隐藏层维度，默认为 4 倍输入维度
        activation=nn.LeakyReLU,  # 使用的激活函数，默认为 GELU
        second_policy_train='random',  # 使用的第二名专家的训练策略
        second_policy_eval='random',  # 使用的第二名专家的验证策略
        second_threshold_train=0.2,  # 训练时使用的第二名专家阈值
        second_threshold_eval=0.2,  # 测试时使用的第二名专家阈值
        capacity_factor_train=1.25,  # 每个专家网络在单个批次中的固定容量，需要额外的容量以防门控不平衡
        capacity_factor_eval=2.,  # capacity_factor_* 应设置为 >=1 的值
        loss_coef=1e-2  # 辅助专家平衡辅助损失的乘数
    )
    inputs = torch.randn(4, 1024, 512)
    out, aux_loss = moe(inputs)  # (4, 1024, 512), (1,)
    print(out.shape, aux_loss.shape)
```



## 三.使用方式(以chatglm2为例)

**基本流程：**

```shell
pip install -r requirements.txt -i https://pypi.douban.com/simple
```

​	**1. 定义config参数和配置、加载数据集（其他参数列表参考第三部分）；**

​	chatglm2参数下载：https://huggingface.co/THUDM/chatglm2-6b；

​	chatglm2数据下载：https://cloud.tsinghua.edu.cn/f/b3f119/a008264b1cabd1/?dl=1

​	**2.编写加载prompt、定义模型（训练和验证函数）**

​	**3.载入数据，启动训练和验证，调试代码，调试完成！保存修改脚本到examples，处理下一个**

**快速启动：** 将examples/basic/glm/basic_language_model_chatglm2.py复制到根目录下，添加断点，启动调试！

```python
# -*- coding: utf-8 -*- 
"""
    @Project ：quickllm 
    @File    ：quickly_start.py
    @Author  ：ys
    @Time    ：2023/12/12 12:12
    官方项目：https://github.com/THUDM/ChatGLM2-6B
"""

import os
import torch
from loguru import logging
from transformers import AutoTokenizer

from quickllm.models import build_transformer_model
from quickllm.generation import SeqGeneration


class ExpertModel:

    def __init__(self):
        self.prompt = "请以一位医疗领域知识图谱专家的角色回答以下问题："
        self.choice = 'default'  # default, int4, 32k
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 来自https://huggingface.co/THUDM/chatglm2-6b；
        self.dir_path = "/path/to/my/pretrain_ckpt/glm/chatglm2-6B"
        self.checkpoint_path = [os.path.join(dir_path, i) for i in os.listdir(dir_path) if i.endswith('.bin')]
        # 来自项目中的：examples/basic/glm/chatglm2-6B/quickllm_config.json
        self.config_path = dir_path + '/quickllm_config.json'

    def build_prompt(self, history):
        for query, response in history:
            self.prompt += f"\n\n用户：{query}"
            self.prompt += f"\n\nChatGLM-6B：{response}"
        return self.prompt

    def build_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.dir_path.replace('/', '\\'), trust_remote_code=True)
        if self.choice in {'default', '32k'}:
            encoder = build_transformer_model(config_path=self.config_path,
                                              checkpoint_path=self.checkpoint_path).half().to(device)
        else:
            encoder = build_transformer_model(config_path=self.config_path,
                                              checkpoint_path=self.checkpoint_path).to(device)

        model = SeqGeneration(encoder, tokenizer, start_id=None, end_id=tokenizer.eos_token_id, mode='random_sample',
                              maxlen=2048, default_rtype='logits', use_states=True)
        return model

    def chat(self, query, history):
        prompt = ""
        for i, (old_query, response) in enumerate(history):
            prompt += "[Round {}]\n\n问：{}\n\n答：{}\n".format(i + 1, old_query, response)
        prompt += "[Round {}]\n\n问：{}\n\n答：".format(len(history) + 1, query)

        for response in self.build_model().stream_generate(prompt, topk=50, topp=0.7, temperature=0.95):
            new_history = history + [(query, response)]
            yield response, new_history

    def main(self):
        history = []
        logging.info("----欢迎使用ChatGLM2-6B模型，修改prompt输入内容进行对话，clear清空对话历史，stop终止程序")
        while True:
            query = input("\nQuestion：")
            if query.strip() == "stop":
                break
            if query.strip() == "clear":
                history = []
                print("----已清空历史对话----")
                continue
            for response, history in self.chat(query, history=history):
                print(build_prompt(history), flush=True)

            print(build_prompt(history), flush=True)
            torch.cuda.empty_cache()


if __name__ == '__main__':
    expert_bot = ExpertModel()
    expert_bot.main()
```



**快速微调**： 将examples/llm/task_chatglm2_lora.py文件转移至根目录下，添加断点，启动调试！

**快速部署：** 将examples/serving/basic_simple_web_serving_simbert.py文件转移至根目录下，添加断点，启动调试！



## 三. 预训练权重
- 若无说明则使用权重自带的`pytorch_model.bin`和`config.json`

| 模型分类| 模型名称 | 权重来源| 权重链接 | 备注(若有)|
| ----- | ----- | ----- | ----- | ----- |
| chatglm   |chatglm-6b | THUDM | [github](https://github.com/THUDM/ChatGLM-6B), [v0.1.0](https://huggingface.co/THUDM/chatglm-6b/tree/v0.1.0), [v1.1.0](https://huggingface.co/THUDM/chatglm-6b/tree/v1.1.0), [int8](https://huggingface.co/THUDM/chatglm-6b-int8), [int4](https://huggingface.co/THUDM/chatglm-6b-int4) | [config](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/glm) |
|       |chatglm2-6b | THUDM | [github](https://github.com/THUDM/ChatGLM2-6B), [v2](https://huggingface.co/THUDM/chatglm2-6b), [int4](https://huggingface.co/THUDM/chatglm2-6b-int4), [32k](https://huggingface.co/THUDM/chatglm2-6b-32k) | [config](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/glm) |
|       |chatglm3-6b | THUDM | [github](https://github.com/THUDM/ChatGLM3), [v3](https://huggingface.co/THUDM/chatglm3-6b), [32k](https://huggingface.co/THUDM/chatglm3-6b-32k) | [config](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/glm) |
| bert| bert-base-chinese| 谷歌bert的torch版 | [torch](https://huggingface.co/bert-base-chinese) | [config](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/bert/google@bert-base-chinese/bert4torch_config.json) |
|     | chinese_L-12_H-768_A-12| 谷歌 | [github](https://github.com/google-research/bert), [tf](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip) | [转换命令](https://huggingface.co/docs/transformers/v4.28.1/en/converting_tensorflow_models), [config](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/bert/google@chinese_L-12_H-768_A-12/bert4torch_config.json) |
|     | chinese-bert-wwm-ext| HFL | [tf/torch](https://github.com/ymcui/Chinese-BERT-wwm)，[torch](https://huggingface.co/hfl/chinese-bert-wwm-ext)| |
|     | bert-base-multilingual-cased| huggingface | [torch](https://huggingface.co/bert-base-multilingual-cased) | [config](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/bert/google@bert-base-chinese/bert4torch_config.json) |
|     | macbert | HFL| [tf/torch](https://github.com/ymcui/MacBERT)，[torch](https://huggingface.co/hfl/chinese-macbert-base) | |
|     | wobert| 追一科技| [tf](https://github.com/ZhuiyiTechnology/WoBERT)，[torch_base](https://huggingface.co/junnyu/wobert_chinese_base)，[torch_plus_base](https://huggingface.co/junnyu/wobert_chinese_plus_base) | |
|     | guwenbert| ethanyt |[torch](https://huggingface.co/ethanyt/guwenbert-base) | [config](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/bert/ethanyt@guwenbert-base/bert4torch_config.json)|
|roberta|chinese-roberta-wwm-ext | HFL | [tf/torch](https://github.com/ymcui/Chinese-BERT-wwm)，[torch](https://huggingface.co/hfl/chinese-roberta-wwm-ext) | |
|     |roberta-small/tiny| 追一科技 & UER| [tf](https://github.com/ZhuiyiTechnology/pretrained-models)，[torch](https://huggingface.co/uer) | [转换脚本](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/roberta/convert_roberta-small.py) |
|     |roberta-base-english| huggingface | [torch](https://huggingface.co/roberta-base) | [config](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/roberta/huggingface@roberta-base-english/bert4torch_config.json) |
| albert|albert| brightmart| [tf](https://github.com/brightmart/albert_zh)，[torch](https://huggingface.co/voidful)，[torch](https://github.com/lonePatient/albert_pytorch) | |
| nezha|NEZHA | 华为| [tf](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/NEZHA-TensorFlow)，[torch](https://github.com/lonePatient/NeZha_Chinese_PyTorch) | |
| xlnet|chinese-xlnet | HFL | [tf/torch](https://github.com/ymcui/Chinese-XLNet) | [config](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/xlnet/hfl@chinese-xlnet-base)|
|deberta| Erlangshen-DeBERTa-v2| IDEA | [torch](https://huggingface.co/IDEA-CCNL/Erlangshen-DeBERTa-v2-320M-Chinese/tree/main) | |
| electra|Chinese-ELECTRA | HFL | [tf](https://github.com/ymcui/Chinese-ELECTRA)，[torch](https://huggingface.co/hfl/chinese-electra-base-discriminator) | |
| ernie|ernie | 百度文心| [paddle](https://github.com/PaddlePaddle/ERNIE)，[torch](https://huggingface.co/nghuyong)| |
| roformer|roformer| 追一科技| [tf](https://github.com/ZhuiyiTechnology/roformer)，[torch](https://huggingface.co/junnyu/roformer_chinese_base) | |
|         |roformer_v2 | 追一科技| [tf](https://github.com/ZhuiyiTechnology/roformer-v2)，[torch](https://huggingface.co/junnyu/roformer_v2_chinese_char_base)| |
| simbert|simbert | 追一科技| [tf](https://github.com/ZhuiyiTechnology/simbert)，[torch_base](https://huggingface.co/peterchou/simbert-chinese-base/tree/main) | [转换脚本](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/simbert/convert_simbert.py) |
|        |simbert_v2/roformer-sim | 追一科技| [tf](https://github.com/ZhuiyiTechnology/roformer-sim)，[torch](https://huggingface.co/junnyu/roformer_chinese_sim_char_base)| |
| gau|GAU-alpha | 追一科技| [tf](https://github.com/ZhuiyiTechnology/GAU-alpha)| [转换脚本](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/gau/convert_GAU_alpha.py) |
| gpt |CDial-GPT| thu-coai| [torch](https://github.com/thu-coai/CDial-GPT) | [config](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/gpt/thu-coai@CDial-GPT-LCCC-base/bert4torch_config.json) |
| gpt2| cmp_lm(26亿)|清华 | [torch](https://github.com/TsinghuaAI/CPM-1-Generate)| [config](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/gpt2/cpm@cpm_lm_2.6b) |
|     | gpt2-chinese-cluecorpussmall|UER | [torch](https://huggingface.co/uer/gpt2-chinese-cluecorpussmall) | [config](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/gpt2/uer@gpt2-chinese-cluecorpussmall)|
|     | gpt2-ml|imcaspar | [tf](https://github.com/imcaspar/gpt2-ml)，[torch](https://github.com/ghosthamlet/gpt2-ml-torch) | [config](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/gpt2/imcaspar@gpt2-ml_15g_corpus_torch) |
| bart| bart_base_chinese|复旦fnlp| [torch](https://github.com/fastnlp/CPT), [v1.0](https://huggingface.co/fnlp/bart-base-chinese/tree/v1.0), [v2.0](https://huggingface.co/fnlp/bart-base-chinese/tree/v2.0)| [config](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/bart/fnlp@bart-base-chinese/bert4torch_config.json) |
| t5  | t5| UER | [torch](https://huggingface.co/uer/t5-base-chinese-cluecorpussmall)| [config_base](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/t5/uer@t5-base-chinese-cluecorpussmall), [config_small](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/t5/uer@t5-small-chinese-cluecorpussmall)|
|     | mt5 | 谷歌| [torch](https://huggingface.co/google/mt5-base)| [config](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/t5/google@mt5_torch_base)|
|     | t5_pegasus| 追一科技| [tf](https://github.com/ZhuiyiTechnology/t5-pegasus) | [config_base](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/t5/sushen@chinese_t5_pegasus_base_torch), [config_small](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/t5/sushen@chinese_t5_pegasus_small_torch)|
|     | chatyuan v1&v2| clue-ai | [torch](https://github.com/clue-ai/ChatYuan) | [config](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/t5/ClueAI@ClueAI-ChatYuan-large-v1)|
|     | PromptCLUE| clue-ai | [torch](https://github.com/clue-ai/PromptCLUE) | [config](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/t5/ClueAI@ClueAI-ChatYuan-large-v1)|
| llama | llama | facebook| [github](https://github.com/facebookresearch/llama) | [config](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/llama)|
|       | llama-2 | facebook| [github](https://github.com/facebookresearch/llama), [7b](https://huggingface.co/meta-llama/Llama-2-7b-hf), [7b-chat](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf), [13b](https://huggingface.co/meta-llama/Llama-2-13b-hf), [13b-chat](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf) | [config](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/llama)|
|       | chinese_llama_alpaca|HFL|[github](https://github.com/ymcui/Chinese-LLaMA-Alpaca) |[config](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/llama)|
|       | Belle_llama| LianjiaTech| [github](https://github.com/LianjiaTech/BELLE), [7B-2M-enc](https://huggingface.co/BelleGroup/BELLE-LLaMA-7B-2M-enc) | [合成说明](https://github.com/LianjiaTech/BELLE/tree/main/models)、[config](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/llama)|
|       | Ziya | IDEA-CCNL | [v1](https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-v1), [v1.1](https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-v1.1), [pretrain-v1](https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-Pretrain-v1) | [config](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/llama) |
|       | Baichuan | baichuan-inc | [github](https://github.com/baichuan-inc/Baichuan), [7B](https://huggingface.co/baichuan-inc/Baichuan-7B), [13B-Base](https://huggingface.co/baichuan-inc/Baichuan-13B-Base), [13B-Chat](https://huggingface.co/baichuan-inc/Baichuan-13B-Chat) | [config](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/llama) |
|       | Baichuan2 | baichuan-inc | [github](https://github.com/baichuan-inc/Baichuan2), [7B-Base](https://huggingface.co/baichuan-inc/Baichuan2-7B-Base), [7B-Chat](https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat), [13B-Base](https://huggingface.co/baichuan-inc/Baichuan2-13B-Base), [13B-Chat](https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat) | [config](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/llama) |
|       | vicuna | lmsys| [7b-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5) | [config](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/llama)|
|       | Yi | 01-ai| [github](https://github.com/01-ai/Yi), [6B](https://huggingface.co/01-ai/Yi-6B), [6B-200K](https://huggingface.co/01-ai/Yi-6B-200K) | [config](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/llama)|
| bloom |bloom | bigscience | [bloom-560m](https://huggingface.co/bigscience/bloom-560m), [bloomz-560m](https://huggingface.co/bigscience/bloomz-560m) | [config](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/bloom) |
| Qwen  |Qwen | 阿里云 | [github](https://github.com/QwenLM/Qwen-7B), [7B](https://huggingface.co/Qwen/Qwen-7B), [7B-Chat](https://huggingface.co/Qwen/Qwen-7B-Chat) | [config](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/Qwen) |
| InternLM|InternLM | 上海人工智能实验室 | [github](https://github.com/InternLM/InternLM), [7B-Chat](https://huggingface.co/internlm/internlm-chat-7b), [7B](https://huggingface.co/internlm/internlm-7b) | [config](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/internlm) |
| Falcon|Falcon | tiiuae | [hf](https://huggingface.co/tiiuae), [RW-1B](https://huggingface.co/tiiuae/falcon-rw-1b), [7B](https://huggingface.co/tiiuae/falcon-7b), [7B-Instruct](https://huggingface.co/tiiuae/falcon-7b-instruct) | [config](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/falcon) |
| embedding| text2vec-base-chinese |shibing624| [torch](https://huggingface.co/shibing624/text2vec-base-chinese) | |
|          | m3e |moka-ai| [torch](https://huggingface.co/moka-ai) |[config](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/embedding/moka-ai@m3e-base/bert4torch_config.json)|
|          | bge |BAAI| [torch](huggingface.co) |[config](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/embedding/moka-ai@m3e-base/bert4torch_config.json)|

## 五. 鸣谢

- 感谢Tongjilibo的[bert4torch](https://github.com/Tongjilibo/bert4torch)，本实现重点参考了这个项目，进行了优化和更新；项目会持续跟进bert4torch的最新实现

- 感谢苏神实现的[bert4keras](https://github.com/bojone/bert4keras)，有些地方参考了bert4keras的源码，在此衷心感谢大佬的无私奉献；大佬的科学空间

  ```bibtex
  @misc{bert4torch,
    title={bert4torch},
    author={Bo Li},
    year={2022},
    howpublished={\url{https://github.com/Tongjilibo/bert4torch}},
  }
  ```

## 六. 引用

```bibtex
@misc{quickllm,
  title={quickllm},
  author={NLP小讲堂},
  year={2022},
  howpublished={\url{https://github.com/zysNLP/quickllm}},
}
```

## 6. 其他

关注公众号《NLP小讲堂》，更多高效内容及时订阅，最新文章和[视频](https://edu.csdn.net/course/detail/39082)同步，[B站关注](https://www.bilibili.com/video/BV1hG411e7Ng/?spm_id_from=333.999.0.0&vd_source=9a2f107418c10b543b13cbd8e1f9e98d)：

[《浅谈MOE的代码原理（一），是否足够对标self-attention？》](https://mp.weixin.qq.com/s/mbXePBZXIiN3aa8sszPzHQ)参考借鉴：[Mistral Transformers](https://github.com/mistralai/mistral-src)，[Mixture of Expert](https://github.com/lucidrains/mixture-of-experts.git)

《Triton复杂又简单：把部署切成厚厚的薄片。。》参考借鉴：[NGC Triton镜像](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver)，[Triton Inference Server GitHub官网](https://github.com/triton-inference-server/server)

《TensorRT-LLM：大模型推理加速必备》参考借鉴[Qwen-TensorRT原理](https://developer.nvidia.com/zh-cn/blog/qwen-model-support-nvidia-tensorrt-llm)，[Qwen-TensorRT代码](https://github.com/Tlntin/Qwen-TensorRT-LLM/tree/main?tab=readme-ov-file)

