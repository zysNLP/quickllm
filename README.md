## 一. 简介

LLM学习资源库。使用pytorch和部分Tensorflow2实现，可以 **<u>*本地运行和调试*</u>** 的大模型LLM相关的应用

#### **核心功能**： Mixtral-8x7B、MOE、ChatGLM3、LLaMa2、 BaChuan、Qwen-TensorRT等

再次强调：强调本地调试、代码跳转、快速掌握LLM！basic_llm\*和basic_run\*分别是调试和运行模式

配套课程[《AIGC大模型理论与工业落地实战》](https://edu.csdn.net/course/detail/39082)，[relation](https://github.com/zysNLP/base_course/tree/main)

### 最新更新：
- 20250516：🐯feat:(learnings/qwen25-vl_sft)添加Qwen2.5-VL-Instruct模型训练、推理、部署服务和请求服务
- 20250514：🐯feat:(learnings/unsloth_models)添加《基于unsloth训练与部署实践DeepSeek-R1 法律推理模型》包括训练、推理、部署服务和请求服务）
- 20250225：🐯feat:(learnings/DeepSeek-V3)添加deepseek-V3官方推理代码（可使用learnings/DeepSeek-V3-codes/inference/model.py脚本在本地GPU(大于40G显存)直接调试！）
- 20250224：🐯feat(quickllm/base_rope.py):新增ROPE简易调试代码
- 20250220：🐯feat(learnings/MLA/*):新增deepseek-MLA注意力机制调试代码
- 20250220：🐯feat(learnings/tiny_grpo/*):新增deepseek-grpo调试代码
- 20231225：🐯feat(basic_llm_moe_transformers.py):新增transformers包的Mixtral-8x7B/MOE源代码调试

### **调试方式!!!：**

​		**格式1. 只有一个脚本文件，将examples中的py脚本复制到quickllm文件夹的同级目录（当前项目根目录）**

​		**格式2. 如果调试脚本是个文件夹，推荐将依赖改成相对路径-格式1的形式；或者在需要使用from quickllm导包调用的脚本代码中添加父级目录**

```python
import sys
sys.path.append('/path/to/directory of quickllm')  # quickllm文件夹的父级目录
```



## 二、快速调试Mixtral-7x8B在transformers模块中的MOE核心代码

Time to debug Mixtral-7x8B-instruct-v0.1 models one line by one line immediately!

```python
# -*- coding: utf-8 -*-
"""
    @Project ：quickllm
    @File    ：basic_llm_moe_transformers.py
    @Author  ：ys
    @Time    ：2023/12/21 18:10
    Mixtral-8x7B 模型中的moe部分，以下代码来自官方transformers库
"""

import torch
torch.manual_seed(123)

from quickllm.layers.moe_by_transformers import MixtralConfig
from quickllm.layers.moe_by_transformers import MixtralSparseMoeBlock

config = MixtralConfig()
moe = MixtralSparseMoeBlock(config)

hidden_states = torch.randn(4, 71, 4096)
hidden_states, router_logits = moe(hidden_states)

print(hidden_states.shape, router_logits.shape)
```



## 三.快速finetune方式(以chatglm2为例)

**安装依赖：**

```shell
pip install -r requirements.txt -i https://pypi.douban.com/simple
```

**快速微调**： 将examples/llm/task_chatglm2_lora.py文件转移至根目录下，添加断点，启动调试！

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

**快速部署：** 将examples/serving/basic_simple_web_serving_simbert.py文件转移至根目录下，添加断点，启动调试！



## 三. 预训练权重
- 若无说明则使用[huggingface官网](https://huggingface.co/models)中对应模型的`pytorch_model.bin`和对应config.json

  

## 四.鸣谢

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

## 五. 引用

```bibtex
@misc{quickllm,
  title={quickllm},
  author={NLP小讲堂},
  year={2022},
  howpublished={\url{https://github.com/zysNLP/quickllm}},
}
```

![star-history-2025526](https://github.com/user-attachments/assets/4832ba81-14d0-413b-9036-345585c76ab2)


## 六. 其他

关注公众号《NLP小讲堂》，更多高效内容及时订阅，最新文章和[视频](https://edu.csdn.net/course/detail/39082)同步，[B站关注](https://www.bilibili.com/video/BV1hG411e7Ng/?spm_id_from=333.999.0.0&vd_source=9a2f107418c10b543b13cbd8e1f9e98d)：

《Mixtral-8x7B-Instruct-v0.1的finetune微调实战》：参考借鉴[Aurora](https://github.com/WangRongsheng/Aurora)，[Firefly](https://github.com/yangjianxin1/Firefly)

[《浅谈MOE的代码原理（一），是否足够对标self-attention？》](https://mp.weixin.qq.com/s/mbXePBZXIiN3aa8sszPzHQ)参考借鉴：[Mistral Transformers](https://github.com/mistralai/mistral-src)，[Mixture of Expert](https://github.com/lucidrains/mixture-of-experts.git)

《Triton复杂又简单：把部署切成厚厚的薄片。。》参考借鉴：[NGC Triton镜像](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver)，[Triton Inference Server GitHub官网](https://github.com/triton-inference-server/server)

《TensorRT-LLM：大模型推理加速必备》参考借鉴：[Qwen-TensorRT原理](https://developer.nvidia.com/zh-cn/blog/qwen-model-support-nvidia-tensorrt-llm)，[Qwen-TensorRT代码](https://github.com/Tlntin/Qwen-TensorRT-LLM/tree/main?tab=readme-ov-file)

《LORA论文解读：大模型的低秩适应》参考借鉴：[LORA论文](https://arxiv.org/pdf/2106.09685.pdf)，[LORA论文解读](https://zhuanlan.zhihu.com/p/624576869)
