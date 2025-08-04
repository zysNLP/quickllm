来源：https://zhuanlan.zhihu.com/p/671007819

## **什么是[模型量化](https://zhida.zhihu.com/search?content_id=237242934&content_type=Article&match_order=1&q=模型量化&zhida_source=entity)**

模型量化（Quantization）是一种用于通过修改权重的精度来减小大型神经网络（包括大型语言模型）大小的技术。LLM量化之所以能够实现，是因为经验结果表明，虽然与神经网络训练和推理相关的一些操作必须利用高精度，但在某些情况下，可以使用明显较低的精度（例如[INT8](https://zhida.zhihu.com/search?content_id=237242934&content_type=Article&match_order=1&q=INT8&zhida_source=entity)）来减小模型的总体大小，从而允许其使用功能较弱的硬件来运行，同时其能力和准确性可接受地降低。

## **为什么要量化模型**

近年来，神经网络的规模急剧增长，使其能够拥有越来越先进的智能能力。[GPT-4](https://zhida.zhihu.com/search?content_id=237242934&content_type=Article&match_order=1&q=GPT-4&zhida_source=entity)和[Falcon](https://zhida.zhihu.com/search?content_id=237242934&content_type=Article&match_order=1&q=Falcon&zhida_source=entity)等大型语言模型以其理解代码和回答问题的能力而闻名。然而，虽然更大的型号提供了更多的功能，但它通常也需要更昂贵的硬件和更多的硬件资源。[模型蒸馏](https://zhida.zhihu.com/search?content_id=237242934&content_type=Article&match_order=1&q=模型蒸馏&zhida_source=entity)和量化在内的几种技术提供了减小模型大小的方法。

![img](https://pic1.zhimg.com/v2-4e2f7fbfd2ddf95f4274ea4e5c46da3a_1440w.jpg)

量化是指将连续的无限值映射到一组较小的离散有限值的过程。在LLM的上下文中，它指的是将模型的权重从高精度数据类型转换为低精度数据类型的过程。

神经网络依赖的是计算机中表示为[张量](https://zhida.zhihu.com/search?content_id=237242934&content_type=Article&match_order=1&q=张量&zhida_source=entity)的数据结构的计算模型。张量是由数字填充的多维矩阵，这些数字可以使用诸如[浮点32](https://zhida.zhihu.com/search?content_id=237242934&content_type=Article&match_order=1&q=浮点32&zhida_source=entity)位（单精度）或浮点64位（双精度）之类的变量存储为浮点数。

![img](https://pic4.zhimg.com/v2-89a7f9688e7fc8e8ee9bd8fcf52a6329_1440w.jpg)

一般来说，在神经网络中使用高精度与更好的精度和更稳定的训练有关。使用高精度在计算上也更昂贵，因为它需要更多和更昂贵的硬件。谷歌和英伟达就某些神经网络操作使用较低精度可能性的研究表明，较低精度可以用于某些训练和推理操作。

除了研究之外，两家公司都开发了硬件和框架来支持较低精度的操作。例如，英伟达T4加速器是采用Tensor核心技术的低精度GPU，其效率明显高于K80。谷歌的TPU引入了bfloat16的概念，这是一种为神经网络优化的特殊原始数据类型。较低精度背后的基本思想是，神经网络并不总是需要使用64位浮点的所有范围才能使其表现良好。

说到这里就不得不说一下深度学习领域中的数据类型问题。

### **神经网络中的数据类型**

![img](https://pic1.zhimg.com/v2-70bdf73b7ee48aedde40f65f4d9ada8a_1440w.jpg)

- **FP32**

在深度学习中，单精度浮点数格式FP32是一种广泛使用的数据格式，其可以表示很大的实数范围，足够深度学习训练和推理中使用。这种格式使用4个bytes（32bits）表示。

- **Tensor Float 32**

Tensor Float 32是Tensor Core支持新的数值类型，从NVIDIA A100中开始支持。A100的普通FP32的峰值计算速度为19.5TOPs，而TF32的峰值计算速度为156TOPs，提升了非常多。

在深度学习中，其实我们对浮点数的表示范围比较看重，而有效数字不是那么重要。在这个前提下，TF直接就把FP32中23个分数值截短为10bits，而指数位仍为8bits，总长度为19(=1+8+10)bits。至于为什么是10bits 就够了，那是因为FP16就只有10bits用来表示分数值。而在实际测试中，FP16的精度水平已经足够应对深度学习负载，只是表示的范围不够广而已。

- **FP16**

FP16是一种半精度浮点格式，深度学习有使用FP16而不是FP32的趋势，因为较低精度的计算对于神经网络来说似乎并不重要。额外的精度没有任何作用，同时速度较慢，需要更多内存并降低通信速度。

- **BFLOAT16**

由Google开发的16位浮点格式称为“Brain Floating Point Format”，简称“bfloat16”。这个名字来源于“Google Brain”，这是谷歌的一个人工智能研究小组。

FP16设计时并未考虑深度学习应用，其动态范围太窄。BFLOAT16解决了这个问题，提供与FP32相同的动态范围。其可以认为是直接将FP32的前16位截取获得的，现在似乎也有取代FP16的趋势。

## **量化是如何缩小模型的？**

我们发现不使用4字节FP32精度转而使用2字节BF16/FP16半精度可以获得几乎相同的推理结果，同时模型大小会减半。这促使我们想进一步削减内存，如果再从2字节半精度转成仅1字节的8bits数据类型，甚至4bits类型呢？实际上，对于大模型最常见的就是8bits量化(FP8/INT8)和4bits量化(FP4/NF4/INT4)。

量化通过减少每个模型权重所需的位数，显著降低了模型的大小。模型一个典型的场景是将权重从FP16（16位浮点）减少到INT4（4位整数）。同时，在内存中传输时，也显著降低了带宽占用。这允许模型在更便宜的硬件上或以更高的速度运行。通过降低权重的精度，LLM的整体质量也会受到一些影响。

研究表明，这种影响因所使用的技术而异，较大的模型受到精度变化的影响较小。更大的型号（超过70B）即使转换为4bits也能保持其性能。一些技术，如NF4，表明对其性能没有影响。因此，对于这些较大的型号，4bits似乎是性能和大小/速度之间的最佳折衷，而对于较小的型号，8bits量化可能更好。

量化算法可以分为两种：

- **训练后量化（PTQ）**：将已经训练好的模型的权重转换为较低的精度，而无需任何再训练。尽管PTQ简单易实现，但由于权重值的精度损失，它可能会略微降低模型的性能。
- **量化感知训练（QAT）**：与PTQ不同，QAT在训练阶段集成了权重转换过程。这通常不会明显降低模型性能，但对计算的要求更高。QLoRA就是一种高度使用QAT的技术。

下面以`Qwen-7B-Chat`为例展示INT8和INT4量化的效果。

![img](https://pic1.zhimg.com/v2-d3959b5be52dcdab82841e582217a6ac_1440w.jpg)

模型效果↑

![img](https://picx.zhimg.com/v2-b09a136802d97ea1ccd232e969f1c751_1440w.jpg)

推理速度↑

![img](https://picx.zhimg.com/v2-ee01acd69f2960c9564ea23a30f6d5f5_1440w.jpg)

显存使用↓

可以看出不管是推理速度和显存使用，量化后模型都能有不错的改变。

## **值得注意的量化技术**

![img](https://pic4.zhimg.com/v2-b51ce27227a36fb30b452bb5017eb7b5_1440w.jpg)

### **[GGML](https://zhida.zhihu.com/search?content_id=237242934&content_type=Article&match_order=1&q=GGML&zhida_source=entity)**

GGML是一个专注于机器学习的C库。它是由Georgi Gerganov创建的，这就是缩写“GG”的意思。该库不仅提供机器学习的基础元素（例如张量），还提供用于分发LLM的独特二进制格式。

一个值得注意的进展是GGML格式正在过渡到GGUF，它支持使用non-llama模型。GGUF格式经过定制，可扩展且经得起未来考验，同时量化对RAM的要求要轻得多。GGML的主要功能之一是便于加载GGML模型并在CPU上执行它们。尽管现在它也允许将某些层卸载到GPU上，这一增强功能不仅加快了推理速度，还为普通VRAM承载过于庞大的LLM提供了解决方法。

GGML可与`llama.cpp`库无缝协作，确保从业者能够有效利用LLM的力量。`llama.cpp`库的主要目标是允许在MacBook上使用INT4量化的LLaMA模型。许多GGML格式的量化模型可以从HuggingFace上找到，并且大多数都是由`TheBlokeAI`推送的。

![img](https://pic4.zhimg.com/v2-a647737093658f6ceeb5a11be79d5ed9_1440w.jpg)

### **[AWQ](https://zhida.zhihu.com/search?content_id=237242934&content_type=Article&match_order=1&q=AWQ&zhida_source=entity)**

AWQ是一种训练后量化（PTQ）方法。它是适用于LLM的高效且准确的低位权重量化（INT3/4），支持指令调整模型和多模态LM。

![img](https://picx.zhimg.com/v2-7a809a1149e3292a697a3966577cfacd_1440w.jpg)

零一万物的`Yi-6B-Chat`和`Yi-34B-Chat`都有使用`AutoAWQ`库实现AWQ量化的INT4量化版本。并且官方给出了量化代码：[https://github.com/01-ai/Yi/tree/main/quantization/awq](https://link.zhihu.com/?target=https%3A//github.com/01-ai/Yi/tree/main/quantization/awq)，读者可以进行尝试。

### **[GPTQ](https://zhida.zhihu.com/search?content_id=237242934&content_type=Article&match_order=1&q=GPTQ&zhida_source=entity)**

GPTQ是一种专门用于GPT Transformers量化的训练后量化（PTQ）方法。具有如下性能：

- **可扩展性：** GPTQ 能够在大约4个GPU小时内压缩大型网络（例如具有1750亿个参数的GPT模型），将位宽减少到每个权重3或4位，同时精度下降非常小。论文指出，随着模型规模的增加，FP16和GPTQ之间的性能差异减小。
- **性能：**该技术使得使用单个GPU对1750亿参数模型运行推理变得可行。
- **推理速度**：与FP16模型相比，GPTQ模型在NVIDIA A100等高端GPU上提供3.25倍的速度提升，在NVIDIA A6000等经济高效的GPU上提供4.5倍的速度提升。

GPTQ只能将模型量化为INT的数据类型，最常用于转换为INT4。下面将使用`AutoGPTQ`库实现GPTQ算法并量化GPT-2模型。在Google Colab上的免费T4就可以尝试该示例。

```text
# !pip install -q auto-gptq transformers
import random

from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from datasets import load_dataset
import torch
from transformers import AutoTokenizer

# Define base model and output directory
model_id = "gpt2"
out_dir = model_id + "-GPTQ"

# Load quantize config, model and tokenizer
quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,
    damp_percent=0.01,
    desc_act=False,
)
model = AutoGPTQForCausalLM.from_pretrained(model_id, quantize_config)
tokenizer = AutoTokenizer.from_pretrained(model_id)
```

量化过程在很大程度上依赖于样本来评估和增强量化的质量。AutoGPT提供了原始模型和新量化模型产生的输出之间的比较方法。提供的样本数量越大，就越有可能进行更准确和有效的比较，从而提高量化质量。

我们使用C4(Colossal Clean Crawled Corpus)数据集来生成我们的样本。从C4数据集中加载1024个样本，对它们进行标记并格式化。

```text
# Load data and tokenize examples
n_samples = 1024
data = load_dataset("allenai/c4", data_files="en/c4-train.00001-of-01024.json.gz", split=f"train[:{n_samples*5}]")
tokenized_data = tokenizer("\n\n".join(data['text']), return_tensors='pt')

# Format tokenized examples
examples_ids = []
for _ in range(n_samples):
    i = random.randint(0, tokenized_data.input_ids.shape[1] - tokenizer.model_max_length - 1)
    j = i + tokenizer.model_max_length
    input_ids = tokenized_data.input_ids[:, i:j]
    attention_mask = torch.ones_like(input_ids)
    examples_ids.append({'input_ids': input_ids, 'attention_mask': attention_mask})
```

然后开始量化过程，并在量化结束后加载模型进行测试。

```text
# Quantize with GPTQ
model.quantize(
    examples_ids,
    batch_size=1,
    use_triton=True,
)

# Save model and tokenizer
model.save_quantized(out_dir, use_safetensors=True)
tokenizer.save_pretrained(out_dir)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Reload model and tokenizer
model = AutoGPTQForCausalLM.from_quantized(
    out_dir,
    device=device,
    use_triton=True,
    use_safetensors=True,
)
tokenizer = AutoTokenizer.from_pretrained(out_dir)

from transformers import pipeline

generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
result = generator("I have a dream", do_sample=True, max_length=50)[0]['generated_text']
print(result)

# I have a dream," she told CNN last week. "I have this dream of helping my mother find her own. But, to tell that for the first time, now that I'm seeing my mother now, just knowing how wonderful it is that
```

### **NF4和bitsandbytes**

NormalFloat数据类型是分位数量化技术的增强，它显示出比4位整数和4位浮点数更好的结果，在信息论意义上是正态分布权重的最佳表示。NF4主要由QLoRA方法使用，以4位精度加载模型以进行微调。

`bitsandbytes`库集成了NF4量化，同时还支持FP4、LLM.int8()量化。`transformers`库已经集成并原生支持了bitsandbytes这个量化库。而且bitsandbytes是量化任何模型的最简单方法之一，因为它不需要量化校准数据及校准过程。任何模型只要含有`torch.nn.Linear`模块，就可以对其进行开箱即用的量化。每当在`transformers`库中添加新架构时，只要其可以用`accelerate`库的`device_map="auto"`加载，用户就可以直接受益于开箱即用的bitsandbytes量化。

```text
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
torch.manual_seed(0)

# Set device to CPU for now
device = 'cpu'

# Load model and tokenizer
model_id = 'gpt2'
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Print model size
print(f"Model size: {model.get_memory_footprint():,} bytes")
# Model size: 510,342,192 bytes
```

GPT-2在FP32中的大小约为487MB。我们使用`bitsandbytes`对模型进行INT8量化并输出其大小。

```text
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_int8 = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', load_in_8bit=True)
print(f"Model size: {model_int8.get_memory_footprint():,} bytes")
# Model size: 176,527,896 bytes
```

有了这行额外的代码，该模型现在几乎小了三倍（168MB vs 487MB）。我们可以比较原始权重和量化权重的分布：

![img](https://pic4.zhimg.com/v2-097fd9430c81053f9f29ffa8f7bc7ab5_1440w.jpg)

在这种情况下，我们看到`-2、-1、0、1、2`等周围的尖峰。这些值对应于INT8格式中存储的参数（非异常值）。可以通过使用`model_int8.parameters()`打印模型的权重来验证它。

以上就是一些流行的权重量化方法，当然还有其他很多不同的量化方法。许多量化库也都支持多种不同的量化策略（例如4bits、5bits和8bits），每种策略都在效率和性能之间提供不同的权衡。

以上就是本文全部内容了，喜欢这篇文章就给个关注吧 。

也欢迎关注我的公众号「ChaosstuffAI」，了解更多关于LLM的知识与AIGC内容！