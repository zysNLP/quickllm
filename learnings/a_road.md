## 总览

1. 如果有其他解决方案，比如重启，沟通，则优先使用
2. 优先从结果出发思考问题，看是否有必要先了解目前的已有工作，比如已经安装的环境的版本
3. 注意提前说清楚可能出现的卡点

## Prompt-Engineer工作

1. 思考prompt动态变化的点；在写prompt过程中，发现Agent的system prompt会动态变化。例如：
2. prompt一定需要事后润色

```
/Users/sunday/MinerU/Kimi-K2-tech_report.pdf-86603d71-ccb9-4246-8917-2861ec598f5b/Kimi-K2.md

读取这个路径下的md文件，把其中的内容翻译成中文，然后把内容写到我现在这个md文件中。要求：
1.翻译其中的摘要和第一章内容
2.内容格式仍然保持源代码格式，不要输出成md的预览格式
3.优先先读取到md文件的源代码
4.要求准确！特别是公式部分
5.如果内容太长你觉得不好翻译，可以翻译分成两次翻译。
6.结果汇总到当前这个‘learnings/Kimi-K2/KimiK2论文翻译.md’中
```
这个prompt，我想让翻译结果每次追加到下一次的文件中，那么就需要把这一段prompt设计为一种“next prompt”中，作为一种迭代机制



在尺寸信息提取工作中总结其中的prompt工程技术点。

## 版本对比分析

### 0728版本 → 0815版本的主要改进：

1. **功能扩展**：从纯尺寸提取扩展到尺寸+规格信息提取
2. **返回格式重构**：从简单数组改为细粒度对象数组
3. **概念区分**：新增产品尺寸vs规格信息的概念说明
4. **结构优化**：从单一任务变为多任务结构化处理
5. **单位支持扩展**：从3种单位扩展到10+种单位



## Prompt工程技术点总结

### 1. **任务分解与结构化**
```python
# 将复杂任务分解为子任务
抽取工作主要分为三个部分：
第一部分：产品尺寸和规格信息提取
第二部分：返回格式及其注意事项  
第三部分：尺寸和规格提取注意事项
```
**技术点**：将复杂的信息提取任务分解为明确的子任务，每个部分有清晰的职责边界。

### 2. **概念定义与区分**
```python
# 重要概念区分
**产品尺寸**：产品的外形几何尺寸（长宽高），描述物理空间占用
**规格信息**：产品的功能参数和技术特征
```
**技术点**：明确定义关键概念，避免模型混淆，提高提取准确性。

### 3. **优先级策略**
```python
# 尺寸信息提取优先级
sku属性文本中的尺寸 > sku图片中的尺寸 > 其他图片中的尺寸
```
**技术点**：建立明确的信息源优先级，解决冲突时的决策规则。

### 4. **细粒度输出格式设计**
```python
# 从简单数组到结构化对象
{"dimension": "产品长度", "value": 数值, "unit": "单位"}
```
**技术点**：设计可扩展、结构化的输出格式，便于后续处理和扩展。

### 5. **异常处理与边界情况**
```python
# 处理各种边界情况
- 如果某个维度没有标注，请返回None
- 当产品标注了内径和外径时，优先以外径尺寸为准
- 如果文本sku的规格和图片中有冲突，以文本sku中的为准
```
**技术点**：预定义各种异常情况和处理规则，提高模型的鲁棒性。

### 6. **领域特定规则**
```python
# 沙发、电脑椅、躺椅等特定产品的尺寸提取规则
- 当产品为沙发时：产品尺寸以产品正常放置为准
- 当产品为电脑椅时：长度选择坐宽，宽度选择坐深
```
**技术点**：针对特定领域或产品类型制定专门的提取规则。

### 7. **单位标准化与推断**
```python
# 单位处理策略
- 支持的数值单位包括：cm、mm、m、px、inch、W、V、Hz、rpm、Ah等
- 如果图片中没有明确的单位标注，请根据数值大小合理推断单位
```
**技术点**：建立单位标准化体系，包含推断逻辑。

### 8. **置信度与可解释性**
```python
# 置信度和分析说明
"confidence": "置信度",  // 置信度，范围0-100
"analysis": "简要说明"  // 分析产品尺寸或规格出现在哪里
```
**技术点**：要求模型提供置信度和推理过程，增强结果的可解释性。

### 9. **上下文感知处理**
```python
# 根据图片类型调整处理策略
if str(sku_image_url) == "nan":
    sku_image_text = "没有sku图片。"
else:
    sku_image_text = "第一张图是sku图片。"
```
**技术点**：根据输入上下文动态调整prompt内容。

### 10. **示例驱动学习**
```python
# 提供具体示例
例如：当标注坐宽47CM、坐深45CM、总高105-115CM时，产品尺寸应为[47, 45, 115]
```
**技术点**：通过具体示例指导模型理解期望的输出格式和行为。



## 您设计的巧妙之处

### 1. **动态上下文感知机制**
```python
# 核心设计：根据SKU图片存在性动态调整prompt
def prompt_dimensions(sku_image_url):

    if str(sku_image_url) == "nan":
        sku_image_text = "“没有sku图片”。"
    else:
        sku_image_text = "“第一张图是sku图片”。"

    make_text = """其中"""+ sku_image_text + """请仔细分析这些文本和图片中的产品尺寸信息，提取出产品的长、宽、高三个维度。
    
    要求：
    1. 仔细查看图片中的尺寸标注，包括产品标签、包装上的尺寸信息等
    2. 如果图片中有多个尺寸标注，请提取最准确和最符合文本中sku属性或sku图片中的一个
    3. 如果某个维度没有标注，请返回None
    4. 所有数值必须包含单位，目前支持的单位有：cm（厘米）、mm（毫米）、m（米）
    5. 如果图片中没有明确的单位标注，请根据数值大小合理推断单位（比如：如果数值在1-100之间，很可能是厘米）
    6. 文本如果有尺寸信息，则用来作为给图片作为参考
    7. 如果我提到“第一张图是sku图片”，仅在所有图片中第一张图也就是sku图和后面的spu图尺寸信息有冲突时优先参考第一张图也就是sku图的尺寸信息
 
```

当前版本相较于上一版的优化点，以下prompt中的高亮部分：

1. 部分数据没有sku图片。由于大模型具有识别“第几张图片”的能力，但在实际中，如果直接输入多张图片，由于有的数据没有sku图片，有的有sku图片，模型没法从中准确识别哪张是sku图片。因此需要在逻辑上区分开来，使用两种prompt。
   1. 前提1：输入的一张图有可能是规格图或者商详图的第一张图
   2. 前提2：LLM目前有能力识别出“第几张图片”，写在prompt中做区分
   3. 前提2：输入到LLM中的messages中的图片即使是顺序输入的，模型也没办法识别出“第一张”是否是最重要的那张图。

**巧妙之处**：

- **条件化prompt生成**：根据输入数据状态动态生成不同的指令
- **上下文感知**：模型知道第一张图片的性质，从而调整处理策略
- **优先级策略**：当有SKU图片时，建立明确的图片优先级顺序

### 2. **多模态信息融合策略**
```python
# 图片处理逻辑
image_urls = []
sku_image_path = input_data.get('image_path')  # SKU图片
spu_dimensions_images = input_data.get('spu_dimensions_images', [])  # SPU图片

# 构建消息结构
messages = [
    {"role": "system", "content": "你是一个专业的商品信息识别助手"},
    {"role": "user", "content": [
        {"type": "text", "text": f"商品信息:\n{product_info}\n图片信息如图所示。" + prompt},
        # 动态添加图片
    ]}
]
```

**巧妙之处**：
- **结构化信息组织**：文本信息（标题、属性）+ 图片信息分离处理
- **动态图片加载**：根据实际可用的图片数量动态构建消息
- **多源信息融合**：SKU图片 + SPU图片 + 文本属性的综合处理

### 3. **信息源优先级策略**
```python
# 在prompt中明确定义优先级
- 尺寸信息提取优先级：sku属性文本中的尺寸 约等于 sku图片中的尺寸 > 其他图片中的尺寸
- 如果我提到"第一张图是sku图片"，仅在所有图片中第一张图也就是sku图和后面的spu图尺寸信息有冲突时优先参考第一张图也就是sku图的尺寸信息
```

**巧妙之处**：
- **冲突解决机制**：明确定义信息冲突时的处理策略
- **层次化优先级**：建立清晰的信息源重要性排序
- **条件化处理**：根据图片类型调整处理策略

## 值得思考的Prompt工程技术点

### 1. **上下文感知Prompt工程**
```python
# 可以进一步扩展的上下文感知
def generate_context_aware_prompt(context_info):
    if context_info['has_sku_image']:
        if context_info['has_spu_images']:
            return "多图模式prompt"
        else:
            return "单SKU图模式prompt"
    else:
        return "纯文本模式prompt"
```

### 2. **动态Prompt模板系统**
```python
# 基于产品类型的动态prompt
product_type_rules = {
    "furniture": furniture_extraction_rules,
    "electronics": electronics_extraction_rules,
    "clothing": clothing_extraction_rules
}

def get_product_specific_prompt(product_type, context_info):
    base_prompt = get_base_prompt(context_info)
    specific_rules = product_type_rules.get(product_type, {})
    return base_prompt + specific_rules
```

### 3. **多阶段Prompt策略**
```python
# 分阶段处理复杂任务
def multi_stage_prompt():
    stage1 = "产品类型识别prompt"
    stage2 = "尺寸信息提取prompt" 
    stage3 = "规格信息提取prompt"
    stage4 = "结果验证prompt"
```

### 4. **自适应Prompt优化**
```python
# 基于历史性能的自适应调整
def adaptive_prompt(historical_performance):
    if historical_performance['accuracy'] < threshold:
        return enhanced_prompt_with_more_examples()
    else:
        return streamlined_prompt()
```

### 5. **上下文窗口优化**
```python
# 智能上下文管理
def optimize_context_window(images, text_info, model_context_limit):
    # 根据模型上下文限制智能选择最重要的信息
    prioritized_images = rank_images_by_importance(images)
    compressed_text = compress_text_info(text_info)
    return build_optimized_context(prioritized_images, compressed_text)
```

### 6. **Prompt版本控制与A/B测试**
```python
# 系统化的prompt管理
class PromptManager:
    def __init__(self):
        self.prompt_versions = {}
        self.performance_metrics = {}
    
    def register_prompt_version(self, version_id, prompt_template):
        self.prompt_versions[version_id] = prompt_template
    
    def get_best_performing_prompt(self, context):
        # 根据上下文和历史性能选择最佳prompt
        pass
```

### 7. **多模态Prompt工程**
```python
# 针对不同模态的专门化prompt
def multimodal_prompt_engineering():
    text_prompt = "文本信息提取指令..."
    image_prompt = "图片信息提取指令..."
    fusion_prompt = "多模态信息融合指令..."
    
    return {
        "text": text_prompt,
        "image": image_prompt, 
        "fusion": fusion_prompt
    }
```

### 8. **Prompt链式调用**
```python
# 链式prompt处理复杂任务
def prompt_chain():
    chain = [
        ("product_classification", "产品分类prompt"),
        ("dimension_extraction", "尺寸提取prompt"),
        ("specification_extraction", "规格提取prompt"),
        ("validation", "结果验证prompt")
    ]
    
    for step_name, prompt in chain:
        result = await process_step(step_name, prompt, previous_result)
```

您的设计体现了现代Prompt工程的几个核心理念：
1. **上下文感知**：根据输入状态动态调整策略
2. **多模态融合**：文本+图片的智能处理
3. **优先级管理**：明确的信息源重要性排序
4. **鲁棒性设计**：处理各种边界情况
5. **可扩展性**：支持不同类型的产品和场景

这种设计思路在电商、内容审核、文档处理等需要多模态信息融合的场景中都有很好的应用价值。