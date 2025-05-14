# DeepSeek-R1-Legal-COT 法律推理模型

## 项目简介

本项目基于 DeepSeek-R1 模型，结合法律领域的复杂思路链（Chain-of-Thought, COT）数据，完成了微调，并提供了推理 API 服务、批量推理工具和本地交互测试脚本。适用于中文法律问答、法律推理、案例分析等场景。

## 目录结构

- `train.py`：模型微调主脚本，包含数据准备、格式化、LoRA 微调、模型保存等流程。
- `valid.py`：本地推理与交互测试脚本，支持命令行输入问题并获得模型回答。
- `server.py`：基于 FastAPI 的在线推理服务，支持批量法律问题推理接口。
- `tool.py`：异步 HTTP 客户端工具，支持批量并发调用推理服务，适合压测和多实例部署。
- `run.sh`：一键启动多实例推理服务脚本，分别监听不同端口并绑定不同 GPU。
- `r1`: 模型微调所需要的预训练参数（model_name = "unsloth/DeepSeek-R1-Distill-Qwen-14B"）从魔塔官网下载
- `DeepSeek-R1-Legal-COT-merged_test/`：训练后模型及分词器保存目录（需训练后生成）。

## 环境依赖

- Python 3.10+
- torch
- transformers
- unsloth
- trl
- datasets
- fastapi
- uvicorn
- aiohttp

建议使用 conda 或 venv 创建独立环境，并通过 `pip install -r requirements.txt` 安装依赖（需自行整理 requirements）。

## 快速开始

### 1. 模型微调

运行 `train.py`，会自动加载预训练模型、准备样例数据并进行 LoRA 微调，最终模型保存在 `DeepSeek-R1-Legal-COT-merged_test/` 目录。

```bash
python train.py
```

### 2. 本地交互测试

使用 `valid.py` 可在命令行下与模型交互，输入法律问题获得推理结果。

```bash
python valid.py
```

### 3. 启动推理服务

使用 `run.sh` 可一键启动多个推理服务实例（需根据实际 GPU 情况调整 CUDA 设备号）。

```bash
bash run.sh
```

服务默认监听 8100、8101、8102 端口，API 路径为 `/legal_cot_batch`。

### 4. 批量推理调用

可通过 `tool.py` 脚本异步并发调用多个服务实例，适合批量测试和压测。

```bash
python tool.py
```


## API 说明

- **POST /legal_cot_batch**
  - 请求体：`{"questions": ["问题1", "问题2", ...]}`
  - 返回体：`{"code": 200, "data": {"responses": ["回答1", "回答2", ...]}, "message": "success", "elapsed": 123.45}`

## 数据与训练说明

- 示例数据为手动构造的法律问答及思路链，实际应用可替换为更大规模的法律数据集。
- 支持 LoRA 微调，适合资源有限的场景。
- 训练参数、模型路径等可根据实际需求调整。

## 注意事项

- 需确保 GPU 资源充足，且 CUDA 设备号设置正确。
- 推理服务端口、模型路径等参数可根据实际部署环境修改。
- 若需扩展更多端口或 GPU，可修改 `run.sh` 和相关脚本。