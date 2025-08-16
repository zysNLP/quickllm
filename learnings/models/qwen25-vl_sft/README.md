# 1. 安装相关docker镜像：nvcr.io/nvidia/pytorch:25.02-py3

```bash
docker pull nvcr.io/nvidia/pytorch:25.02-py3
```

# 2. 启动docker

```bash
docker run -idt --network host --shm-size=64g --name vllm --restart=always --gpus all -v /data2/users/yszhang/quickllm:/quickllm nvcr.io/nvidia/pytorch:25.02-py3 /bin/bash
```

# 3. 在魔塔中下载相关模型

```bash
pip install modelscope
modelscope download --model Qwen/Qwen2.5-VL-7B-Instruct --local_dir /data2/users/yszhang/quickllm/qwen2.5-vl-instruct
```

# 4.进入docker容器，安装conda环境；下载LLama-Factory

```bash
docker exec -it vllm /bin/bash
cd /quickllm
bash Miniconda3-latest-Linux-x86_64.sh
conda create -n sft python=3.11
conda activate sft

git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

# 5. 启动LLaMA-Factory的web ui

```bash
llamafactory-cli webui
```

# 6. 训练模型、融合lora参数
```bash
# 融合后的模型路径/quickllm/LLaMA-Factory/qwen2.5-mmlm0513

llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path /quickllm/qwen2.5-vl-instruct \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template qwen2_vl \
    --flash_attn auto \
    --dataset_dir data \
    --dataset mllm_demo \
    --cutoff_len 2048 \
    --learning_rate 5e-05 \
    --num_train_epochs 3.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 100 \
    --warmup_steps 0 \
    --packing False \
    --report_to none \
    --output_dir saves/Qwen2.5-VL-7B-Instruct/lora/train_2025-05-16-05-48-02 \
    --bf16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --optim adamw_torch \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0 \
    --lora_target all
```

# 7. 创建conda环境安装vllm/transformers

```bash
conda create -n vllm python=3.11
conda activate vllm
pip install vllm -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install transformers -i https://pypi.tuna.tsinghua.edu.cn/simple
```

# 8. 启动vllm+fastapi服务

```bash
python server.py
```

# 9. 请求服务

```bash
python req.py
```