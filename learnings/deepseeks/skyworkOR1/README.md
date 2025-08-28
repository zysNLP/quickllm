下载蒸馏后的模型

huggingface-cli download --resume-download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --local-dir ./
huggingface-cli download --resume-download openbmb/MiniCPM-V-4_5 --local-dir ./MiniCPM-V-4_5

modelscope download --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B  --local_dir ./DeepSeek-R1-Distill-Qwen-1.5B
modelscope download --model OpenBMB/MiniCPM-V-4_5 --local_dir ./MiniCPM-V-4_5