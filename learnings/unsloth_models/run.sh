CUDA_VISIBLE_DEVICES=2 uvicorn server_us:app --host 0.0.0.0 --port 8100 --workers 1 &
CUDA_VISIBLE_DEVICES=3 uvicorn server_us:app --host 0.0.0.0 --port 8101 --workers 1 &
CUDA_VISIBLE_DEVICES=4 uvicorn server_us:app --host 0.0.0.0 --port 8102 --workers 1 &


CUDA_VISIBLE_DEVICES=3 vllm serve /quickllm/us_models/DeepSeek-R1-Legal-COT-merged_test \
  --max_model 4096 \
  --port 8100 \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --served-model-name "R1-14B" \
  --gpu-memory-utilization 0.9
