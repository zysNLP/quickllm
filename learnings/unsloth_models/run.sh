CUDA_VISIBLE_DEVICES=2 uvicorn server_us:app --host 0.0.0.0 --port 8100 --workers 1 &
CUDA_VISIBLE_DEVICES=3 uvicorn server_us:app --host 0.0.0.0 --port 8101 --workers 1 &
CUDA_VISIBLE_DEVICES=4 uvicorn server_us:app --host 0.0.0.0 --port 8102 --workers 1 &