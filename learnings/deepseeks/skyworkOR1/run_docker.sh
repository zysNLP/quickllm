docker run -idt \
  --network host \
  --shm-size=64g \
  --name sky \
  --restart=always \
  --gpus all \
  -v /data4/path/skyworks:/workspaces \
  whatcanyousee/verl:vemlp-th2.4.0-cu124-vllm0.6.3-ray2.10-te2.0-megatron0.11.0-v0.0.6 \
  /bin/bash