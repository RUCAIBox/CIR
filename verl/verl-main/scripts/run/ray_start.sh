#export VLLM_ATTENTION_BACKEND=XFORMERS

#ray start --head --node-ip-address job-56c913a7-0ba5-4a3f-993c-3ff962878d82-worker-1 --num-gpus 8 --port 6370 --dashboard-port 6371
#ray start --head --node-ip-address job-56c913a7-0ba5-4a3f-993c-3ff962878d82-worker-0 --num-gpus 8 --port 6380 --dashboard-port 6381
#ray start --head --node-ip-address job-61cf1a7f-2e74-4ccd-8600-8b7f86b825e5-worker-1 --num-gpus 8 --dashboard-port 8236

#ray start --head --node-ip-address job-a9098b9f-eac8-4dd7-9636-c0e77922d5a0-worker-2 --num-gpus 8 --port 6311 --dashboard-port 6312
ray start --head --node-ip-address  --num-gpus 8 --port 6307 --dashboard-port 6308

#ray start --head --node-ip-address job-61cf1a7f-2e74-4ccd-8600-8b7f86b825e5-worker-3 --num-gpus 8 --port 6333 --dashboard-port 6334

#ray start --head --node-ip-address job-61cf1a7f-2e74-4ccd-8600-8b7f86b825e5-worker-4 --num-gpus 8 --port 6365 --dashboard-port 6366