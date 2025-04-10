set -x 
sudo nvidia-smi -i 0 -c 0  # Set GPU to Default (Shared) Mode
sudo nvidia-smi -i 1 -c 0  # Set GPU to Default (Shared) Mode

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "/home/fyang62/misc/OpenRLHF"}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 2 \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 2 \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node 1 \
   --colocate_actor_ref \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 1 \
   --pretrain Qwen/Qwen2.5-7B-Instruct-1M \
   --critic_pretrain Qwen/Qwen2.5-0.5B-Instruct \
   --remote_rm_url /home/fyang62/misc/OpenRLHF/reward_func.py \
   --save_path ./checkpoint/Qwen2.5-7B-Instruct-1M-logic \
   --micro_train_batch_size 8 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 16 \
   --rollout_batch_size 1024 \
   --max_samples 100000 \
   --max_epochs 1 \
   --prompt_max_len 1024 \
   --generate_max_len 1024 \
   --zero_stage 2 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 0.001 \
   --prompt_data data/kk_train \
   --input_key quiz \
   --input_template "<|im_start|>system\nYou are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>.  Now the user asks you to solve a logical reasoning problem. After thinking, when you finally reach a conclusion, clearly state the identity of each character within <answer> </answer> tags. i.e., <answer> (1) Zoey is a knight\n(2) ... </answer>.\n<|im_end|>\n<|im_start|>user\n{}\n<|im_end|>\n<|im_start|>assistant\n<think>" \
   --normalize_reward \
   --adam_offload \
   --flash_attn \
   --packing_samples \
   --gradient_checkpointing \
   --use_wandb False


# --reward_num_nodes 1 \
# --reward_num_gpus_per_node 1 \
# --colocate_critic_reward \
# --reward_pretrain Qwen/Qwen2.5-0.5B-Instruct \
# --apply_chat_template \
# --load_checkpoint \

# --runtime-env-json='{"setup_commands": ["pip install openrlhf[vllm]"]}' [Install deps]
# --ref_reward_offload [Offload to CPU]
# --remote_rm_url http://localhost:5000/get_reward

# --vllm_sync_backend nccl (Only for multi-nodes with vLLM 0.6.4+ or vLLM 0.4.2)

sudo nvidia-smi -i 0 -c 3  # Set GPU to Exclusive_Process Mode
sudo nvidia-smi -i 1 -c 3  # Set GPU to Exclusive_Process Mode

# ray start --head --num-gpus 2
# ray start --address {MASTER-NODE-ADDRESS}:6379  --num-gpus 2
