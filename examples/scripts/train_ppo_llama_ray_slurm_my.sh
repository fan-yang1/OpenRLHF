#!/bin/bash

#SBATCH -p  gpu-a100
#SBATCH -A  a100acct
#SBATCH -J  rlcf
#SBATCH -N 2                       # num of nodes
#SBATCH -t  02:00:00           # wall time
#SBATCH --ntasks-per-node=1        # tasks per node, must be 1 to run ray
#SBATCH --gpus-per-node=2          # gpus per node
#SBATCH --cpus-per-task=24       # cpus per gpu

# #SBATCH --overcommit               # needed for pytorch
# #SBATCH --output=/home/fyang62/slurm_output/%x_%j.out
# #SBATCH --mail-user="yangfanhuster@gmail.com"         #email for reporting
# # SBATCH --mail-type=FAIL           # only send email on failure

# project settings
OPENRLHF_PATH="$HOME/misc/OpenRLHF"
# MOUNT="$OPENRLHF_PATH:/openrlhf,$HOME/.cache:/root/.cache"
# IMAGE_NAME="nvcr.io/nvidia/pytorch:24.07-py3"
RAY_VERSION=2.12.0

LOG_DIR="$OPENRLHF_PATH/logs"
JOBLOG="$LOG_DIR/train_ppo_llama_ray_slurm_my-$SLURM_JOB_ID.log"
echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} started ..." &>> ${JOBLOG}

# launch ray daemon
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST") # Getting the node names
nodes_array=( $nodes )
node_1=${nodes_array[0]}
ip=$node_1

port=6379
ip_head=$ip:$port
export ip_head
echo "IP Head: $ip_head"  &>> ${JOBLOG}

echo "STARTING HEAD at $node_1"  &>> ${JOBLOG}
srun --nodes=1 --ntasks=1 -w "$node_1" bash -c \
    "ray start --head --node-ip-address=$ip --port=$port \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_NODE}" --block" &>> ${JOBLOG} &
sleep 10s

worker_num=$((SLURM_JOB_NUM_NODES)) #number of nodes other than the head node
for ((i = 1; i < worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "STARTING WORKER $i at $node_i"  &>> ${JOBLOG}
    srun --nodes=1 --ntasks=1 -w "$node_i" bash -c \
        "ray start --address "$ip_head" \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_NODE}" --block" &>> ${JOBLOG} &
    sleep 1s;
done

sleep 30s
echo "READY TO SUBMIT RAY JOB"  &>> ${JOBLOG}

# ===== submit ray job =====
# Job start
# srun --overlap --nodes=1 --ntasks=1 -w "$node_1" bash -c \
ray job submit --address=http://localhost:8265 \
    --runtime-env-json='{"working_dir": "/home/fyang62/misc/OpenRLHF", "pip": "/home/fyang62/misc/OpenRLHF/requirements.txt"}' \
    -- python3 -m openrlhf.cli.train_ppo_ray \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 2 \
    --reward_num_nodes 1 \
    --reward_num_gpus_per_node 2 \
    --critic_num_nodes 1 \
    --critic_num_gpus_per_node 2 \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 2 \
    --vllm_num_engines 0 \
    --vllm_tensor_parallel_size 2 \
    --colocate_critic_reward \
    --colocate_actor_ref \
    --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
    --reward_pretrain OpenRLHF/Llama-3-8b-rm-mixture \
    --save_path ./checkpoint/ray-llama-3-8b-rlhf \
    --micro_train_batch_size 8 \
    --train_batch_size 128 \
    --micro_rollout_batch_size 16 \
    --rollout_batch_size 1024 \
    --max_samples 10000 \
    --max_epochs 1 \
    --prompt_max_len 1024 \
    --generate_max_len 1024 \
    --zero_stage 2 \
    --bf16 \
    --actor_learning_rate 5e-7 \
    --critic_learning_rate 9e-6 \
    --init_kl_coef 0.01 \
    --prompt_data OpenRLHF/prompt-collection-v0.1 \
    --input_key context_messages \
    --apply_chat_template \
    --normalize_reward \
    --adam_offload \
    --flash_attn \
    --vllm_sync_backend nccl \
    --gradient_checkpointing \
    --use_wandb False &>> ${JOBLOG}

# --packing_samples \

echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} stopped ..." &>> ${JOBLOG}