#!/usr/bin/env bash

set -x

# set dist args
# SINGLE=1
nproc_per_node=${ARNOLD_WORKER_GPU}

if [ ! -z "$SINGLE" ] && [ "$SINGLE" != "0" ]; then
  echo "[single node alone] SINGLE=$SINGLE"
  nnodes=1
  node_rank=0
  nproc_per_node=1
  master_addr=127.0.0.1
  master_port=12345
else
  MASTER_NODE_ID=0
  nnodes=${ARNOLD_WORKER_NUM}
  node_rank=${ARNOLD_ID}
  master_addr="METIS_WORKER_${MASTER_NODE_ID}_HOST"
  master_addr=${!master_addr}
  master_port="METIS_WORKER_${MASTER_NODE_ID}_PORT"
  master_port=${!master_port}
  ports=(`echo $master_port | tr ',' ' '`)
  master_port=${ports[0]}
fi

echo "[nproc_per_node: ${nproc_per_node}]"
echo "[nnodes: ${nnodes}]"
echo "[node_rank: ${node_rank}]"
echo "[master_addr: ${master_addr}]"
echo "[master_port: ${master_port}]"

# set up envs
export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0


BED=checkpoints
LOCAL_OUT=local_output
mkdir -p $BED
mkdir -p $LOCAL_OUT


export COMPILE_GAN=0
export USE_TIMELINE_SDK=1
export CUDA_TIMER_STREAM_KAFKA_CLUSTER=bmq_data_va
export CUDA_TIMER_STREAM_KAFKA_TOPIC=megatron_cuda_timer_tracing_original_v2
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

wandb offline
exp_name=debug
bed_path=checkpoints/${exp_name}/
data_path='/hdd/jcaicedo/projects/dinov3/zarr_datasets/CHAMMI.zarr'
video_data_path=''
# BSQ VAE: infinity_vae_d16 -> codebook_dim=16 (vae_type=16), 5 down levels => patch_size=16,
# so apply_spatial_patchify must stay 0. Checkpoint is a full training dump; the loader reads
# its ["vae"] key directly, so no conversion is needed.
vae_ckpt='/hdd/jcaicedo/morphem/infinity_morphemv2/bsq_vae_weights/infinity_vae_d16.pth'
local_out_path=$LOCAL_OUT/${exp_name}

rm -rf ${bed_path}
rm -rf ${local_out_path}

torchrun \
--nproc_per_node=${nproc_per_node} \
--nnodes=${nnodes} \
--node_rank=${node_rank} \
--master_addr=${master_addr} \
--master_port=${master_port} \
train.py \
--ep=100 \
--opt=adamw \
--cum=3 \
--sche=lin0 \
--fp16=2 \
--ada=0.9_0.97 \
--tini=-1 \
--tclip=5 \
--flash=0 \
--alng=5e-06 \
--saln=1 \
--cos=1 \
--enable_checkpointing=full-block \
--local_out_path ${local_out_path} \
--task_type='t2i' \
--bed=${bed_path} \
--data_path=${data_path} \
--video_data_path=${video_data_path} \
--exp_name=${exp_name} \
--tblr=6e-3 \
--pn 0.06M \
--model=layer12 \
--lbs=4 \
--workers=8 \
--short_cap_prob 0.5 \
--use_streaming_dataset 1 \
--iterable_data_buffersize 30000 \
--cond_type=morphem \
--morphem_path=CaicedoLab/MorphEm \
--morphem_img_size=224 \
--morphem_normalize=morphem \
--morphem_saturation_noise=1 \
--morphem_maxlen=16 \
--morphem_cond_channels=1 \
--dataset_type=chammi_zarr \
--chammi_read_threads=8 \
--vae_type 16 \
--vae_ckpt=${vae_ckpt} \
--wp 0.00000001 \
--wpe=1 \
--dynamic_resolution_across_gpus 1 \
--enable_dynamic_length_prompt 1 \
--reweight_loss_by_scale 1 \
--add_lvl_embeding_only_first_block 1 \
--rope2d_each_sa_layer 1 \
--rope2d_normalized_by_hw 2 \
--use_fsdp_model_ema 0 \
--always_training_scales 100 \
--use_bit_label 1 \
--zero=2 \
--save_model_iters_freq 100 \
--log_freq=50 \
--checkpoint_type='torch' \
--prefetch_factor=16 \
--noise_apply_strength 0.3 \
--noise_apply_layers 13 \
--apply_spatial_patchify 0 \
--use_flex_attn=False \
--pad=128