



# checkpoint=--checkpoint-activations
TP=2
HIDDEN=2048
HEAD=16
NLAYERS=24


GLOBAL_BATCH=576
MICRO_BATCH=48

# DS_CONFIG=ds_config.json


# cat <<EOT > $DS_CONFIG
# {
#   "train_batch_size" : $GLOBAL_BATCH,
#   "train_micro_batch_size_per_gpu": $MICRO_BATCH,
#   "steps_per_print": 1,
#    "amp":{
#     "enabled":true, 
#     "opt_level": "O3"
#    },
#   "wall_clock_breakdown" : true
# }

# EOT
# 
ds_args=""
# ds_args=" --deepspeed ${ds_args}"
# ds_args=" --no-pipeline-parallel ${ds_args}" 
# ds_args=" --deepspeed_config=$DS_CONFIG ${ds_args}"
# ds_args=" --deepspeed-activation-checkpointing ${ds_args}"
# deepspeed \


OMP_NUM_THREADS=128 torchrun --nproc_per_node 8 --master_port 23333 \
 pretrain_gpt.py \
--data-path /home/guest_01/train_test/datasets/openwiki \
--micro-batch-size $MICRO_BATCH  \
--global-batch-size $GLOBAL_BATCH \
--hidden-size $HIDDEN  \
--num-attention-heads $HEAD  \
--num-layers $NLAYERS  \
--max-position-embeddings 512  \
--seq-length 512  \
--attention-dropout 0.0  \
--hidden-dropout 0.0  \
--lr 6e-4  \
--min-lr 6e-5  \
--lr-warmup-iters 0  \
--adam-beta1 0.9  \
--adam-beta2 0.95  \
--sgd-momentum 0.0  \
--train-iters 10  \
--log-interval 1  \
--eval-iters 1 \
--vocab-file /home/guest_01/train_test/Megatron-DeepSpeed/vocab.json \
--merge-file /home/guest_01/train_test/Megatron-DeepSpeed/merges.txt \
--lr-decay-iters 10 \
--lr-decay-style cosine \
--dataloader-type cyclic \
--initial-loss-scale 32768 \
--tensor-model-parallel-size $TP \
$checkpoint \
${ds_args} \
 --fp16

# --optimizer sgd --sgd-momentum 0.0  \



# --fp16  \


# DATA_PATH=/home/guest_01/train_test/datasets/openwiki
# BASE_PATH=/home/guest_01/train_test/Megatron-DeepSpeed

# DS_CONFIG=ds_config.json

# TP=1
# PP=1
# HIDDEN=4096
# HEAD=32
# NLAYERS=48


# GLOBAL_BATCH=256
# MICRO_BATCH=32
# ZERO_STAGE=3
# checkpoint=--checkpoint-activations
# OUTPUT_DIR=ds_z${ZERO_STAGE}_nl${NLAYERS}_hs${HIDDEN}_gb${GLOBAL_BATCH}_mb${MICRO_BATCH}
# #OUTPUT_DIR=baseline_nl${NLAYERS}_hs${HIDDEN}_gb${GLOBAL_BATCH}_mb${MICRO_BATCH}
# mkdir -p $OUTPUT_DIR

# cat <<EOT > $DS_CONFIG
# {
#   "train_batch_size" : $GLOBAL_BATCH,
#   "train_micro_batch_size_per_gpu": $MICRO_BATCH,
#   "steps_per_print": 1,
#   "zero_optimization": {
#     "stage": $ZERO_STAGE
#   },
#   "fp16": {
#     "enabled": true,
#     "initial_scale_power": 12
#   },
#   "wall_clock_breakdown" : true
# }

# EOT


# #   "optimizer": {
# #     "type": "Adam",
# #     "params": {
# #       "lr": 0.001,
# #       "betas": [
# #         0.8,
# #         0.999
# #       ],
# #       "eps": 1e-8,
# #       "weight_decay": 3e-7
# #     }
# #   }

# export NCCL_DEBUG=warn 

# ds_args=""
# ds_args=" --deepspeed ${ds_args}"
# ds_args=" --no-pipeline-parallel ${ds_args}" 
# ds_args=" --deepspeed_config=$DS_CONFIG ${ds_args}"
# ds_args=" --zero-stage=$ZERO_STAGE ${ds_args}"
# # ds_args=" --deepspeed-activation-checkpointing ${ds_args}"


# # OMP_NUM_THREADS=128 torchrun --nproc_per_node 8 --master_port 23333 \

# deepspeed \
#  pretrain_gpt.py \
#     --tensor-model-parallel-size $TP \
#     --pipeline-model-parallel-size $PP \
#     --num-layers $NLAYERS \
#     --hidden-size $HIDDEN \
#     --num-attention-heads $HEAD \
#     --seq-length 512 \
#     --loss-scale 12 \
#     --max-position-embeddings 512 \
#     --micro-batch-size $MICRO_BATCH \
#     --global-batch-size $GLOBAL_BATCH \
#     --train-iters 10 \
#     --lr 6.0e-5 \
#     --min-lr 6.0e-6 \
#     --lr-decay-style cosine \
#     --log-interval 1 \
#     --eval-iters 1 \
#     --data-path $DATA_PATH \
#     --vocab-file $BASE_PATH/vocab.json \
#     --merge-file $BASE_PATH/merges.txt \
#     --save-interval 1000 \
#     --split 98,2,0 \
#     --clip-grad 1.0 \
#     --weight-decay 0.1 \
#     --adam-beta1 0.9 \
#     --adam-beta2 0.95 \
#     --init-method-std 0.006 \
#     --fp16 \
#     --dataloader-type cyclic \
#     --tensorboard-dir $OUTPUT_DIR \
#     $ds_args \
#     --exit-interval 5000 \
#     $checkpoint \
#     | tee ${OUTPUT_DIR}/output.log


# # #     # --checkpoint-activations \

# # #     # --dataloader-type cyclic \
# # #     # --optimizer sgd --sgd-momentum 0.0 \
