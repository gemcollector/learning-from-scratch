task_name="FrankaPickPixels"
encoder_name="r3m"  # vits-mae-hoi, moco, r3m
agent_name="${encoder_name}_${task_name}"
iterations=2001
drac=True
use_convnet=True
DA=True
is_image=True
pretrain_dir=/To/Your/Path

python tools/train.py \
        task=${task_name} \
        agent_name=${agent_name} \
        seed=5 \
        sim_device="cuda:5" \
        rl_device="cuda:5" \
        train.encoder.name=${encoder_name} \
        train.encoder.pretrain_dir=${pretrain_dir} \
        train.encoder.pretrain_type=none \
        train.learn.max_iterations=${iterations} \
        DA=${DA} \
        is_image=${is_image} \
        normalize=False \
        drac=${drac} \
        use_convnet=${use_convnet} \
        clip_observations=255
