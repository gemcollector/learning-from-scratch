task_name="FrankaReachPixels"
agent_name="try_FrankaReachPixels"
pretrain_dir=/To/Your/Path


python tools/train.py \
        task=${task_name} \
        agent_name=${agent_name} \
        seed=5 \
        sim_device="cuda:0" \
        rl_device="cuda:0" \
        train.encoder.pretrain_dir=${pretrain_dir} \
        train.encoder.pretrain_type=none \
        train.encoder.freeze=False \
        train.learn.max_iterations=2001 \
        DA=True \
        normalize=False \
        drac=True \
        use_convnet=True \
        clip_observations=255
