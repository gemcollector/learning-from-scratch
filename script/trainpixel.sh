task_name="KukaReachPixels"
encoder_name="moco"  # vitb-mae-egosoup
agent_name="${encoder_name}_${task_name}"
pretrain_dir=/To/Your/Path


# for mvp, the noramlize should be set to True
python tools/train.py \
        task=${task_name} \
        agent_name=${agent_name} \
        seed=55 \
        sim_device="cuda:7" \
        rl_device="cuda:7" \
        train.encoder.name=${encoder_name} \
        train.encoder.pretrain_dir=${pretrain_dir]} \
        train.encoder.pretrain_type=hoi \
        train.encoder.freeze=True \
        train.learn.max_iterations=501 \
        normalize=False
