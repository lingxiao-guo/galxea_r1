###参数注意事项

* 修改一下 --nproc_per_node 的数字,即可实现多卡训练(>=1即可) . --num_epochs 参数需要注意 (8卡10个epocho大约30min)
* 在train.py中的#save checkpoint and eval  if epoch % 20 == 0 and local_rank == 0: 20表示20个epoch保存一个checkpoint,可以适你num_epochs的参数来修改

torchrun --nproc_per_node=8   galaxea_act/train.py --dataset_dir /mnt/mnt_0/ssk/r1_data/00027_normal --ckpt_dir /data/checkpoints/00027-128-cs100-taoge --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 64 --dim_feedforward 3200 --seed 0 --num_epochs 10 --lr 5e-5 --task_name 00027-128-cs100-taoge --run_name 00027-128-cs100-taoge --arm_type 2 --tf 9d --with_chassis

torchrun --nproc_per_node=8   galaxea_act/train.py --dataset_dir /mnt/mnt_0/ssk/r1_data/00027_normal --ckpt_dir /data/checkpoints/00027-128-cs100-taoge --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 64 --dim_feedforward 3200 --seed 0 --num_epochs 350 --lr 5e-5 --task_name 00027-128-cs100-taoge --run_name 00027-128-cs100-taoge --arm_type 2 --tf 9d --with_chassis