!CUDA_VISIBLE_DEVICES=0 python tools/train_net_step.py --dataset HPA \
  --cfg configs/baselines/vgg26_HPA.yaml --bs 1 --nw 4 --iter_size 1 