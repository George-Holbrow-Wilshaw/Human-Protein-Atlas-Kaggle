!CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --cfg configs/baselines/vgg26_HPA.yaml \
  --load_ckpt Outputs/vgg26_HPA/Feb17-21-24-05_29f2c8d60d66_step \
  --dataset HPA_test 