import os
os.system('python train.py -v "SCANSR_X4_ixi" --train_yaml "train_SCANSR_X4_ixi.yaml"')

os.system('python train.py -v "SCANSR_X2_ixi" --train_yaml "train_SCANSR_X2_ixi.yaml" --phase "finetune" --ckpt "68"')
# x2图片输入时未进行切片(loss值过拟合)
os.system('python train.py -v "SCANSR_X2_ixi_1" --train_yaml "train_SCANSR_X2_ixi_1.yaml"')

####   k值的消融实验   #####
# k1 [1/2]
os.system('python train.py -v "SCANSR_k1_X4_ixi" --train_yaml "train_SCANSR_k1_X4_ixi.yaml"')
# k2 [1/6,1/3]
os.system('python train.py -v "SCANSR_k2_X4_ixi" --train_yaml "train_SCANSR_k2_X4_ixi.yaml"')
# [SCANSR_k2_X4_ixi], Best ixi PSNR: 37.0104 @ epoch 52
# Elapsed [0:07:52.623532], PSNR: 37.0289, SSIM: 0.9724
# k3 [1/5,1/2]-> 1/5,1/4,1/3,1/2
os.system('python train.py -v "SCANSR_k3_X4_ixi" --train_yaml "train_SCANSR_k3_X4_ixi.yaml"')
# k4 [1/4,2/3]-> 1/4,1/3,1/2,2/3
os.system('python train.py -v "SCANSR_k4_X4_ixi" --train_yaml "train_SCANSR_k4_X4_ixi.yaml"')
# k5 [1/3,3/4] -> 1/3,2/4,3/4
os.system('python train.py -v "SCANSR_k5_X4_ixi" --train_yaml "train_SCANSR_k5_X4_ixi.yaml"')

# k [1/2,4/5] -> 1/2,2/3,3/4,4/5 *** SCANSR

# k6 [2/3,5/6] -> 2/3,3/4,4/5,5/6
os.system('python train.py -v "SCANSR_k6_X4_ixi" --train_yaml "train_SCANSR_k6_X4_ixi.yaml"')
# k7 [1/7,1/4]
os.system('python train.py -v "SCANSR_k7_X4_ixi" --train_yaml "train_SCANSR_k7_X4_ixi.yaml"')
# k8 [1/8,1/5]
os.system('python train.py -v "SCANSR_k8_X4_ixi" --train_yaml "train_SCANSR_k8_X4_ixi.yaml"')


# 不同前馈网络的烧蚀研究
os.system('python train.py -v "SCANSR_FFN_X4_ixi" --train_yaml "train_SCANSR_FFN_X4_ixi.yaml"')
os.system('python train.py -v "SCANSR_MSFM_X4_ixi" --train_yaml "train_SCANSR_MSFM_X4_ixi.yaml"')



