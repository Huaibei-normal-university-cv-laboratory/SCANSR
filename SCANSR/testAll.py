import os
os.system('python test.py -v "DCAMSR_X4_ixi" -s 95')
# [DCAMSR_X4_ixi], Best ixi PSNR: 36.0094 @ epoch 95
# Elapsed [0:10:22.587154], PSNR: 36.4484, SSIM: 0.9703
os.system('python test.py -v "ECFNet_X4_ixi" -s 96')
# Best ixi PSNR: 35.8770 @ epoch 96
# Elapsed [0:19:16.637890], PSNR: 36.6255, SSIM: 0.9717
os.system('python test.py -v "DCAMSR1_X4_ixi" -s 83')
# [DCAMSR1_X4_ixi], Best ixi PSNR: 37.0091 @ epoch 83
# Elapsed [0:10:21.627154], PSNR: 37.1465, SSIM: 0.9728

os.system('python test.py -v "DCAMSR_X2_ixi" -s 84')
# [DCAMSR_X2_ixi], Best ixi PSNR: 37.5964, SSIM: 0.9788 @ epoch 84
# Elapsed [0:07:58.518153], PSNR: 38.5127, SSIM: 0.9836
# ----------------------------------------
# Best PSNR: 37.3897 @ epoch 55
# Elapsed [0:07:09.157108], PSNR: 38.3386, SSIM: 0.9833
# ---------------------------------------
# Best PSNR: 37.2826 @ epoch 41
# Elapsed [0:07:50.703777], PSNR: 38.2008, SSIM: 0.9829
os.system('python test.py -v "ECFNet_X2_ixi" -s 85')
# Best PSNR: 37.6584 @ epoch 85
# Elapsed [0:11:58.204421], PSNR: 38.3703, SSIM: 0.9834
# ---------------------------------------
# Best PSNR: 37.5908 @ epoch 66
# Elapsed [0:12:03.448304], PSNR: 38.3374, SSIM: 0.9834
# ---------------------------------------7.3324 @ epoch 36
# # Elapsed [0:10:45.529468], PSNR: 38.0721, SSIM: 0.9828
# # ---------------------------------------
# # Best PSNR: 37.5273 @ epoch 40
# # Elapsed [0:10:36.578176], PSNR: 38.2141, SSIM: 0.9829
# Best PSNR: 3
os.system('python test.py -v "DCAMSR1_X2_ixi" -s 81')
# [DCAMSR1_X2_ixi], Best ixi PSNR: 37.9484 @ epoch 81
# Elapsed [0:30:36.689710], PSNR: 38.3998, SSIM: 0.9828  ----------1

# os.system('python test.py -v "DCAMSR2_X2_ixi" -s 97')
# [DCAMSR2_X2_ixi], Best ixi PSNR: 37.9083 @ epoch 97
# Elapsed [0:37:44.780638], PSNR: 38.2095, SSIM: 0.9826

####   k值的消融实验   #####
# k1 [1/2]
os.system('python test.py -v "SCANSR_k1_X4_ixi" -s 96')  #[1/2]
# [SCANSR_k1_X4_ixi], Best ixi PSNR: 36.9544 @ epoch 96
# Elapsed [0:08:08.431938], PSNR: 37.0486, SSIM: 0.9728
# k2 [1/6,1/3]
os.system('python test.py -v "SCANSR_k2_X4_ixi" -s 88')  #[1/6,1/3]
# [SCANSR_k2_X4_ixi], Best ixi PSNR: 37.0104 @ epoch 52
# Elapsed [0:07:52.623532], PSNR: 37.0289, SSIM: 0.9724
# [SCANSR_k2_X4_ixi], Best ixi PSNR: 36.8077 @ epoch 88
# Elapsed [0:07:56.088040], PSNR: 37.1093, SSIM: 0.9728
# k3 [1/5,1/2]
os.system('python test.py -v "SCANSR_k3_X4_ixi" -s 86')  #[1/5,1/2]
# [SCANSR_k3_X4_ixi], Best ixi PSNR: 36.8628 @ epoch 59
# Elapsed [0:07:34.576510], PSNR: 37.0125, SSIM: 0.9727
# [SCANSR_k3_X4_ixi], Best ixi PSNR: 36.9771 @ epoch 86
# Elapsed [0:11:57.900787], PSNR: 37.2513, SSIM: 0.9731
# k4 [1/4,2/3]
os.system('python test.py -v "SCANSR_k4_X4_ixi" -s 95')  #[1/4,2/3]
# [SCANSR_k4_X4_ixi], Best ixi PSNR: 36.8461 @ epoch 95
# Elapsed [0:07:32.965379], PSNR: 36.8528, SSIM: 0.9722
# [SCANSR_k4_X4_ixi], Best ixi PSNR: 36.7925 @ epoch 95
# Elapsed [0:11:54.153143], PSNR: 37.0214, SSIM: 0.9722
# k5 [1/3,3/4]
os.system('python test.py -v "SCANSR_k5_X4_ixi" -s 83')
# [SCANSR_k5_X4_ixi], Best ixi PSNR: 36.9186 @ epoch 86
# Elapsed [0:12:24.917541], PSNR: 37.0011, SSIM: 0.9726
# [SCANSR_k5_X4_ixi], Best ixi PSNR: 36.9615 @ epoch 83
# Elapsed [0:09:41.663686], PSNR: 37.1003, SSIM: 0.9725
# k6 [2/3,5/6]
os.system('python test.py -v "SCANSR_k6_X4_ixi" -s 91')
# [SCANSR_k6_X4_ixi], Best ixi PSNR: 36.7826 @ epoch 91
# Elapsed [0:07:57.774250], PSNR: 36.9967, SSIM: 0.9727

os.system('python test.py -v "SCANSR_k7_X4_ixi" -s 47')
# [SCANSR_k7_X4_ixi], Best ixi PSNR: 37.0122 @ epoch 47
# Elapsed [0:08:53.187949], PSNR: 37.0296, SSIM: 0.9725

os.system('python test.py -v "SCANSR_k8_X4_ixi" -s 68')
# [SCANSR_k8_X4_ixi], Best ixi PSNR: 36.7336 @ epoch 68
# Elapsed [0:07:48.576274], PSNR: 37.0600, SSIM: 0.9724

os.system('python test.py -v "SCANSR_FFN_X4_ixi" -s 59')
# [SCANSR_FFN_X4_ixi], Best ixi PSNR: 36.7468 @ epoch 70
# Elapsed [0:11:08.560492], PSNR: 36.9911, SSIM: 0.9717
# [SCANSR_FFN_X4_ixi], Best ixi PSNR: 36.8415 @ epoch 59
# Elapsed [0:07:07.504181], PSNR: 36.9200, SSIM: 0.9719
os.system('python test.py -v "SCANSR_MSFM_X4_ixi" -s 26')
# [SCANSR_MSFM_X4_ixi], Best ixi PSNR: 36.6695 @ epoch 26
# Elapsed [0:07:25.561948], PSNR: 36.9575, SSIM: 0.9723