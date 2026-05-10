#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: test.py
# Created Date: Saturday July 3rd 2021
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Sunday, 23rd April 2023 2:56:31 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2021 Shanghai Jiao Tong University
#############################################################



import os
import argparse
from torch.backends import cudnn
from utilities.json_config import readConfig
from utilities.reporter import Reporter
import warnings
import sys

warnings.filterwarnings('ignore')
base_root = os.path.dirname(__file__)
sys.path.insert(0, base_root)

def str2bool(v):
    return v.lower() in ('true')


####################################################################################
# To configure the seting of training\finetune\test
#
####################################################################################
def getParameters():
    parser = argparse.ArgumentParser()
    # general settings
    parser.add_argument('-v', '--version', type=str, default='OmniSR_X4_DIV2K_test',
                        help="version name for train, test, finetune")

    parser.add_argument('-c', '--cuda', type=int, default=0)  # >0 if it is set as -1, program will use CPU
    parser.add_argument('-s', '--checkpoint_epoch', type=int, default=966,
                        help="checkpoint epoch for test phase or finetune phase")
    # test
    parser.add_argument('-t', '--test_script_name', type=str, default='tester_Matlab')
    parser.add_argument('-b', '--batch_size', type=int, default=1)
    parser.add_argument('-n', '--node_ip', type=str, default='localhost')

    parser.add_argument('--test_dataset_name', type=str, default='ixi',
                        choices=['DIV2K', 'B100', 'Urban100', 'Set5', 'Set14', "Manga109",'ixi','BraTs2018'])
    return parser.parse_args()


ignoreKey = [
    "dataloader_workers",
    "log_root_path",
    "project_root",
    "project_summary",
    "project_checkpoints",
    "project_samples",
    "project_scripts",
    "reporter_path",
    "use_specified_data",
    "specified_data_paths",
    "dataset_path",
    "cuda",
    "test_script_name",
    "test_dataloader",
    "test_dataset_path",
    "save_test_result",
    "test_batch_size",
    "node_name",
    "checkpoint_epoch",
    "test_dataset_path",
    "test_dataset_name",
    "patch_test"]


####################################################################################
# This function will create the related directories before the 
# training\fintune\test starts
# Your_log_root (version name)
#   |---summary/...
#   |---samples/... (save evaluated images)
#   |---checkpoints/...
#   |---scripts/...
#
####################################################################################
def createDirs(sys_state):
    # the base dir
    if not os.path.exists(sys_state["log_root_path"]):
        os.makedirs(sys_state["log_root_path"])

    # create dirs
    sys_state["project_root"] = os.path.join(sys_state["log_root_path"],
                                             sys_state["version"])

    project_root = sys_state["project_root"]
    if not os.path.exists(project_root):
        os.makedirs(project_root)

    sys_state["project_summary"] = os.path.join(project_root, "summary")
    if not os.path.exists(sys_state["project_summary"]):
        os.makedirs(sys_state["project_summary"])

    sys_state["project_checkpoints"] = os.path.join(project_root, "checkpoints")
    if not os.path.exists(sys_state["project_checkpoints"]):
        os.makedirs(sys_state["project_checkpoints"])

    sys_state["project_samples"] = os.path.join(project_root, "samples")
    if not os.path.exists(sys_state["project_samples"]):
        os.makedirs(sys_state["project_samples"])

    sys_state["project_scripts"] = os.path.join(project_root, "scripts")
    if not os.path.exists(sys_state["project_scripts"]):
        os.makedirs(sys_state["project_scripts"])

    sys_state["reporter_path"] = os.path.join(project_root, sys_state["version"] + "_report")


def main():
    config = getParameters()
    # speed up the program
    cudnn.benchmark = True

    sys_state = {}

    # set the GPU number
    # if config.cuda >= 0:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = str(config.cuda)

    # read system environment paths
    env_config = readConfig(os.path.join(base_root, 'env/env.json'))
    # env_config = readConfig('env/env.json')
    env_config = env_config["path"]
    sys_state["env_config"] = env_config

    # obtain all configurations in argparse
    config_dic = vars(config)
    for config_key in config_dic.keys():
        sys_state[config_key] = config_dic[config_key]

    # =======================Test Phase=========================#
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    # TODO modify below lines to obtain the configuration
    sys_state["log_root_path"] = os.path.join(base_root,env_config["train_log_root"])

    sys_state["test_samples_path"] = os.path.join(base_root,env_config["test_log_root"],
                                                  sys_state["version"], "samples")

    if not os.path.exists(sys_state["test_samples_path"]):
        os.makedirs(sys_state["test_samples_path"])

    # Create dirs
    createDirs(sys_state)
    config_json = os.path.join(sys_state["project_root"], env_config["config_json_name"])

    # Read model_config.json
    json_obj = readConfig(config_json)
    for item in json_obj.items():
        if item[0] in ignoreKey:
            pass
        else:
            sys_state[item[0]] = item[1]

    # Get checkpoints

    data_key = config.test_dataset_name.lower()
    sys_state["test_dataset_path"] = env_config["test_dataset_paths"][data_key]
    sys_state["test_dataset_names"] = config.test_dataset_name

    # TODO get the checkpoint file path
    sys_state["ckp_name"] = {}

    # Get the test configurations
    sys_state["com_base"] = "train_logs.%s.scripts." % sys_state["version"]

    # make a reporter
    report_path = os.path.join(base_root,env_config["test_log_root"], sys_state["version"],
                               sys_state["version"] + "_report")
    reporter = Reporter(report_path)
    reporter.writeConfig(sys_state)

    # Display the test information
    # TODO modify below lines to display your configuration information
    moduleName = "test_scripts." + sys_state["test_script_name"]
    print("Start to run test script: {}".format(moduleName))
    print("Test version: %s" % sys_state["version"])
    print("Test Script Name: %s" % sys_state["test_script_name"])

    package = __import__(moduleName, fromlist=True)
    testerClass = getattr(package, 'Tester')
    tester = testerClass(sys_state, reporter)
    tester.test()


if __name__ == '__main__':

    main()
#     python test.py -v "OmniSR_X4_DIV2K" -s 994 -t tester_Matlab --test_dataset_name "Urban100"
# Best Urban100 PSNR: 26.5808 @ epoch 89
# Best Urban100 PSNR: 26.6039 @ epoch 102    PSNR: 26.6550, SSIM: 0.8037
# Best Urban100 PSNR: 26.6824 @ epoch 153
# Best Urban100 PSNR: 26.7593 @ epoch 297
# urban100 PSNR: 26.7987, SSIM: 0.8083
# Set5:PSNR: 32.4218, SSIM: 0.8992
# Set14 :PSNR: 28.5967, SSIM: 0.7901
# Manga109 PSNR: 31.2834, SSIM: 0.9188
# BSD100:PSNR: 27.7957, SSIM: 0.7447
# python test.py -v "OmniSR_X4_DIV2K_hat01" -s 153 -t tester_Matlab --test_dataset_name "Urban100"
# 服务器上
# urban100 PSNR: 26.6799, SSIM: 0.8050
# Set5 PSNR: 32.3789, SSIM: 0.8983
# Set14 PSNR: 28.5267, SSIM: 0.7887
# Manga109 PSNR: 31.1607, SSIM: 0.9173
# B100  PSNR: 27.7727, SSIM: 0.7440
