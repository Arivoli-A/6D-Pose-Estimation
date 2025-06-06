# main.py for 6D Pose Estimation

import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import utils.misc as util
#import data_utils.samplers as samplers
#from data_utils import build_dataset
from engine import train_one_epoch, pose_evaluate, bop_evaluate
from models import build_model
from evaluation_tools.pose_evaluator_init import build_pose_evaluator
from inference_tools.inference_engine import inference