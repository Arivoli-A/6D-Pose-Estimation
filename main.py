# main.py for 6D Pose Estimation

import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import src.utils.misc as util
#import data_utils.samplers as samplers
#from data_utils import build_dataset

from src.model.deformable_transformer import DeformableTransformer
from src.model.pose_estimation_transformer import PoseEstimation, SetCriterion
from src.model.position_encoding import PositionEncodingSine
from src.model.backbone import backbone_data_parser, Joiner
from src.utils.pose_matcher import HungarianMatcher, PoseMatcher

from src.utils.engine import train_one_epoch, pose_evaluate
from src.utils.dataset_preparation import build_dataset
#from src.utils.evaluation_tools.pose_evaluator_init import build_pose_evaluator
# from src.utils.inference_engine import inference

# Learning parameters
lr = 2e-4  # Learning rate for main optimizer
lr_backbone_names = ["backbone.0"]  # Names of backbone parameters with different LR
lr_backbone = 2e-5  # Learning rate for backbone
lr_linear_proj_names = ['reference_points', 'sampling_offsets']  # Names of linear projection params
lr_linear_proj_mult = 0.1  # Multiplier for LR of linear projection params
batch_size = 16  # Batch size for training
eval_batch_size = 16  # Batch size for evaluation
weight_decay = 1e-4  # Weight decay for optimizer
epochs = 50  # Number of training epochs
lr_drop = 100  # Epoch at which learning rate is dropped
lr_drop_epochs = None  # Specific epochs to drop LR (optional)
clip_max_norm = 0.1  # Max norm for gradient clipping

# Backbone configuration
backbone = 'yolov4'  # Backbone architecture: yolov4, maskrcnn, or fasterrcnn
backbone_cfg = 'configs/ycbv_yolov4-csp.cfg'  # Path to backbone config file
backbone_weights = None  # Path to pretrained backbone weights or None
backbone_conf_thresh = 0.4  # Confidence threshold for backbone detections
backbone_iou_thresh = 0.5  # IOU threshold for non-max suppression (NMS)
backbone_agnostic_nms = False  # Whether to perform class-agnostic NMS
dilation = False  # Replace stride with dilation in last conv block if True
position_embedding = 'sine'  # Positional embedding type ('sine' or 'learned')
position_embedding_scale = 2 * 3.141592653589793  # Scale for positional embedding
num_feature_levels = 3  # Number of feature levels used

# PoET-specific configs
bbox_mode = 'backbone'  # Bounding box mode for query embedding: gt/backbone/jitter
class_mode = 'specific'  # Class mode: specific or agnostic
backbone_type = 'output'
loss = 'split'
nn_layer = 3

# Transformer parameters
enc_layers = 6  # Number of encoder layers
dec_layers = 6  # Number of decoder layers
dim_feedforward = 1024  # Size of feedforward layers in transformer
hidden_dim = 256  # Dimension of transformer embeddings
dropout = 0.1  # Dropout rate in transformer layers
nheads = 8  # Number of attention heads
num_queries = 10  # Number of query slots
dec_n_points = 4  # Number of points for decoder attention
enc_n_points = 4  # Number of points for encoder attention

# Matcher parameters
matcher_type = 'pose'  # Type of matcher used
set_cost_class = 1  # Class coefficient in matching cost
set_cost_bbox = 1  # L1 box coefficient in matching cost
set_cost_giou = 2  # GIoU box coefficient in matching cost

# Loss parameters
aux_loss = False  # Use auxiliary decoding losses (loss at each layer)
translation_loss_coef = 1  # Weight for translation loss
rotation_loss_coef = 1  # Weight for rotation loss

# Dataset parameters
dataset = 'hopev2'  # Dataset to train/evaluate on ('ycbv' or 'lmo')
dataset_path = './dataset/HOPE_20250606_192620'  # Path to dataset root
train_set = "train"  # Dataset split to train on
eval_set = "test"  # Dataset split to evaluate on
synt_background = None  # Directory for synthetic background images (optional)
n_classes = 28  # Number of object classes - for HOPE dataset
jitter_probability = 0.5  # Probability to apply jitter to bounding boxes
rgb_augmentation = False  # Use RGB augmentation during training
grayscale = False  # Use grayscale augmentation during training
strides = [8,16,32]
num_channels = [128, 256, 512]


# Evaluator parameters
eval_interval = 10  # Evaluate model every N epochs
class_info = '/annotations/classes.json'  # Path to JSON with class names
models = '/models_eval/'  # Directory containing 3D object models
model_symmetry = '/annotations/symmetries.json'  # JSON file with class symmetries

# Inference parameters
inference = False  # Run PoET in inference mode
inference_path = None  # Path containing files for inference
inference_output = None  # Directory to save inference results

# Miscellaneous
sgd = False  # Use SGD optimizer if True
save_interval = 5  # Save checkpoint every N epochs
output_dir = './output'  # Directory to save outputs/checkpoints
device = 'cuda'  # Device for training/testing ('cuda' or 'cpu')
seed = 42  # Random seed for reproducibility
resume = ''  # Path to checkpoint to resume training from
start_epoch = 0  # Epoch to start training at
eval_mode = False  # Run model in evaluation mode
eval_bop = False  # Run model in BOP challenge evaluation mode
num_workers = 0  # Number of data loader worker threads
cache_mode = False  # Cache images in memory if True
override_resumed_lr_drop = True
pre_trained_weights = False
pre_trained_weights_location= './pre_train'
pre_train_weight_filename = './pre_train/pre_train_weight_name.txt'

# Learning parameters
learning_args = {
    'lr': lr,
    'lr_backbone_names': lr_backbone_names,
    'lr_backbone': lr_backbone,
    'lr_linear_proj_names': lr_linear_proj_names,
    'lr_linear_proj_mult': lr_linear_proj_mult,
    'batch_size': batch_size,
    'eval_batch_size': eval_batch_size,
    'weight_decay': weight_decay,
    'epochs': epochs,
    'lr_drop': lr_drop,
    'lr_drop_epochs': lr_drop_epochs,
    'clip_max_norm': clip_max_norm,
    'sgd': sgd,
}

# Backbone configuration
backbone_args = {
    'backbone': backbone,
    'backbone_cfg': backbone_cfg,
    'backbone_weights': backbone_weights,
    'backbone_conf_thresh': backbone_conf_thresh,
    'backbone_iou_thresh': backbone_iou_thresh,
    'backbone_agnostic_nms': backbone_agnostic_nms,
    'dilation': dilation,
    'position_embedding': position_embedding,
    'position_embedding_scale': position_embedding_scale,
    'num_feature_levels': num_feature_levels,
    'strides': strides,
    'num_channels' : num_channels,
}

# PoET-specific configs
poet_args = {
    'bbox_mode': bbox_mode,
    'num_queries': num_queries,
    'num_feature_levels': num_feature_levels,
    'n_classes': n_classes,
    'backbone_type' : backbone_type,
    'loss' : loss,
    'n_nn_layer': nn_layer,
}

# Transformer parameters
transformer_args = {
    'num_encoder_layers': enc_layers,
    'num_decoder_layers': dec_layers,
    'dim_feedforward': dim_feedforward,
    'd_model': hidden_dim,
    'dropout': dropout,
    'nhead': nheads,
    'num_feature_levels': num_feature_levels,
    'dec_n_points': dec_n_points,
    'enc_n_points': enc_n_points,
}

# Matcher parameters
matcher_args = {
    'cost_class': set_cost_class,
    'cost_bbox': set_cost_bbox,
    'cost_giou': set_cost_giou,
}

# Loss parameters
loss_args = {
    'aux_loss': aux_loss,
    'translation_loss_coef': translation_loss_coef,
    'rotation_loss_coef': rotation_loss_coef,
}

# Dataset parameters
dataset_args = {
    'dataset': dataset,
    'dataset_path': dataset_path,
    'train_set': train_set,
    'eval_set': eval_set,
    'synthetic_background': synt_background,
    'n_classes': n_classes,
    'jitter_probability': jitter_probability,
    'rgb_augmentation': rgb_augmentation,
    'grayscale': grayscale,
    'bbox_mode': bbox_mode,
    'loss' : loss,
}

# Evaluator parameters
evaluator_args = {
    'eval_interval': eval_interval,
    'class_info': class_info,
    'models': models,
    'model_symmetry': model_symmetry,
}

# Inference parameters
inference_args = {
    'inference': inference,
    'inference_path': inference_path,
    'inference_output': inference_output,
}

# Miscellaneous parameters
misc_args = {
    'save_interval': save_interval,
    'output_dir': output_dir,
    'device': device,
    'seed': seed,
    'resume': resume,
    'start_epoch': start_epoch,
    'eval': eval_mode,
    'eval_bop': eval_bop,
    'num_workers': num_workers,
    'cache_mode': cache_mode,
    'override_resumed_lr_drop' : override_resumed_lr_drop,
    'pre_trained_weights':pre_trained_weights,
    'pre_trained_weights_location':pre_trained_weights_location,
    'pre_trained_weights_filename' : pre_train_weight_filename,
}


def main():

    device = torch.device(misc_args["device"])
    
    #seed for reproducibility
    seed = misc_args["seed"] 
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    deform_transformer = DeformableTransformer(**transformer_args)
    
    # backbone 
    N_steps = transformer_args['d_model'] // 2
    position_encoding = PositionEncodingSine(N_steps, normalize = True)
    
    backbone_data = backbone_data_parser(backbone_args['strides'],backbone_args['num_channels'])
    backbone = Joiner(backbone_data, position_encoding)

    
    model = PoseEstimation(backbone = backbone, transformer = deform_transformer, **poet_args)
    model.to(device)
    
    matcher = PoseMatcher() # HungarianMatcher(**matcher_args)
    
    weight_dict = {'loss_trans': loss_args['translation_loss_coef'], 'loss_rot': loss_args['rotation_loss_coef']}
    losses = ['translation', 'rotation']
    criterion = SetCriterion(matcher, weight_dict, losses)
    criterion.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    # pose_evaluator = build_pose_evaluator(args) - to implement

    # Build the dataset for training and validation
    dataset_train = build_dataset(data_type =dataset_args['train_set'], dataset_path=dataset_args['dataset_path'],loss_type = dataset_args['loss'])  # Modified
    dataset_val = build_dataset(data_type =dataset_args['eval_set'],dataset_path=dataset_args['dataset_path'],loss_type = dataset_args['loss'])
    
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, learning_args['batch_size'], drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=util.collate_fn, num_workers=misc_args['num_workers'],
                                   pin_memory=True)
    
    data_loader_val = DataLoader(dataset_val, learning_args['eval_batch_size'], sampler=sampler_val,
                                 drop_last=False, collate_fn=util.collate_fn, num_workers=misc_args['num_workers'],
                                 pin_memory=True)

    # lr_backbone_names = ["backbone.0", "backbone.neck", "input_proj", "transformer.encoder"]
    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    for n, p in model.named_parameters():  # n : name, p : parameter
        if "backbone" in n:   
            p.requires_grad = False
        print(n)

    param_dicts = [
        {
            "params":
                [p for n, p in model.named_parameters()
                 if not match_name_keywords(n, learning_args['lr_backbone_names']) and not match_name_keywords(n,learning_args['lr_linear_proj_names']) and p.requires_grad],
            "lr": learning_args['lr'],
        },
        {
            "params": [p for n, p in model.named_parameters() if
                       match_name_keywords(n, learning_args['lr_backbone_names']) and p.requires_grad],
            "lr": learning_args['lr_backbone'],
        },
        {
            "params": [p for n, p in model.named_parameters() if
                       match_name_keywords(n, learning_args['lr_linear_proj_names']) and p.requires_grad],
            "lr": learning_args['lr'] * learning_args['lr_linear_proj_mult'],
        }
    ]
    if learning_args['sgd']:
        optimizer = torch.optim.SGD(param_dicts, lr=learning_args['lr'], momentum=0.9,
                                    weight_decay=learning_args['weight_decay'])
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=learning_args['lr'],
                                      weight_decay=learning_args['weight_decay'])
        
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, learning_args['lr_drop'])

    if misc_args['pre_trained_weights']:
        breakpoint()
        with open(misc_args['pre_trained_weights_filename'], 'r') as file:
            lines = file.readlines()
            last_line = lines[-1].strip() 
        weights_path = f"{misc_args['pre_trained_weights_location']}/{last_line}"
        print(f"Loaded weights from: {weights_path}")
        
        weights = torch.load(weights_path, map_location='cpu')
        missing_keys, unexpected_keys = model.load_state_dict(weights['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))

    output_dir = Path(misc_args['output_dir'])
    # Load checkpoint
    if misc_args['resume']:
        checkpoint = torch.load(misc_args['resume'], map_location='cpu')
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
        if not misc_args['eval'] and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            import copy
            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg['lr'] = pg_old['lr']
                pg['initial_lr'] = pg_old['initial_lr']
            print(optimizer.param_groups)
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            # This is a hack for doing experiment that resume from checkpoint and also modify lr scheduler
            #  (e.g., decrease lr in advance).
            
            if misc_args['override_resumed_lr_drop']:
                print(
                    'Warning: (hack) misc_args[override_resumed_lr_drop] is set to True, so lr_drop would override lr_drop in resumed lr_scheduler.')
                lr_scheduler.step_size = learning_args['lr_drop']
                lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
            lr_scheduler.step(lr_scheduler.last_epoch)
            misc_args['start_epoch'] = checkpoint['epoch'] + 1

    # Evaluate the models performance
    if misc_args['eval']:
        if misc_args['resume']:
            eval_epoch = checkpoint['epoch']
        else:
            eval_epoch = None

        #pose_evaluate(model, matcher, pose_evaluator, data_loader_val, misc_args['eval_set'], dataset_args['bbox_mode'],
         #             args.rotation_representation, device, args.output_dir, eval_epoch)
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(misc_args['start_epoch'], learning_args['epochs']):
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, learning_args['clip_max_norm'])
        lr_scheduler.step()
        if misc_args['output_dir']:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 5 epochs
            if (epoch + 1) % learning_args['lr_drop'] == 0 or (epoch + 1) % misc_args['save_interval'] == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                util.save_on_master({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': [learning_args,backbone_args,poet_args,transformer_args,matcher_args,loss_args,
                                dataset_args,evaluator_args,inference_args,misc_args],
                }, checkpoint_path)

        # Do evaluation on the validation set every n epochs
        if epoch % evaluator_args['eval_interval'] == 0:
           # pose_evaluate(model, matcher, pose_evaluator, data_loader_val, args.eval_set, args.bbox_mode,
             #             args.rotation_representation, device, args.output_dir, epoch)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

        if misc_args['output_dir'] and util.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    print('Evaluate final trained model')
    eval_start_time = time.time()
    #pose_evaluate(model, matcher, pose_evaluator, data_loader_val, args.eval_set, args.bbox_mode,
     #             args.rotation_representation, device, args.output_dir)
    eval_total_time = time.time() - eval_start_time
    eval_total_time_str = str(datetime.timedelta(seconds=int(eval_total_time)))
    print('Evaluation time {}'.format(eval_total_time_str))

if __name__ == "__main__":
    main()