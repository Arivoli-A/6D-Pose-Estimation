# ------------------------------------------------------------------------
# PoET: Pose Estimation Transformer for Single-View, Multi-Object 6D Pose Estimation
# Copyright (c) 2022 Thomas Jantos (thomas.jantos@aau.at), University of Klagenfurt - Control of Networked Systems (CNS). All Rights Reserved.
# Licensed under the BSD-2-Clause-License with no commercial use [see LICENSE for details]
# ------------------------------------------------------------------------

import numpy as np
import json
import evaluation_tools.model_tools as model_tools
from evaluation_tools.pose_evaluator import PoseEvaluator
from evaluation_tools.pose_evaluator_lmo import PoseEvaluatorLMO
import os # Add this import for path manipulation and listing directories


# Functions to initialize the PoseEvaluator module
def load_classes(path):
    """
    Load the class information for HOPE.
    For HOPE, class names are derived directly from the .obj model filenames.
    This function now serves as a placeholder. The actual list of classes
    will be dynamically built during the model loading process.
    """
    return {}


def load_model_info(points):
    """
    Computes information about the 3D model, specifically its diameter.
    This is used because HOPE does not provide a pre-computed models_info.json.
    """
    infos = {}
    extents = 2 * np.max(np.absolute(points), axis=0)
    infos['diameter'] = np.sqrt(np.sum(extents * extents))
    infos['min_x'], infos['min_y'], infos['min_z'] = np.min(points, axis=0)
    infos['max_x'], infos['max_y'], infos['max_z'] = np.max(points, axis=0) # Corrected typo: should be max not min
    return infos


def load_models(path, classes_dict):
    """
    Loads 3D object models for the HOPE dataset.
    This function now iterates through the 'meshes/eval' directory,
    loads .obj files, converts their units from centimeters to meters,
    and dynamically computes their diameter.
    
    Args:
        path (str): The path to the directory containing the .obj models (e.g., 'hope-dataset/meshes/eval/').
        classes_dict (dict): A placeholder dictionary for class information (not used directly here for HOPE).
    
    Returns:
        tuple: A tuple containing:
            - models (dict): Dictionary of loaded 3D models keyed by class name.
            - models_info (dict): Dictionary of model information (e.g., diameter) keyed by class name.
            - classes_list (list): Sorted list of discovered class names from the model files.
    """
    models = {}
    models_info = {}
    hope_classes_list = []

    for model_file in os.listdir(path):
        if model_file.endswith(".obj"):
            class_name = os.path.splitext(model_file)[0] # Extracts "AlphabetSoup" from "AlphabetSoup.obj"
            hope_classes_list.append(class_name)
            model_path = os.path.join(path, model_file)

            # Use the new load_obj function from model_tools
            model_data = model_tools.load_obj(model_path)
            
            # HOPE models are in centimeters, convert to meters for consistency in evaluation
            model_data['pts'] = model_data['pts'] / 100.0 # Convert cm to meters
            
            models[class_name] = model_data
            
            # Compute model info (diameter and extents) dynamically from the loaded points
            models_info[class_name] = load_model_info(model_data['pts'])
            
    return models, models_info, sorted(hope_classes_list)


def load_model_symmetry(path, classes_list):
    """
    Loads symmetry information for HOPE objects.
    Since no explicit symmetry file was provided for HOPE, a predefined list
    of commonly symmetric objects from the dataset is used.
    This list should be reviewed and extended based on the actual physical
    properties of the 28 HOPE objects.
    
    Args:
        path (str): Placeholder for a symmetry file path (not used in this version for HOPE).
        classes_list (list): The list of class names discovered from the models.
        
    Returns:
        dict: A dictionary mapping class names to a boolean indicating symmetry (True for symmetric).
    """
    model_symmetry = {}

    # Define common symmetric objects in the HOPE dataset.
    # This list is based on typical object geometries and should be verified.
    hope_symmetric_objects = [
        "AlphabetSoup",      # Cylindrical can
        "BlackOlives",       # Cylindrical can
        "Butter",            # Rectangular block, can be symmetric
        "Coffee",            # Cylindrical can
        "CreamCheese",       # Rectangular box, can be symmetric
        "GreenBeans",        # Cylindrical can
        "HotSauce",          # Cylindrical bottle
        "MacaroniAndCheese", # Rectangular box, can be symmetric
        "Milk",              # Rectangular carton, can be symmetric
        "Mustard",           # Cylindrical bottle
        "OrangeJuice",       # Rectangular carton, can be symmetric
        "Parmesan",          # Cylindrical container
        "Pineapple",         # Cylindrical can
        "Salsa",             # Cylindrical jar
        "TomatoSauce",       # Cylindrical can
        # Add or remove other objects based on visual inspection of their 3D models.
    ]

    for cls_name in classes_list:
        model_symmetry[cls_name] = False # Default to non-symmetric
        if cls_name in hope_symmetric_objects:
            model_symmetry[cls_name] = True

    return model_symmetry


def build_pose_evaluator(args):
    """
    Function to build the Pose Evaluator, adapted for the HOPE dataset.
    
    Args:
        args (object): An object containing dataset configuration attributes,
                       e.g., args.dataset, args.dataset_path, args.models, args.model_symmetry.
                       
    Returns:
        object: An instance of PoseEvaluatorLMO configured for the HOPE dataset.
    """
    # For HOPE, load_classes is a placeholder as class names come from model files.
    classes_dict = load_classes(getattr(args, 'class_info', "")) # Use getattr to handle missing 'class_info' safely

    # args.models should be set to 'meshes/eval/' relative to args.dataset_path
    models_path = os.path.join(args.dataset_path, args.models)
    models, models_info, classes_list = load_models(models_path, classes_dict) # Now returns the actual list of classes

    # args.model_symmetry is optional; if not provided, load_model_symmetry uses defaults.
    symmetries_path = getattr(args, 'model_symmetry', "")
    if symmetries_path:
        symmetries_path = os.path.join(args.dataset_path, symmetries_path)

    model_symmetry = load_model_symmetry(symmetries_path, classes_list) # Pass the list of classes

    # Use PoseEvaluatorLMO for HOPE, as it uses diameter-based ADD/ADD-S thresholds,
    # which is a common and appropriate metric for this type of dataset.
    if args.dataset == 'ycbv':
        evaluator = PoseEvaluator(models, classes_list, models_info, model_symmetry)
    elif args.dataset == 'lmo':
        evaluator = PoseEvaluatorLMO(models, classes_list, models_info, model_symmetry)
    elif args.dataset == 'hope': # Case for the HOPE dataset
        evaluator = PoseEvaluatorLMO(models, classes_list, models_info, model_symmetry)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}. Supported datasets are 'ycbv', 'lmo', 'hope'.")
    return evaluator