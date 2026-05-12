# @title Setup and Imports { display-mode: "form" }
# @markdown (double click to see the code)


import math
import os
import sys
import random
import glob
import json
import shutil
import base64
import re
from tqdm import tqdm
import chromadb

project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import imageio
import magnum as mn
import numpy as np


from matplotlib import pyplot as plt

# function to display the topdown map
from PIL import Image

import habitat_sim
from habitat_sim.utils import common as utils
from habitat_sim.utils import viz_utils as vut
from habitat.utils.visualizations import maps
import torch
import quaternion
import cv2

# Local application/library specific imports
from src.conceptgraph.utils.general_utils import (
    ObjectClasses,
    measure_time,
    filter_detections,
    mask_iou,
)
from src.conceptgraph.slam.slam_classes import MapObjectDict, DetectionDict, to_tensor
from src.conceptgraph.slam.utils import (
    filter_gobs,
    filter_objects,
    get_bounding_box,
    init_process_pcd,
    denoise_objects,
    merge_objects,
    detections_to_obj_pcd_and_bbox,
    processing_needed,
    resize_gobs,
    merge_obj2_into_obj1,
)
from src.conceptgraph.slam.mapping import (
    compute_spatial_similarities,
    compute_visual_similarities,
    aggregate_similarities,
    match_detections_to_objects,
    merge_obj_matches,
)
from Scripts.System_Prompt import *
from src.eval_utils_gpt_aeqa import call_openai_api_qwen3vlplus
from src.conceptgraph.utils.model_utils import compute_clip_features_batched
from src.conceptgraph.utils.ious import mask_subtract_contained
from src.geom import get_cam_intr, IoU
import supervision as sv
from collections import Counter
from typing import List, Optional, Tuple, Dict, Union
from src.tsdf_planner import SnapShot
from src.hierarchy_clustering import SceneHierarchicalClustering

# Model Imports
try:
    from ultralytics import YOLO
    import clip
except ImportError:
    print("Please install necessary libraries: ultralytics, clip")


# ============================================================================================
# Helper Functions
# ============================================================================================

# Change to do something like this maybe: https://stackoverflow.com/a/41432704
def display_sample(output_path, ix, rgb_obs, semantic_obs=np.array([]), depth_obs=np.array([])):
    from habitat_sim.utils.common import d3_40_colors_rgb
    import time

    os.makedirs(output_path, exist_ok=True)

    if rgb_obs.ndim == 3 and rgb_obs.shape[2] == 4:
        rgb_img = Image.fromarray(rgb_obs, mode="RGBA").convert("RGB")
    elif rgb_obs.ndim == 3 and rgb_obs.shape[2] == 3:
        rgb_img = Image.fromarray(rgb_obs, mode="RGB")
    else:
        rgb_img = Image.fromarray(rgb_obs).convert("RGB")

    arr = [rgb_img]
    titles = ["rgb"]

    plt.figure(figsize=(12, 8))
    for i, data in enumerate(arr):
        ax = plt.subplot(1, len(arr), i + 1)
        ax.axis("off")
        ax.set_title(titles[i])
        ax.imshow(data)
        filename = os.path.join(output_path, f"{titles[i]}_{ix}.png")
        Image.fromarray(np.asarray(data)).convert("RGB").save(filename)
    # plt.show(block=False)
    plt.close() # Close figure to prevent memory leaks in loops

# 使用Matplotlib显示地图
# display a topdown map with matplotlib
def display_map(output_path, topdown_map, key_points=None):
    plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)
    ax.axis("off")
    plt.imshow(topdown_map)
    # plot points on map
    if key_points is not None:
        for point in key_points:
            plt.plot(point[0], point[1], marker="o", markersize=10, alpha=0.8)
    # plt.savefig(os.path.join(output_path, "topdown_map.png"))
    # plt.show(block=False)
    plt.close()

def display_map_2(output_path, topdown_map, key_points=None):
    plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)
    ax.axis("off")
    plt.imshow(topdown_map)
    # plot points on map
    if key_points is not None:
        for point in key_points:
            plt.plot(point[0], point[1], marker="o", markersize=10, alpha=0.8)
    plt.savefig(os.path.join(output_path, "topdown_map_draw.png"))
    # plt.show(block=False)
    plt.close()

def overlay_info_on_image(image, labels_2d, labels_3d):
    img = np.asarray(image)
    if img.ndim == 3 and img.shape[2] > 3:
        img = img[:, :, :3]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    img = np.ascontiguousarray(img)
    text1 = (
        f"2D: {', '.join(labels_2d)}" if labels_2d and len(labels_2d) > 0 else "2D: []"
    )
    text2 = (
        f"3D: {', '.join(labels_3d)}" if labels_3d and len(labels_3d) > 0 else "3D: []"
    )
    cv2.putText(img, text1, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    cv2.putText(img, text2, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    return img

# This function generates a config for the simulator.
def make_cfg(settings):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_id = settings["scene"]
    # sim_cfg.scene_dataset_config_file = settings["scene_dataset"]
    sim_cfg.enable_physics = settings["enable_physics"]

    # Note: all sensors must have the same resolution
    sensor_specs = []

    color_sensor_spec = habitat_sim.CameraSensorSpec()
    color_sensor_spec.uuid = "color_sensor"
    color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    color_sensor_spec.resolution = [settings["height"], settings["width"]]
    color_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    color_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    if "hfov" in settings:
        try:
            color_sensor_spec.hfov = float(settings["hfov"])
        except Exception:
            pass
    sensor_specs.append(color_sensor_spec)

    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    if "hfov" in settings:
        try:
            depth_sensor_spec.hfov = float(settings["hfov"])
        except Exception:
            pass
    sensor_specs.append(depth_sensor_spec)

    semantic_sensor_spec = habitat_sim.CameraSensorSpec()
    semantic_sensor_spec.uuid = "semantic_sensor"
    semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
    semantic_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    if "hfov" in settings:
        try:
            semantic_sensor_spec.hfov = float(settings["hfov"])
        except Exception:
            pass
    sensor_specs.append(semantic_sensor_spec)

    # Here you can specify the amount of displacement in a forward action and the turn angle
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=30.0)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=30.0)
        ),
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])

# convert 3d points to 2d topdown coordinates
def convert_points_to_topdown(pathfinder, points, meters_per_pixel):
    points_topdown = []
    bounds = pathfinder.get_bounds()
    for point in points:
        # convert 3D x,z to topdown x,y
        px = (point[0] - bounds[0][0]) / meters_per_pixel
        py = (point[2] - bounds[0][2]) / meters_per_pixel
        points_topdown.append(np.array([px, py]))
    return points_topdown

# ============================================================================================
# Classes
# ============================================================================================

class DetectionModel:
    def __init__(self, device="cuda"):
        yolo_ckpt_default = "yolov8x-world.pt"
        yolo_ckpt_path = os.path.join(project_root, yolo_ckpt_default)
        self.model = YOLO(yolo_ckpt_path if os.path.exists(yolo_ckpt_path) else yolo_ckpt_default)
        self.model.to(device)
        self.classes = []

    def set_classes(self, classes):
        self.classes = classes
        self.model.set_classes(classes)

    def predict(self, image, conf=0.1, verbose=False):
        return self.model.predict(image, conf=conf, verbose=verbose)

class SimpleSceneGraph:
    def __init__(self, cfg, device="cuda"):
        self.cfg = cfg
        self.device = device
        self.objects = MapObjectDict()
        self.object_id_counter = 1
        self.snapshots = {}
        self.frames = {}
        
        
        # Initialize models
        # print("Initializing models...")
        self.detection_model = DetectionModel(device=device)
        
        # SAM removed per user request; det masks will be omitted

        # CLIP
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
        self.clip_tokenizer = clip.tokenize

        # Object Classes (Dummy or loaded)
        # For simplicity, we define some common classes or load from file if available
        self.obj_classes = ObjectClasses(
            classes_file_path="data/scannet200_classes.txt",
            bg_classes=["wall", "floor", "ceiling"],
            skip_bg=True,
            class_set="scannet200",
        )
        # Manually set some classes for YOLO-World if needed, or rely on default
        # self.obj_classes.classes = ["chair", "table", "sofa", "bed", "plant"] 
        self.detection_model.set_classes(self.obj_classes.get_classes_arr())


    def update(self, rgb, depth, intrinsics, cam_pose, frame_idx, img_path, target_obj_mask=None):
        
        # 1. Detection
        results = self.detection_model.predict(rgb, conf=0.5, verbose=False)
        confidences = results[0].boxes.conf.cpu().numpy()
        detection_class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        xyxy_tensor = results[0].boxes.xyxy
        xyxy_np = xyxy_tensor.cpu().numpy()
        
        detection_class_labels = [
            f"{self.obj_classes.get_classes_arr()[class_id]} {class_idx}"
            for class_idx, class_id in enumerate(detection_class_ids)
        ]

        # 2. Segmentation removed; create rectangular masks from detection boxes
        H, W = rgb.shape[0], rgb.shape[1]
        masks_np = np.zeros((len(xyxy_np), H, W), dtype=np.bool_)
        for i, (x1, y1, x2, y2) in enumerate(xyxy_np):
            xi1 = max(0, min(W - 1, int(x1)))
            yi1 = max(0, min(H - 1, int(y1)))
            xi2 = max(0, min(W, int(np.ceil(x2))))
            yi2 = max(0, min(H, int(np.ceil(y2))))
            if xi2 > xi1 and yi2 > yi1:
                masks_np[i, yi1:yi2, xi1:xi2] = True

        curr_det = sv.Detections(
            xyxy=xyxy_np.astype(np.float32),
            confidence=confidences.astype(np.float32),
            class_id=detection_class_ids.astype(np.int64),
            mask=masks_np,
        )

        if len(curr_det) == 0:
            return rgb, [], [], {}, None

        # 3. Filter Detections
        curr_det, labels = filter_detections(
            image=rgb,
            detections=curr_det,
            classes=self.obj_classes,
            given_labels=detection_class_labels,
            iou_threshold=0.5,
            min_mask_size_ratio=0.0,
            confidence_threshold=0.5,
        )
        
        if curr_det is None or len(curr_det) == 0:
            return rgb, [], [], {}, None
        
        # Calculate best object (highest confidence)
        best_object = None
        if len(curr_det) > 0:
             max_conf_idx = np.argmax(curr_det.confidence)
             best_class_id = curr_det.class_id[max_conf_idx]
             best_object = self.obj_classes.get_classes_arr()[best_class_id]

        # 4. CLIP Features
        image_crops, image_feats, text_feats = compute_clip_features_batched(
            rgb,
            curr_det,
            self.clip_model,
            self.clip_preprocess,
            self.clip_tokenizer,
            self.obj_classes.get_classes_arr(),
            self.device,
        )
        image_feats = image_feats.astype(np.float32)

        raw_gobs = {
            "xyxy": curr_det.xyxy,
            "confidence": curr_det.confidence,
            "class_id": curr_det.class_id,
            "mask": curr_det.mask,
            "classes": self.obj_classes.get_classes_arr(),
            "image_crops": image_crops,
            "image_feats": image_feats,
            "text_feats": text_feats,
            "detection_class_labels": detection_class_labels,
        }
        
        # Resize gobs if needed (skipping for simplicity as rgb is already consistent)
        gobs = raw_gobs # Simplified
        
        # 5. 3D Projection
        obj_pcds_and_bboxes = measure_time(detections_to_obj_pcd_and_bbox)(
            depth_array=depth,
            masks=gobs["mask"],
            cam_K=intrinsics[:3, :3],
            image_rgb=rgb,
            trans_pose=cam_pose,
            min_points_threshold=50,
            spatial_sim_type="iou",
            obj_pcd_max_points=5000,
            device=self.device,
        )
        
        gobs["bbox"] = [obj["bbox"] if obj else None for obj in obj_pcds_and_bboxes]
        gobs["pcd"] = [obj["pcd"] if obj else None for obj in obj_pcds_and_bboxes]
        
        # Make detection list
        detection_list = DetectionDict()
        curr_frame_det_ids = []
        for mask_idx in range(len(gobs["mask"])):
             if gobs["pcd"][mask_idx] is None: continue
             
             curr_class_name = gobs["classes"][gobs["class_id"][mask_idx]]
             curr_class_idx = gobs["class_id"][mask_idx]
             
             detected_object = {
                "id": self.object_id_counter,
                "class_name": curr_class_name,
                "class_id": [curr_class_idx],
                "num_detections": 1,
                "conf": gobs["confidence"][mask_idx],
                "pcd": gobs["pcd"][mask_idx],
                "bbox": gobs["bbox"][mask_idx],
                "clip_ft": to_tensor(gobs["image_feats"][mask_idx]).float(),
                "image": img_path
             }
             detection_list[self.object_id_counter] = detected_object
             curr_frame_det_ids.append(self.object_id_counter)
             self.object_id_counter += 1

        if len(detection_list) == 0:
            det_visualize = sv.Detections(
                xyxy=curr_det.xyxy,
                confidence=curr_det.confidence,
                class_id=curr_det.class_id,
                mask=curr_det.mask,
            )
            labels_vis = [
                f"{self.obj_classes.get_classes_arr()[class_id]} {confidence:0.2f}"
                for class_id, confidence in zip(det_visualize.class_id, det_visualize.confidence)
            ]
            annotated_image = rgb.copy()
            box_annotator = sv.BoundingBoxAnnotator()
            annotated_image = box_annotator.annotate(scene=annotated_image, detections=det_visualize)
            label_annotator = sv.LabelAnnotator()
            annotated_image = label_annotator.annotate(scene=annotated_image, detections=det_visualize, labels=labels_vis)
            if labels is not None:
                detection_labels_2d = []
                for lbl in labels:
                    parts = lbl.rsplit(" ", 1)
                    if len(parts) == 2 and parts[1].isdigit():
                        detection_labels_2d.append(parts[0])
                    else:
                        detection_labels_2d.append(lbl)
            else:
                detection_labels_2d = [
                    self.obj_classes.get_classes_arr()[class_id]
                    for class_id in curr_det.class_id
                ]
            return annotated_image, detection_labels_2d, [], {}, best_object

        # 6. Merge with Map
        if len(self.objects) == 0:
            self.objects.update(detection_list)
        else:
            spatial_sim = compute_spatial_similarities(
                spatial_sim_type="iou",
                detection_list=detection_list,
                objects=self.objects,
                downsample_voxel_size=0.025,
            )
            visual_sim = compute_visual_similarities(detection_list, self.objects)
            agg_sim = aggregate_similarities(
                match_method="sim_sum",
                phys_bias=0.0,
                spatial_sim=spatial_sim,
                visual_sim=visual_sim,
            )
            match_indices = match_detections_to_objects(
                agg_sim=agg_sim,
                detection_threshold=0.5,
                existing_obj_ids=list(self.objects.keys()),
                detected_obj_ids=list(detection_list.keys()),
            )
            for detected_obj_id, existing_obj_match_id in match_indices:
                if existing_obj_match_id is None:
                    self.objects[detected_obj_id] = detection_list[detected_obj_id]
                else:
                    detected_obj = detection_list[detected_obj_id]
                    matched_obj = self.objects[existing_obj_match_id]
                    merged_obj = merge_obj2_into_obj1(
                        obj1=matched_obj,
                        obj2=detected_obj,
                        downsample_voxel_size=0.025,
                        dbscan_remove_noise=True,
                        dbscan_eps=0.1,
                        dbscan_min_points=10,
                        spatial_sim_type="iou",
                        device=self.device,
                        run_dbscan=False,
                    )
                    class_id_counter = Counter(merged_obj["class_id"])
                    most_common_class_id = class_id_counter.most_common(1)[0][0]
                    most_common_class_name = self.obj_classes.get_classes_arr()[
                        most_common_class_id
                    ]
                    merged_obj["class_name"] = most_common_class_name
                    self.objects[existing_obj_match_id] = merged_obj

        # Visualization
        annotated_image = rgb.copy()
        det_visualize = sv.Detections(
                xyxy=gobs["xyxy"],
                confidence=gobs["confidence"],
                class_id=gobs["class_id"],
            )
        # Generate labels for visualization
        labels_vis = [
            f"{self.obj_classes.get_classes_arr()[class_id]} {confidence:0.2f}"
            for class_id, confidence in zip(det_visualize.class_id, det_visualize.confidence)
        ]
        
        box_annotator = sv.BoundingBoxAnnotator()
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=det_visualize)
        label_annotator = sv.LabelAnnotator()
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=det_visualize, labels=labels_vis)
        
        if labels is not None:
            detection_labels_2d = []
            for lbl in labels:
                parts = lbl.rsplit(" ", 1)
                if len(parts) == 2 and parts[1].isdigit():
                    detection_labels_2d.append(parts[0])
                else:
                    detection_labels_2d.append(lbl)
        else:
            detection_labels_2d = [
                self.obj_classes.get_classes_arr()[class_id]
                for class_id in curr_det.class_id
            ]
        detection_labels_3d = []
        for mask_idx in range(len(gobs["mask"])):
            if gobs["pcd"][mask_idx] is None:
                continue
            curr_class_name = gobs["classes"][gobs["class_id"][mask_idx]]
            detection_labels_3d.append(curr_class_name)

        eps = 0.05
        obj_infos_2d = []
        for i in range(len(gobs["xyxy"])):
            curr_class_name = gobs["classes"][gobs["class_id"][i]]
            x1, y1, x2, y2 = gobs["xyxy"][i]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            obj_infos_2d.append((curr_class_name, float(cx / W), float(cy / H)))

        scene_graph = {}
        for i in range(len(obj_infos_2d)):
            li, xi, yi = obj_infos_2d[i]
            if li not in scene_graph:
                scene_graph[li] = {}
            for j in range(len(obj_infos_2d)):
                if i == j:
                    continue
                lj, xj, yj = obj_infos_2d[j]
                dx = xj - xi
                dy = yj - yi
                m = max(abs(dx), abs(dy))
                if m <= eps:
                    direction = "same"
                else:
                    if abs(dx) >= abs(dy):
                        direction = "right" if dx > 0 else "left"
                    else:
                        direction = "down" if dy > 0 else "up"
                scene_graph[li][lj] = direction

        return annotated_image, detection_labels_2d, detection_labels_3d, scene_graph, best_object




def convert_scene_graph_to_text(scene_graph):
    descriptions = []
    for subject, relations in scene_graph.items():
        for object_name, relation in relations.items():
            if relation == "right":
                desc = "to the left of"
            elif relation == "left":
                desc = "to the right of"
            elif relation == "down":
                desc = "above"
            elif relation == "up":
                desc = "below"
            elif relation == "same":
                desc = "at the same position as"
            else:
                continue
            
            descriptions.append(f'"{subject}" is {desc} the "{object_name}"')

    if not descriptions:
        return ""

    # Format: "The spatial relationship where [sent1]. [Sent2]. ..."
    formatted_sentences = [descriptions[0] + "."]
    for d in descriptions[1:]:
        # Capitalize first letter
        d = d[0].upper() + d[1:] + "."
        formatted_sentences.append(d)
        
    return "The spatial relationship where " + " ".join(formatted_sentences)


# ============================================================================================
# Main Function
# ============================================================================================

def main(data_root, EXP_Name, output_path, sim_settings, num_scenes_to_process=None):
    # Initialize ChromaDB
    chroma_db_path = os.path.join(output_path, EXP_Name, "chroma_db")
    if not os.path.exists(chroma_db_path):
        os.makedirs(chroma_db_path)
    chroma_client = chromadb.PersistentClient(path=chroma_db_path)
    collection = chroma_client.get_or_create_collection(name="nav_experiences")

    # Get list of all GLB files in the training directory
    
    scene_files = glob.glob(os.path.join(data_root, "**/*.glb"), recursive=True)
    
    # Filter to keep only basis.glb files if needed, or just all glb files
    scene_files = [f for f in scene_files if "basis.glb" in f]
    
    if num_scenes_to_process is not None:
        if len(scene_files) > num_scenes_to_process:
             print(f"Randomly selecting {num_scenes_to_process} scenes from {len(scene_files)} found.")
             random.shuffle(scene_files)
             scene_files = scene_files[:num_scenes_to_process]
    
    print(f"Found {len(scene_files)} scenes.")
    
    for scene_idx, test_scene in enumerate(tqdm(scene_files, desc="Processing Scenes")):
        # print(f"Processing scene {scene_idx+1}/{len(scene_files)}: {test_scene}")
        
        # Extract scene ID for directory naming
        scene_id = os.path.basename(os.path.dirname(test_scene))
        scene_output_dir = os.path.join(output_path, EXP_Name, f"eps_{scene_id}")
        
        sim_settings["scene"] = test_scene
        cfg = make_cfg(sim_settings)

        from habitat_sim.gfx import LightInfo, LightPositionModel
        
        # Create Simulator
        try:
            # print("Initializing Simulator...")
            sim = habitat_sim.Simulator(cfg)
        except Exception as e:
            print(f"Failed to initialize simulator for scene {test_scene}: {e}")
            continue
        
        # Setup Lighting
        my_scene_lighting_setup = [
                LightInfo(vector=[0.0, 2.0, 0.6, 1.0], model=LightPositionModel.Global)
            ]
        sim.set_light_setup(my_scene_lighting_setup, "my_scene_lighting")

        # Initialize Scene Graph
        # We need to re-initialize scene graph for each scene to clear previous objects
        scene_graph = SimpleSceneGraph(cfg=None, device="cuda" if torch.cuda.is_available() else "cpu")

        # Check pathfinder
        if not sim.pathfinder.is_loaded:
            print("Pathfinder not initialized, skipping scene.")
            sim.close()
            continue

        # Generate 5 random path seeds
        # If len(path_points) < 2, skip and try another seed until we get 5 valid paths
        valid_paths_count = 0
        # We'll try a maximum number of attempts to avoid infinite loops
        max_attempts = 100 
        attempt = 0
        valid_paths = 5
        
        while valid_paths_count < valid_paths and attempt < max_attempts:
            attempt += 1
            path_seed = random.randint(0, 100000)
            
            sim.pathfinder.seed(path_seed)
            sample1 = sim.pathfinder.get_random_navigable_point()
            sample2 = sim.pathfinder.get_random_navigable_point()
            path = habitat_sim.ShortestPath()
            path.requested_start = sample1
            path.requested_end = sample2
            found_path = sim.pathfinder.find_path(path)
            path_points = path.points
            
            if not found_path or len(path_points) < 2 or len(path_points) > 10:
                continue
            
            # Valid path found
            # print(f"  Processing seed {valid_paths_count+1}/5 (seed={path_seed}) with {len(path_points)} points")
            
            # Directory for this seed
            seed_output_dir = os.path.join(scene_output_dir, f"seed_{path_seed}")
            os.makedirs(seed_output_dir, exist_ok=True)
            
            seed_data = [] # To store JSON data for this seed
            
            # Initialize Agent
            agent = sim.initialize_agent(sim_settings["default_agent"])
            
            # Iterate through path points
            path_valid = True # Flag to track path validity
            for ix, point in enumerate(path_points):
                # Check if previous points made the path invalid
                if not path_valid:
                    break

                # Calculate orientation
                if ix < len(path_points) - 1:
                    tangent = path_points[ix + 1] - point
                    # Handle zero length tangent
                    if np.linalg.norm(tangent) < 1e-6:
                        tangent = np.array([0, 0, -1]) # Default
                else:
                    # For the last point, use the same orientation as the previous segment
                    if len(path_points) > 1:
                         tangent = point - path_points[ix - 1]
                    else:
                         tangent = np.array([0, 0, -1])

                tangent_orientation_matrix = mn.Matrix4.look_at(
                    point, point + tangent, np.array([0, 1.0, 0])
                )
                tangent_orientation_q = mn.Quaternion.from_matrix(
                    tangent_orientation_matrix.rotation()
                )
                
                # Define rotations: 0, +120 (clockwise?), -120 (counter-clockwise?)
                # Habitat coordinate system: Y is up, -Z is forward.
                # Rotating around Y axis.
                rotations = [
                    (0, ""),
                    (-120, "_120"),  # Clockwise 120
                    (120, "_-120")   # Counter-clockwise 120
                ]
                
                point_data = {
                    "point_idx": ix,
                    "point_coord": point.tolist(),
                    "views": []
                }
                
                point_dir = os.path.join(seed_output_dir, f"point_{ix}")
                os.makedirs(point_dir, exist_ok=True)

                for angle_deg, suffix in rotations:
                    # Apply rotation
                    rot_quat = mn.Quaternion.rotation(mn.Deg(angle_deg), mn.Vector3(0, 1.0, 0))
                    final_orientation = tangent_orientation_q * rot_quat
                    
                    agent_state = habitat_sim.AgentState()
                    agent_state.position = point
                    agent_state.rotation = utils.quat_from_magnum(final_orientation)
                    agent.set_state(agent_state)
                    
                    observations = sim.get_sensor_observations()
                    rgb = observations["color_sensor"]
                    depth = observations["depth_sensor"]
                    
                    # Process with Scene Graph
                    current_state = agent.get_state()
                    sensor_state = current_state.sensor_states.get("color_sensor", None)
                    if sensor_state is None:
                        sensor_state = current_state
                        
                    cam_pose = np.eye(4)
                    cam_pose[:3, :3] = quaternion.as_rotation_matrix(sensor_state.rotation)
                    cam_pose[:3, 3] = sensor_state.position
                    cam_intrinsics = get_cam_intr(sim_settings["width"], sim_settings["height"], sim_settings.get("hfov", 120))
                    
                    rgb_np = np.asarray(rgb)
                    rgb_vis = rgb_np[..., :3] if rgb_np.ndim == 3 and rgb_np.shape[2] >= 3 else rgb_np
                    
                    # Update scene graph to get labels
                    annotated_img, all_2d_labels, all_3d_labels, pairwise_scene_graph, object = scene_graph.update(
                        rgb_vis,
                        depth,
                        cam_intrinsics,
                        cam_pose,
                        frame_idx=ix, # Use point index as frame index
                        img_path=f"point_{ix}{suffix}",
                        target_obj_mask=None,
                    )
                    # print(f"angle_deg: {angle_deg}, object: {object}")
                    if angle_deg == 0:
                        best_object = object
                    
                    # Convert scene graph to natural language
                    scene_graph_text = convert_scene_graph_to_text(pairwise_scene_graph)
                    
                    # Check validity for the last point's first view (angle 0)
                    # We only check the first view (angle 0) of the last point as per requirement interpretation.
                    # Or should we check ALL views of the last point? The requirement says "last point", implying the location.
                    # However, "analyze whether there are identifiable 2D targets" usually refers to what is seen.
                    # Assuming checking the 0-degree view (forward facing) or ANY view?
                    # Requirement: "After each path ends, analyze if the last point has identifiable 2D targets."
                    # And "If not, delete...". And "(2) Last point has multiple identical targets... delete..."
                    # We will check the view with angle 0 for simplicity as it is the forward view.
                    # Or better: Check if ANY view at the last point has valid targets?
                    # Let's check the primary view (angle 0) for now, as that's the direction of travel.
                    if ix == len(path_points) - 1 and angle_deg == 0:
                        # (1) Check if 2D labels exist
                        if not all_2d_labels:
                            # No 2D targets found
                            path_valid = False
                            # print(f"Invalid path: No 2D targets at last point {ix}, view 0.")
                        # elif 'fan' in all_2d_labels:
                        #     # Fan found
                        #     path_valid = False
                        #     # print(f"Invalid path: Fan found at last point {ix}, view 0.")
                        elif 'power outlet' in all_2d_labels:
                            # Power outlet found
                            path_valid = False
                            # print(f"Invalid path: Power outlet found at last point {ix}, view 0.")
                        elif 'door' in all_2d_labels:
                            # Door found
                            path_valid = False
                            # print(f"Invalid path: Door found at last point {ix}, view 0.")
                        
                        # # (2) Check for duplicate targets
                        # elif len(all_2d_labels) != len(set(all_2d_labels)):
                        #      # Duplicate labels found
                        #      path_valid = False
                        #      # print(f"Invalid path: Duplicate targets {all_2d_labels} at last point {ix}, view 0.")
                    
                    if not path_valid:
                        break # Break inner loop (rotations)

                    # Save Image
                    img_filename = f"point_{ix}{suffix}.png"
                    img_save_path = os.path.join(point_dir, img_filename)
                    
                    # Convert RGBA to RGB if needed
                    if rgb_np.shape[2] == 4:
                        img_to_save = cv2.cvtColor(rgb_np, cv2.COLOR_RGBA2RGB)
                    else:
                        img_to_save = rgb_np
                    
                    # Save using CV2 (expects BGR)
                    img_to_save_bgr = cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR)
                    img_to_save_bgr = cv2.resize(img_to_save_bgr, (256, 256))
                    cv2.imwrite(img_save_path, img_to_save_bgr)
                    
                    # Store data
                    view_data = {
                        "angle": angle_deg,
                        "filename": img_filename,
                        "all_2d_labels": all_2d_labels,
                        "all_3d_labels": all_3d_labels,
                        "scene_graph": pairwise_scene_graph,
                        "scene_graph_text": scene_graph_text,
                    }
                    point_data["views"].append(view_data)
                
                if not path_valid:
                    break # Break outer loop (points)

                seed_data.append(point_data)
            
            if not path_valid:
                # Clean up and continue to next seed
                if os.path.exists(seed_output_dir):
                    shutil.rmtree(seed_output_dir)
                continue

            # Generate EQA Question
            # Get data from the last point's first view (angle 0)
            last_point_view = seed_data[-1]["views"][0]
            
            
            # print("formatted_prompt:", formatted_prompt)
            
            # Call LLM
            try:
                # For image content, we need the path to the image of the last point, view 0
                # The image is saved at os.path.join(point_dir, img_filename)
                # We need to reconstruct the path or use what we have.
                # The last point index is len(path_points) - 1
                last_point_idx = len(path_points) - 1
                # Reconstruct point_dir for the last point
                last_point_dir = os.path.join(seed_output_dir, f"point_{last_point_idx}")
                last_img_path = os.path.join(last_point_dir, last_point_view["filename"])
                
                # Read image and encode to base64
                with open(last_img_path, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

                # Format the EQA prompt
                sys_prompt, formatted_prompt_eqa = EQA_Prompt_format(
                    img=encoded_string,
                    objects=last_point_view["all_2d_labels"],
                    core_relationship=last_point_view["scene_graph_text"]
                )
                _, formatted_prompt_objectgoal = ObjectGoal_Prompt_format(
                    img=encoded_string,
                    objects=last_point_view["all_2d_labels"],
                    core_relationship=last_point_view["scene_graph_text"]
                )
                # _, formatted_prompt_textgoal = TextGoal_Prompt_format(
                #     img=encoded_string,
                #     objects=last_point_view["all_2d_labels"],
                #     core_relationship=last_point_view["scene_graph_text"]
                # )
                
                # Prepare content for call_openai_api
                # contents is a list of tuples: (text,) or (text, image_base64)
                # Based on format_content in eval_utils..., it expects tuples.
                # If tuple len is 1, it's text. If len is 2, it's text + image.
                # But wait, call_openai_api takes 'contents'.
                # format_content implementation:
                # for c in contents:
                #    formated_content.append({"type": "text", "text": c[0]})
                #    if len(c) == 2: ... image ...
                
                # So we should pass [("", encoded_string)] or just the image?
                # The system prompt is passed separately.
                # The user content usually contains the image.
                # Let's pass an empty text with the image, or a instruction "Here is the image."
                
                
                eqa_q_raw = call_openai_api_qwen3vlplus(sys_prompt, formatted_prompt_eqa)
                
                # Parse eqa_q
                match = re.search(r"Task Format:\s*(.*?)\s*Question:\s*(.*?)\s*Answer:\s*(.*)", eqa_q_raw, re.IGNORECASE | re.DOTALL)
                if match:
                    task_format = match.group(1).strip()
                    eqa_q = match.group(2).strip()
                    eqa_a = match.group(3).strip()
                    
                    # Clean up square brackets if present in the matched groups (robustness)
                    task_format = re.sub(r"^\[|\]$", "", task_format).strip()
                    eqa_q = re.sub(r"^\[|\]$", "", eqa_q).strip()
                    eqa_a = re.sub(r"^\[|\]$", "", eqa_a).strip()
                    
                else:
                    # Fallback pattern for [Key] Value format
                    match_brackets = re.search(r"\[Task Format\]\s*(.*?)\s*\[Question\]\s*(.*?)\s*\[Answer\]\s*(.*)", eqa_q_raw, re.IGNORECASE | re.DOTALL)
                    if match_brackets:
                        task_format = match_brackets.group(1).strip()
                        eqa_q = match_brackets.group(2).strip()
                        eqa_a = match_brackets.group(3).strip()
                    else:
                        print(f"Invalid question format at seed {path_seed}. Discarding path.")
                        raise ValueError("Invalid Question Format")
                print(f"scene_id: {scene_id}, path_seed: {path_seed}, task format: {task_format}, eqa_q: {eqa_q}, eqa_a: {eqa_a}")

                # objectgoal = best_object
                objectgoal = call_openai_api_qwen3vlplus(sys_prompt, formatted_prompt_objectgoal)
                print(f"scene_id: {scene_id}, path_seed: {path_seed}, objectgoal: {objectgoal}")

                # # textgoal = call_openai_api_qwen3vlplus(sys_prompt, formatted_prompt_textgoal)
                # print(f"scene_id: {scene_id}, path_seed: {path_seed}, textgoal: {textgoal}")
                
                
                # Loop for each point to get eqa_experience
                for i, point_data in enumerate(seed_data):
                    point_view = point_data["views"][0]
                    
                    # Construct image path for this point
                    point_dir = os.path.join(seed_output_dir, f"point_{i}")
                    img_path = os.path.join(point_dir, point_view["filename"])
                    
                    # Read image
                    with open(img_path, "rb") as image_file:
                        point_encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

                    # Experience Prompt
                    exp_sys_eqa, exp_content_eqa = EXP_EQA(eqa_q, point_encoded_string, point_view["all_2d_labels"], point_view["scene_graph_text"])
                    eqa_experience = call_openai_api_qwen3vlplus(exp_sys_eqa, exp_content_eqa)
                    
                    # Add ObjectGoal and TextGoal
                    exp_sys_obj, exp_content_obj = EXP_ObjectGoal(objectgoal, point_encoded_string, point_view["all_2d_labels"], point_view["scene_graph_text"])
                    objectgoal_experience = call_openai_api_qwen3vlplus(exp_sys_obj, exp_content_obj)
                    
                    # # exp_sys_text, exp_content_text = EXP_TextGoal(textgoal, point_encoded_string, point_view["all_2d_labels"], point_view["scene_graph_text"])
                    # textgoal_experience = call_openai_api_qwen3vlplus(exp_sys_text, exp_content_text)

                    # Extract core experience text (keep content up to "THEN prioritize this path.")
                    def extract_core_experience(text):
                        delimiter = "THEN"
                        if delimiter in text:
                            return text.split(delimiter)[0] + "THEN prioritize this path."
                        return text # Should not happen if validated, but safe fallback

                    eqa_experience = extract_core_experience(eqa_experience)
                    objectgoal_experience = extract_core_experience(objectgoal_experience)
                    # # textgoal_experience = extract_core_experience(textgoal_experience)
                    # Validate Experience Formats
                    def validate_experience(text, type_name):
                        if type_name == "eqa":
                            # Check for EQA format keys
                            if not ("IF answering" in text and "AND observing" in text and "THEN prioritize this path" in text):
                                return False
                        elif type_name in ["object_goal", "text_goal"]:
                             # Check for Goal format keys
                            if not ("IF searching for" in text and "AND observing" in text and "THEN prioritize this path" in text):
                                return False
                        return True

                    if not (validate_experience(eqa_experience, "eqa") and 
                            validate_experience(objectgoal_experience, "object_goal")): # and 
                            # validate_experience(textgoal_experience, "text_goal")):
                        print(f"Invalid experience format at seed {path_seed}, point {i}. Discarding path.")
                        # Set path invalid flag (we need a flag that propagates out)
                        # Since we are inside the point loop, we can break and set a flag.
                        raise ValueError("Invalid Experience Format")
                    

                    # print(f"scene_id: {scene_id}, path_seed: {path_seed}, point_{i}, eqa: {eqa_experience[:20]}..., obj: {objectgoal_experience[:20]}..., text: {textgoal_experience[:20]}...")
                    
                    point_data["eqa_experience"] = eqa_experience
                    point_data["objectgoal_experience"] = objectgoal_experience
                    # point_data["textgoal_experience"] = textgoal_experience

                    # Store in ChromaDB
                    # Prepare metadata and documents for ChromaDB
                    # We store each type of experience as a separate document
                    
                    # ID format: scene_pathseed_pointindex_type
                    base_id = f"{scene_id}_{path_seed}_{i}"
                    
                    experiences = [
                        ("eqa", eqa_experience),
                        ("object_goal", objectgoal_experience),
                        # ("text_goal", textgoal_experience)
                    ]
                    
                    ids = []
                    documents = []
                    metadatas = []
                    
                    for exp_type, exp_text in experiences:
                        if exp_text: # Ensure text is not empty
                            ids.append(f"{base_id}_{exp_type}")
                            documents.append(exp_text)
                            metadatas.append({
                                "type": exp_type,
                                "scene_id": scene_id,
                                "path_seed": str(path_seed),
                                "point_index": i,
                                "generated_question": eqa_q if eqa_q else "",
                                "object_goal": objectgoal if objectgoal else "",
                                # "text_goal": textgoal if textgoal else ""
                            })
                    
                    if documents:
                        collection.add(
                            documents=documents,
                            metadatas=metadatas,
                            ids=ids
                        )
                            


                # Store all information in seed data
                seed_data_dict = {
                    "path_seed": path_seed,
                    "points": seed_data,
                    "task_format": task_format,
                    "eqa": eqa_q,
                    "answer": eqa_a,
                    "objectgoal": objectgoal,
                    # "textgoal": textgoal,
                }
                
            except Exception as e:
                print(f"Error: {e}")
                # If invalid format, we remove the seed dir and continue
                if str(e) == "Invalid Experience Format" or str(e) == "Invalid Question Format":
                    if os.path.exists(seed_output_dir):
                        shutil.rmtree(seed_output_dir)
                    continue

                seed_data_dict = {
                    "path_seed": path_seed,
                    "points": seed_data,
                    "generated_question": None,
                    "error": str(e)
                }

            # Save JSON for this seed
            json_path = os.path.join(seed_output_dir, f"seed_{path_seed}.json")
            with open(json_path, "w") as f:
                json.dump(seed_data_dict, f, indent=2)
            
            valid_paths_count += 1



        sim.close()

if __name__ == "__main__":
    data_root = "/path/to/scene_datasets/hm3d_v0.2/train"
    EXP_Name = "HM3D"
    output_path = "results/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    rgb_sensor = True 
    depth_sensor = True
    semantic_sensor = True
    
    sim_settings = {
        "width": 1280,
        "height": 1280,
        "hfov": 120,
        "default_agent": 0,
        "sensor_height": 1.5,
        "color_sensor": rgb_sensor,
        "depth_sensor": depth_sensor,
        "semantic_sensor": semantic_sensor,
        "seed": 1,
        "enable_physics": False,
    }
    
    main(data_root, EXP_Name, output_path, sim_settings, num_scenes_to_process=None)
