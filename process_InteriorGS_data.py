import os
import sys
import glob
import shutil
import json
import base64
import re
import cv2
import numpy as np
import torch
from PIL import Image
import supervision as sv
from collections import Counter
from tqdm import tqdm
import chromadb
import random
import argparse

# Add project root to path
project_root = "/path/to/SAGE"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Imports from project
from src.conceptgraph.utils.general_utils import ObjectClasses, filter_detections
from src.conceptgraph.utils.model_utils import compute_clip_features_batched
from src.eval_utils_gpt_aeqa import call_openai_api_qwen3vlplus
from Scripts.System_Prompt import EQA_Prompt_format, ObjectGoal_Prompt_format, EXP_EQA, EXP_ObjectGoal

# Import DetectionModel
try:
    from sceneimg_gen_final import DetectionModel, convert_scene_graph_to_text
except ImportError:
    print("Could not import from sceneimg_gen_final. Defining classes locally.")
    from ultralytics import YOLO
    import clip
    
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

        formatted_sentences = [descriptions[0] + "."]
        for d in descriptions[1:]:
            d = d[0].upper() + d[1:] + "."
            formatted_sentences.append(d)
            
        return "The spatial relationship where " + " ".join(formatted_sentences)

# Initialize ChromaDB
chroma_db_path = "/path/to/chroma_db"
if not os.path.exists(chroma_db_path):
    os.makedirs(chroma_db_path)
chroma_client = chromadb.PersistentClient(path=chroma_db_path)
collection = chroma_client.get_or_create_collection(name="nav_experiences")

# Initialize Models
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

detection_model = DetectionModel(device=device)
obj_classes = ObjectClasses(
    classes_file_path=os.path.join(project_root, "data/scannet200_classes.txt"),
    bg_classes=["wall", "floor", "ceiling"],
    skip_bg=True,
    class_set="scannet200",
)
detection_model.set_classes(obj_classes.get_classes_arr())

import clip
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
clip_tokenizer = clip.tokenize

def process_image_2d(img_path):
    # Load image
    image_bgr = cv2.imread(img_path)
    if image_bgr is None:
        print(f"Failed to load image: {img_path}")
        return None
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    H, W = image_rgb.shape[:2]
    
    # 1. Detection
    results = detection_model.predict(image_rgb, conf=0.5, verbose=False)
    if not results:
        return None
        
    confidences = results[0].boxes.conf.cpu().numpy()
    detection_class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
    xyxy_tensor = results[0].boxes.xyxy
    xyxy_np = xyxy_tensor.cpu().numpy()
    
    detection_class_labels = [
        f"{obj_classes.get_classes_arr()[class_id]} {class_idx}"
        for class_idx, class_id in enumerate(detection_class_ids)
    ]

    # 2. Create Masks (Rectangular)
    masks_np = np.zeros((len(xyxy_np), H, W), dtype=bool)
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
        return {
            "all_2d_labels": [],
            "scene_graph_text": "",
            "image_base64": "", # Will fill later if needed
            "annotated_image": image_rgb
        }

    # 3. Filter Detections
    curr_det, labels = filter_detections(
        image=image_rgb,
        detections=curr_det,
        classes=obj_classes,
        given_labels=detection_class_labels,
        iou_threshold=0.5,
        min_mask_size_ratio=0.0,
        confidence_threshold=0.5,
    )
    
    if curr_det is None or len(curr_det) == 0:
        return {
            "all_2d_labels": [],
            "scene_graph_text": "",
            "annotated_image": image_rgb
        }

    # 4. CLIP Features (Optional but good to keep consistency)
    image_crops, image_feats, text_feats = compute_clip_features_batched(
        image_rgb,
        curr_det,
        clip_model,
        clip_preprocess,
        clip_tokenizer,
        obj_classes.get_classes_arr(),
        device,
    )

    # 5. Scene Graph (2D)
    # Calculate labels
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
            obj_classes.get_classes_arr()[class_id]
            for class_id in curr_det.class_id
        ]
        
    # Calculate relationships
    eps = 0.05
    obj_infos_2d = []
    # We need to map back from filtered detections to class names
    for i in range(len(curr_det.xyxy)):
        # If labels provided, use them, else use class_id
        curr_class_name = detection_labels_2d[i]
        x1, y1, x2, y2 = curr_det.xyxy[i]
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

    scene_graph_text = convert_scene_graph_to_text(scene_graph)
    
    # Visualization (Optional)
    annotated_image = image_rgb.copy()
    # ... visualization logic if needed ...

    return {
        "all_2d_labels": detection_labels_2d,
        "scene_graph_text": scene_graph_text,
        "annotated_image": annotated_image
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run on a single trajectory for testing")
    args = parser.parse_args()

    root_dir = "/path/to/CG-DATA-InteriorGS"
    print(f"Scanning {root_dir}...")
    sys.stdout.flush()
    
    # Get all scene directories
    scene_dirs = sorted(glob.glob(os.path.join(root_dir, "*_*")))
    
    for scene_dir in tqdm(scene_dirs, desc="Scenes"):
        scene_id = os.path.basename(scene_dir)
        traj_dirs = sorted(glob.glob(os.path.join(scene_dir, "traj_*")))
        
        for traj_dir in tqdm(traj_dirs, desc=f"Trajectories in {scene_id}", leave=False):
            # Check if JSON already exists? 
            # If we want to overwrite/reprocess, we ignore existence.
            # But let's check for "seed_xxxx.json" equivalent.
            traj_name = os.path.basename(traj_dir)
            json_path = os.path.join(traj_dir, f"{traj_name}.json")
            
            # Identify points
            point_dirs = sorted(glob.glob(os.path.join(traj_dir, "point_*")), 
                               key=lambda x: int(os.path.basename(x).split('_')[1]))
            
            if not point_dirs:
                print(f"Empty trajectory: {traj_dir}. Deleting.")
                shutil.rmtree(traj_dir)
                continue
            
            # --- Validation Step ---
            # Check last point, front view
            last_point_dir = point_dirs[-1]
            last_img_path = os.path.join(last_point_dir, f"{os.path.basename(last_point_dir)}_front.png")
            
            if not os.path.exists(last_img_path):
                # Fallback to any png?
                print(f"Missing front image in {last_point_dir}. Checking other views.")
                pngs = glob.glob(os.path.join(last_point_dir, "*.png"))
                if not pngs:
                    print(f"No images in {last_point_dir}. Invalid path.")
                    shutil.rmtree(traj_dir)
                    continue
                last_img_path = pngs[0]

            # Process last image for validation
            result_last = process_image_2d(last_img_path)
            if not result_last:
                print(f"Processing failed for {last_img_path}. Deleting path.")
                shutil.rmtree(traj_dir)
                continue
                
            all_2d_labels = result_last["all_2d_labels"]
            
            path_valid = True
            if not all_2d_labels:
                path_valid = False
            elif 'power outlet' in all_2d_labels:
                path_valid = False
            elif 'door' in all_2d_labels:
                path_valid = False
            
            if not path_valid:
                print(f"Invalid path (labels: {all_2d_labels}): {traj_dir}. Deleting.")
                shutil.rmtree(traj_dir)
                continue
            
            # --- Generation Step ---
            # 1. Generate EQA and ObjectGoal using Last Point
            with open(last_img_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            
            # EQA
            sys_prompt, formatted_prompt_eqa = EQA_Prompt_format(
                img=encoded_string,
                objects=result_last["all_2d_labels"],
                core_relationship=result_last["scene_graph_text"]
            )
            
            # ObjectGoal
            _, formatted_prompt_objectgoal = ObjectGoal_Prompt_format(
                img=encoded_string,
                objects=result_last["all_2d_labels"],
                core_relationship=result_last["scene_graph_text"]
            )
            
            try:
                # Call LLM for EQA
                eqa_q_raw = call_openai_api_qwen3vlplus(sys_prompt, formatted_prompt_eqa)
                
                # Parse EQA
                match = re.search(r"Task Format:\s*(.*?)\s*Question:\s*(.*?)\s*Answer:\s*(.*)", eqa_q_raw, re.IGNORECASE | re.DOTALL)
                if match:
                    task_format = match.group(1).strip()
                    eqa_q = match.group(2).strip()
                    eqa_a = match.group(3).strip()
                    
                    task_format = re.sub(r"^\[|\]$", "", task_format).strip()
                    eqa_q = re.sub(r"^\[|\]$", "", eqa_q).strip()
                    eqa_a = re.sub(r"^\[|\]$", "", eqa_a).strip()
                else:
                     match_brackets = re.search(r"\[Task Format\]\s*(.*?)\s*\[Question\]\s*(.*?)\s*\[Answer\]\s*(.*)", eqa_q_raw, re.IGNORECASE | re.DOTALL)
                     if match_brackets:
                        task_format = match_brackets.group(1).strip()
                        eqa_q = match_brackets.group(2).strip()
                        eqa_a = match_brackets.group(3).strip()
                     else:
                        print(f"Invalid question format in {traj_dir}. Deleting.")
                        shutil.rmtree(traj_dir)
                        continue

                # Call LLM for ObjectGoal
                objectgoal = call_openai_api_qwen3vlplus(sys_prompt, formatted_prompt_objectgoal)
                
            except Exception as e:
                print(f"LLM Error in {traj_dir}: {e}")
                # shutil.rmtree(traj_dir) # Optional: delete if LLM fails?
                continue

            # 2. Generate Experience for EACH point
            traj_data_points = []
            valid_experience = True
            
            for i, point_dir in enumerate(point_dirs):
                # Use front view for experience generation
                point_name = os.path.basename(point_dir)
                img_path = os.path.join(point_dir, f"{point_name}_front.png")
                if not os.path.exists(img_path):
                     # fallback
                     pngs = glob.glob(os.path.join(point_dir, "*.png"))
                     if pngs: img_path = pngs[0]
                     else: continue
                
                # Process Image
                res = process_image_2d(img_path)
                if not res: continue
                
                with open(img_path, "rb") as image_file:
                    point_encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                
                # Generate Experiences
                try:
                    exp_sys_eqa, exp_content_eqa = EXP_EQA(eqa_q, point_encoded_string, res["all_2d_labels"], res["scene_graph_text"])
                    eqa_experience = call_openai_api_qwen3vlplus(exp_sys_eqa, exp_content_eqa)
                    
                    exp_sys_obj, exp_content_obj = EXP_ObjectGoal(objectgoal, point_encoded_string, res["all_2d_labels"], res["scene_graph_text"])
                    objectgoal_experience = call_openai_api_qwen3vlplus(exp_sys_obj, exp_content_obj)
                    
                    # Clean up experience text
                    def extract_core_experience(text):
                        delimiter = "THEN"
                        if delimiter in text:
                            return text.split(delimiter)[0] + "THEN prioritize this path."
                        return text

                    eqa_experience = extract_core_experience(eqa_experience)
                    objectgoal_experience = extract_core_experience(objectgoal_experience)
                    
                    # tqdm.write(f"  [Point {i}]")
                    # tqdm.write(f"    EQA Experience: {eqa_experience}")
                    # tqdm.write(f"    Obj Experience: {objectgoal_experience}")
                    
                    # Validate Experience
                    def validate_experience(text, type_name):
                        if type_name == "eqa":
                            if not ("IF answering" in text and "AND observing" in text and "THEN prioritize this path" in text):
                                return False
                        elif type_name == "object_goal":
                            if not ("IF searching for" in text and "AND observing" in text and "THEN prioritize this path" in text):
                                return False
                        return True

                    if not (validate_experience(eqa_experience, "eqa") and validate_experience(objectgoal_experience, "object_goal")):
                        print(f"Invalid experience format at {point_dir}. Invalidating path.")
                        valid_experience = False
                        break
                    
                    # Save Point Data
                    point_data = {
                        "point_idx": i,
                        "point_dir": point_dir,
                        "view_front": {
                            "filename": os.path.basename(img_path),
                            "all_2d_labels": res["all_2d_labels"],
                            "scene_graph_text": res["scene_graph_text"]
                        },
                        "eqa_experience": eqa_experience,
                        "objectgoal_experience": objectgoal_experience
                    }
                    traj_data_points.append(point_data)
                    
                    # Store in ChromaDB
                    base_id = f"{scene_id}_{traj_name}_{i}"
                    
                    experiences_to_store = [
                        ("eqa", eqa_experience),
                        ("object_goal", objectgoal_experience)
                    ]
                    
                    ids = []
                    documents = []
                    metadatas = []
                    
                    for exp_type, exp_text in experiences_to_store:
                        if exp_text:
                            ids.append(f"{base_id}_{exp_type}")
                            documents.append(exp_text)
                            metadatas.append({
                                "type": exp_type,
                                "scene_id": scene_id,
                                "traj_name": traj_name,
                                "point_index": i,
                                "generated_question": eqa_q if eqa_q else "",
                                "object_goal": objectgoal if objectgoal else ""
                            })
                            
                    if documents:
                        collection.add(
                            documents=documents,
                            metadatas=metadatas,
                            ids=ids
                        )

                except Exception as e:
                    print(f"Error generating experience for {point_dir}: {e}")
                    valid_experience = False
                    break
            
            if not valid_experience:
                shutil.rmtree(traj_dir)
                continue
                
            # Save JSON
            final_data = {
                "scene_id": scene_id,
                "traj_name": traj_name,
                "points": traj_data_points,
                "task_format": task_format,
                "eqa_question": eqa_q,
                "eqa_answer": eqa_a,
                "objectgoal": objectgoal
            }
            
            with open(json_path, "w") as f:
                json.dump(final_data, f, indent=2)
                
            print(f"Successfully processed {traj_dir}")

if __name__ == "__main__":
    main()
