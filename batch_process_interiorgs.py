#!/usr/bin/env python3
"""
Batch Process InteriorGS Scenes:
1. Traverse all scenes in data/InteriorGS
2. Generate 10 random trajectories per scene
3. Render images (256x256) to data/CG-DATA-InteriorGS
4. Do NOT save intermediate trajectory data
"""

import os
import sys
import glob
import shutil
import argparse
from tqdm import tqdm

# Import pipeline functions
# Assuming this script is in the same directory as pipeline_trajectory_and_render.py
import pipeline_trajectory_and_render as pipeline

def main():
    parser = argparse.ArgumentParser(description="Batch Process InteriorGS")
    parser.add_argument("--data_root", type=str, default="/path/to/data/InteriorGS", help="Input dataset root")
    parser.add_argument("--output_root", type=str, default="/path/to/InteriorGS", help="Output root")
    parser.add_argument("--num_trajs", type=int, default=10, help="Number of trajectories per scene")
    parser.add_argument("--resolution", type=int, default=256, help="Render resolution")
    parser.add_argument("--skip_analysis", action="store_true", help="Skip detection and scene analysis (faster, only images)")
    
    args = parser.parse_args()
    
    # Find all scene directories (e.g., 0001_839920)
    # They are direct subdirectories of data_root
    scene_dirs = sorted(glob.glob(os.path.join(args.data_root, "*_*")))
    scene_dirs = [d for d in scene_dirs if os.path.isdir(d) and os.path.basename(d)[0].isdigit()]
    
    print(f"Found {len(scene_dirs)} scenes in {args.data_root}")
    print(f"Output to: {args.output_root}")
    
    # Configure Trajectory Generation
    traj_cfg = pipeline.TrajConfig(
        num_trajs=args.num_trajs,
        seed=0,
        clearance_px=2,
        allow_unknown=False,
        min_clearance_m=0.5,
        min_path_m=2.0,
        max_path_m=12.0,
        max_pair_tries=300,
        eight_connected=True,
        forbid_diagonal_corner_cut=True,
        num_points_min=5,
        num_points_max=10,
        cam_height_m=1.5,
        side_yaw_offset_deg=120.0,
        save_debug_maps=False, # Don't need debug maps if we delete them anyway
        save_free_mask_debug=False,
        save_dist_debug=False
    )

    for scene_dir in tqdm(scene_dirs, desc="Processing Scenes"):
        scene_id = os.path.basename(scene_dir)
        
        # Define paths
        # We use a temporary directory inside the output folder for trajectories, then delete it
        scene_output_dir = os.path.join(args.output_root, scene_id)
        temp_traj_dir = os.path.join(scene_output_dir, "temp_trajs")
        
        # 1. Generate Trajectories
        # generate_trajectories returns the number of generated paths
        # It creates output directories automatically
        
        # Clean temp dir if exists
        if os.path.exists(temp_traj_dir):
            shutil.rmtree(temp_traj_dir)
            
        try:
            num_generated = pipeline.generate_trajectories(scene_dir, temp_traj_dir, traj_cfg)
        except Exception as e:
            print(f"[Error] Failed to generate trajectories for {scene_id}: {e}")
            continue
            
        if num_generated == 0:
            print(f"[Warning] No trajectories generated for {scene_id}")
            if os.path.exists(temp_traj_dir):
                shutil.rmtree(temp_traj_dir)
            continue
            
        # 2. Render
        # Check for PLY file
        ply_name = "3dgs_uncompressed.ply"
        ply_path = os.path.join(scene_dir, ply_name)
        if not os.path.exists(ply_path):
            ply_name = "3dgs_compressed.ply"
            ply_path = os.path.join(scene_dir, ply_name)
            
        if not os.path.exists(ply_path):
            print(f"[Error] No 3DGS ply file found for {scene_id}")
            shutil.rmtree(temp_traj_dir)
            continue
            
        try:
            pipeline.render_trajectories(
                ply_path=ply_path,
                traj_root=temp_traj_dir,
                out_root=scene_output_dir, # Render directly to scene folder
                width=args.resolution,
                height=args.resolution,
                hfov_deg=120.0,
                pitch_deg=0.0,
                views=["front", "left", "right"],
                max_frames=0, # No limit
                skip_analysis=args.skip_analysis
            )
        except Exception as e:
            print(f"[Error] Failed to render {scene_id}: {e}")
        
        # 3. Cleanup Trajectory Data
        # User requested not to save trajectory data
        if os.path.exists(temp_traj_dir):
            shutil.rmtree(temp_traj_dir)
            
    print("Batch processing completed.")

if __name__ == "__main__":
    main()
