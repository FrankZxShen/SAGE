# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import difflib
from typing import Any


# Metadata
REWARD_NAME = "nav"
REWARD_TYPE = "batch"


def format_reward(response: str) -> float:
    # Check for strict format:
    # 1. "Frontier Image i\nReason:..."
    # 2. "Memory Image i\nAnswer:..."
    
    # Using regex to match start of string or newline, allowing for whitespace flexibility
    # We look for "Frontier Image <num>" followed by newline and "Reason:"
    frontier_pattern = re.compile(r"Frontier\s+Image\s+\d+\s*\n\s*Reason:", re.IGNORECASE)
    
    # We look for "Memory Image <num>" followed by newline and "Answer:"
    memory_pattern = re.compile(r"Memory\s+Image\s+\d+\s*\n\s*Answer:", re.IGNORECASE)
    
    if frontier_pattern.search(response) or memory_pattern.search(response):
        return 1.0
    return 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    # Ensure strict format compliance first
    if format_reward(response) == 0.0:
        return -10.0

    # Capture category and index
    pattern = r"(Memory|Frontier)\s+Image\s+(\d+)"
    
    match = re.search(pattern, response, re.IGNORECASE)
    gt_match = re.search(pattern, ground_truth, re.IGNORECASE)
    
    if match and gt_match:
        # Check both category and index
        # Normalize category to lowercase for comparison
        cat_resp = match.group(1).lower()
        idx_resp = int(match.group(2))
        
        cat_gt = gt_match.group(1).lower()
        idx_gt = int(gt_match.group(2))
        
        if cat_resp == cat_gt and idx_resp == idx_gt:
            reward = 1.0
            
            # Semantic Similarity for Memory Images
            # User requirement: reward must be 2 * similarity if "Memory Image i" is correctly selected.
            if cat_gt == "memory":
                # Extract Answer content
                # Format: "Memory Image i\nAnswer: [Content]"
                
                def extract_answer(text):
                    # Find "Answer:" case insensitive
                    # Split by "Answer:" and take the last part (or the part immediately following)
                    parts = re.split(r"Answer:", text, flags=re.IGNORECASE)
                    if len(parts) > 1:
                        return parts[1].strip()
                    return ""
                
                ans_resp = extract_answer(response)
                ans_gt = extract_answer(ground_truth)
                
                if ans_resp and ans_gt:
                    # Note: User requested using embedding model at "/home/szx/.cache/chroma/onnx_models/all-MiniLM-L6-v2/onnx/"
                    # However, dependencies (sentence_transformers, onnxruntime) might not be available in the environment.
                    # We use difflib.SequenceMatcher as a robust fallback for semantic similarity.
                    sim = difflib.SequenceMatcher(None, ans_resp, ans_gt).ratio()
                    reward += 1 * sim
            
            return reward
        elif cat_gt == "memory" and cat_resp == "frontier":
            # Penalty for choosing Frontier when Memory was required
            return -1.0
        elif cat_gt == "frontier" and cat_resp == "memory":
        # Penalty for choosing Memory when Frontier was required
            return -1.0
        elif cat_resp == cat_gt and idx_resp != idx_gt:
            # Penalty for choosing correct category but wrong Image index
            return -0.5
        # elif cat_resp == "memory":
        #      # Penalty for choosing wrong Memory Image
        #      return -5.0
            
    return 0.0


def compute_score(reward_inputs: list[dict[str, Any]], format_weight: float = 0.1) -> list[dict[str, float]]:
    scores = []
    for reward_input in reward_inputs:
        response = reward_input["response"]
        
        # Apply the same Qwen2.5-VL format handling as in math.py just in case
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", response)
        
        format_score = format_reward(response)
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
        
        scores.append(
            {
                "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
                "format": format_score,
                "accuracy": accuracy_score,
            }
        )

    return scores
