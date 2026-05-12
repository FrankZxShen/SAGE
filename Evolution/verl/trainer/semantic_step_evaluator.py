import re
from collections import defaultdict
from typing import List, Dict, Any
import logging
import os

logger = logging.getLogger(__name__)

def predefined_llm_scoring_function(step_content: Dict[str, Any]) -> float:
    """
    Predefined function to score a step using an LLM.
    
    Args:
        step_content (Dict[str, Any]): Contains prompt, response, and metadata.
        
    Returns:
        float: The score for the step (e.g., 0.0 to 1.0).
    """
    # TODO: Implement actual LLM call here
    # For now, return a dummy score
    return 0.0

class SemanticStepEvaluator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.scoring_function = predefined_llm_scoring_function

    def evaluate_batch(self, batch):
        """
        Process the batch to reconstruct trajectories and evaluate steps.
        
        Args:
            batch (DataProto): The batch data from rollout.
        """
        # 1. Decode prompts and responses
        if 'prompts' not in batch.batch or 'responses' not in batch.batch:
            logger.warning("Batch does not contain prompts or responses. Skipping semantic evaluation.")
            return

        prompts_ids = batch.batch['prompts']
        responses_ids = batch.batch['responses']
        
        # Decode to string
        prompts_str = self.tokenizer.batch_decode(prompts_ids, skip_special_tokens=True)
        responses_str = self.tokenizer.batch_decode(responses_ids, skip_special_tokens=True)
        
        # 2. Retrieve metadata from batch.non_tensor_batch
        # The NavDataset stores metadata in non_tensor_batch (eps_path, seed_path, y_idx)
        
        trajectories = defaultdict(list)
        non_tensor = batch.non_tensor_batch
        
        # Check if keys exist
        has_meta = all(k in non_tensor for k in ['eps_path', 'seed_path', 'y_idx'])
        
        if not has_meta:
            logger.warning("Batch missing metadata (eps_path, seed_path, y_idx). Cannot group trajectories correctly.")
            # Fallback to simple batch index grouping if critical metadata is missing
            # or just skip grouping logic
            pass

        for idx, (prompt, response) in enumerate(zip(prompts_str, responses_str)):
            if has_meta:
                # Extract from non_tensor_batch
                eps_path = non_tensor['eps_path'][idx]
                seed_path = non_tensor['seed_path'][idx]
                point_id = non_tensor['y_idx'][idx]
                
                eps_id = os.path.basename(eps_path)
                seed_id = os.path.basename(seed_path)
                
                step_info = {
                    'batch_idx': idx,
                    'eps': eps_id,
                    'seed': seed_id,
                    'point': point_id,
                    'prompt': prompt,
                    'response': response
                }
                # print("step_info:",step_info)
                # exit()
                key = (eps_id, seed_id)
                trajectories[key].append(step_info)
            else:
                # Without metadata, we can't group by episode/seed
                # Just treat each sample as independent or log error
                pass

        # 4. Sort steps by point_X (from small to large)
        processed_trajectories = []
        for (eps, seed), steps in trajectories.items():
            # Sort by point_id
            steps.sort(key=lambda x: x['point'])
            
            traj_data = {
                'eps': eps,
                'seed': seed,
                'steps': steps
            }
            processed_trajectories.append(traj_data)
            
        # 5. Call predefined function to score each step
        self._score_trajectories(processed_trajectories)
        
        return processed_trajectories

    def _score_trajectories(self, trajectories):
        """
        Iterate through trajectories and score each step.
        """
        for traj in trajectories:
            eps = traj['eps']
            seed = traj['seed']
            for step in traj['steps']:
                score = self.scoring_function(step)
                step['semantic_score'] = score
                # Optional: Store the score back or log it
                # logger.info(f"Evaluated {eps}/{seed} point {step['point']}: {score}")
