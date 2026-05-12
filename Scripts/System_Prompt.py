
import re
import random


System_Prompt = '''
You are an indoor navigation task designer. Your goal is to create a visual task that is verifiable within the image based \*strictly\* on the following raw labels and scene graph data.
'''

def System_Prompt_format():
    return System_Prompt

# EQA
def EQA_Prompt_format(img, objects, core_relationship):
    sys_prompt = System_Prompt
    content = []
    
    text = "- **Image:** "
    content.append((text, img))
    content.append(("\n",))

    text = f"- **Detected Objects:** {objects}\n"
    text += f"- **Spatial Relationship to leverage:** {core_relationship}\n\n"
    
    text += "Please follow these guidelines:\n"
    text += '1. Identification: Based on the "semantic variation summary," determine the most prominent **object** or **object relationship** (e.g., a conspicuous item) that is clearly revealed only in the image.\n'
    text += '2. Motivation: This task is important for the agent as a valuable learning point.\n'
    text += '3. Focus: Generate a specific **natural language task** based on the discovery from Guideline 1.\n'
    text += "4. Note that you should give a direct question and the corresponding answer that can be understood by others. Don't mention words like 'image', 'on the left of the image', etc.\n\n"
    
    # text += "Task Format:\n"
    # text += "1. object recognition (e.g., What is the white object on the wall above the TV?)\n"
    # text += "2. object localization (e.g., Where is the orange painting?)\n"
    # text += "3. attribute recognition (e.g., Is there space on the dining table to work on my laptop?)\n"
    # text += "4. object state recognition (e.g., Are the ceiling lights in the living room turned on?)\n"
    # text += "5. counting (e.g., How many pillows are on the sofa?)\n"
    # text += "6. world knowledge (e.g., What type of car is in the garage?)\n"
    # text += "7. spatial understanding (e.g., Does the bedroom have a lot of furniture?)\n"
    # text += "8. functional reasoning (e.g., I want to check my outfit for a dinner party, how can I do this?)\n\n"
    
    # Weighted random selection of task format
    task_weights = {
        "object recognition (e.g., What is the white object on the wall above the TV?)": 0.10,
        "object localization (e.g., Where is the orange painting?)": 0.20,
        "attribute recognition (e.g., Is there space on the dining table to work on my laptop?)": 0.20,
        "object state recognition (e.g., Are the ceiling lights in the living room turned on?)": 0.10,
        "counting (e.g., How many pillows are on the sofa?)": 0.05,
        "world knowledge (e.g., What type of car is in the garage?)": 0.15,
        "spatial understanding (e.g., Does the bedroom have a lot of furniture?)": 0.10,
        "functional reasoning (e.g., I want to check my outfit for a dinner party, how can I do this?)": 0.10
    }
    target_task = random.choices(list(task_weights.keys()), weights=list(task_weights.values()), k=1)[0]
    
    text += f"Your task is to generate a Question and the corresponding Answer specifically for the Task Format: **{target_task}**\n"
    
    text += "Return your generated Question and Answer in the following format:\n"
    text += "Task Format: [Format]\nQuestion: [Question]\nAnswer: [Answer]\n"
    
    text += "\nExample:\n"
    text += "Task Format: functional reasoning\n"
    text += "Question: It's too bright in the living room, how can I make it darker?\n"
    text += "Answer: Lower the shades over the porch door.\n"
    
    content.append((text,))
    
    return sys_prompt, content

# ObjectGoal
def ObjectGoal_Prompt_format(img, objects, core_relationship):
    sys_prompt = System_Prompt
    content = []
    
    text = "- **Image:** "
    content.append((text, img))
    content.append(("\n",))
    
    text = f"- **Detected Objects:** {objects}\n"
    text += f"- **Spatial Relationship to leverage:** {core_relationship}\n\n"
    
    text += "Please follow these guidelines:\n"
    text += '1. Identification: Based on the "semantic variation summary," determine the most prominent **object** that is clearly revealed only in the image.\n'
    text += '2. Motivation: This task is important for the agent as a valuable learning point.\n'
    text += '3. Focus: Generate one specific **Object** based on the discovery from Guideline 1.\n'
    
    text += "Task Format: \n"
    text += "[Object] (e.g., table)\n\n"
    
    text += "Return your generated object **WITHOUT ANY ADDITIONAL EXPLANATION**:\n"
    
    content.append((text,))
    
    return sys_prompt, content

# TextGoal
def TextGoal_Prompt_format(img, objects, core_relationship):
    sys_prompt = System_Prompt
    content = []
    
    text = "- **Image:** "
    content.append((text, img))
    content.append(("\n",))
    
    text = f"- **Detected Objects:** {objects}\n"
    text += f"- **Spatial Relationship to leverage:** {core_relationship}\n\n"
    
    text += "Please follow these guidelines:\n"
    text += '1. Identification: Based on the "semantic variation summary," determine the most prominent **object** or **object relationship** (e.g., a conspicuous item) that is clearly revealed only in the image.\n'
    text += '2. Motivation: This task is important for the agent as a valuable learning point.\n'
    text += '3. Focus: Generate a specific **natural language description** based on the discovery from Guideline 1.\n'
    text += "4. The description you generate must not contain the word: 'image'.\n\n"
    
    text += "Task Format: \n"
    text += '[Object description] (e.g., "A large, white plastic bag, likely containing scrap material or debris, sits on the concrete floor directly behind a long wooden workbench on the left side of the room, positioned between the bench and a doorway leading to another area.")\n\n'
    
    text += "ONLY return your generated object description:\n"
    
    content.append((text,))
    
    return sys_prompt, content


def EXP_EQA(Q, img, objects, core_relationship):
    sys_prompt = 'You are a strategic information integration expert proficient in robotic indoor navigation analysis. Your objective is to analyze the relationship between the given Question, Current Image, Objects and Spatial Relationship, distilling it into concise navigational historical experience.'
    content = []
    text = f'Question: {Q}\n'
    text += f'Detected Objects: {objects}\n'
    text += f'Spatial Relationship: {core_relationship}\n\n'
    text += 'In order to answer the Question, the robot has selected the current image as the key trajectory step:\n'

    content.append((text,))
    content.append((f"Current Image: ", img))
    text = 'Task: Deduce the link between this image and the potential answer.\n'
    text += 'Output Format:\n'
    text += 'IF answering "[Question]" AND observing "[Visual Cues in image]", THEN prioritize this path.\n'
    text += '(e.g. IF answering "What is on the couch?" AND observing "a corridor leading to living room." THEN prioritize this path.)\n'
    content.append((text,))

    return sys_prompt, content


def EXP_ObjectGoal(objectgoal, img, objects, core_relationship):
    sys_prompt = 'You are a strategic information integration expert proficient in robotic indoor navigation analysis. Your objective is to analyze the relationship between the given Goal Object, Current Image, Objects and Spatial Relationship, distilling it into concise navigational historical experience.'
    content = []
    text = f'Goal Object: {objectgoal}\n'
    text += f'Detected Objects: {objects}\n'
    text += f'Spatial Relationship: {core_relationship}\n\n'
    text += 'In order to find the Goal Object, the robot has selected the current image as the key trajectory step:\n'

    content.append((text,))
    content.append((f"Current Image: ", img))
    text = 'Task: Deduce the link between this image and the Goal Object.\n'
    text += 'Output Format:\n'
    text += 'IF searching for "[Goal Object]" AND observing "[Visual Cues in image]", THEN prioritize this path.\n'
    text += '(e.g. IF searching for "Bed" AND observing "a door near the picture leading to the bedroom." THEN prioritize this path.)\n'
    content.append((text,))

    return sys_prompt, content


def EXP_TextGoal(textgoal, img, objects, core_relationship):
    sys_prompt = 'You are a strategic information integration expert proficient in robotic indoor navigation analysis. Your objective is to analyze the relationship between the given Textual Description, Current Image, Objects and Spatial Relationship, distilling it into concise navigational historical experience.'
    content = []
    text = f'Textual Description: {textgoal}\n'
    text += f'Detected Objects: {objects}\n'
    text += f'Spatial Relationship: {core_relationship}\n\n'
    text += 'In order to find the object described in the Textual Description, the robot has selected the current image as the key trajectory step:\n'

    content.append((text,))
    content.append((f"Current Image: ", img))
    text = 'Task: Deduce the link between this image and the object described in the Textual Description.\n'
    text += 'Output Format:\n'
    text += 'IF searching for "[Textual Description]" AND observing "[Visual Cues in image]", THEN prioritize this path.\n'
    text += '(e.g. IF searching for "A stainless steel oven is positioned directly beneath a matching microwave, both built into wooden cabinetry, with a power outlet located to the left of the microwave and another power outlet situated to the right of the oven." AND observing "a kitchen with oven, microwave, and multiple power outlets implied by spatial relationships." THEN prioritize this path.)\n'
    content.append((text,))
    
    return sys_prompt, content


### Some problem, set to Object GOAL
# def EXP_ImgGoal(img_goal, img, objects, core_relationship):
#     sys_prompt = 'You are a strategic information integration expert proficient in robotic indoor navigation analysis. Your objective is to analyze the relationship between the given Goal Image, Current Image, Objects and Spatial Relationship, distilling it into concise navigational historical experience.'
#     content = []
#     content.append((f"Goal Image: ", img_goal))

#     text = f'Detected Objects: {objects}\n'
#     text += f'Spatial Relationship: {core_relationship}\n\n'
#     text += 'In order to find the Goal Image, the robot has selected the current image as the key trajectory step:\n'

#     content.append((text,))
#     content.append((f"Current Image: ", img))
#     text = 'Task: Deduce the link between this visual scene and the Goal Object. Identify why this specific scene is a correct navigational choice.\n'
#     text += 'Output Format:\n'
#     text += 'IF searching for "[Goal Object]" AND observing "[Visual Cues in image]", THEN prioritize this path.\n'
#     text += '(e.g. IF searching for "Bed" AND observing "a door near the picture leading to the bedroom" THEN prioritize this path.)\n'
#     content.append((text,))
    
#     return sys_prompt, content
