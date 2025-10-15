import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import openai
import base64
import string
import cv2

client  = openai.OpenAI(api_key="API_KEY")

def encode_image(image_path):
    """Encodes an image to base64 format."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
def encode_image_array(img_array):
    """Encodes an RGB NumPy array to base64 format."""
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    success, buffer = cv2.imencode(".png", img_bgr)
    if not success:
        raise ValueError("Image encoding failed")

    img_base64 = base64.b64encode(buffer).decode("utf-8")
    return img_base64

def asko4mini(image_array, prompt_type):
    """Sends an image to GPT-o4mini for classification."""
    classMapping = {    
        "undamaged": 0,
        "lightly damaged": 1,
        "moderately damaged": 2,
        "severely damaged": 3,
        "undamaged.": 0,
        "lightly damaged.": 1,
        "moderately damaged.": 2,
        "severely damaged.": 3
    }

    encoded_image = encode_image_array(image_array)
    if prompt_type == "detailed":
        # Detailed prompt
        response = client.chat.completions.create(
            model="o4-mini",
            messages=[
                {"role": "system", "content": [{"type": "text", "text": "You are an expert in structural damage assessment for buildings. You will be provided with images of building components that have been masked. Your task is to classify the level of damage for each masked component based on visual cues in the image. The damage levels are defined as follows: "
                        "Undamaged: The component shows no visible signs of damage, degradation, or distress. There are no cracks, spalls, corrosion, deformations, or missing elements. The surface appears intact and in good condition. "

                        "Lightly Damaged: The component exhibits minor damage that does not significantly affect its structural integrity or functionality. This may include: "
                            "Hairline cracks (less than 1mm wide). "
                            "Minor surface spalling (small chips or flakes missing). "
                            "Superficial staining or discoloration. "
                            "Slight corrosion with minimal material loss. "

                        "Moderately Damaged: The component shows more significant damage that could potentially compromise its structural integrity or functionality in the long term. This may include: "
                            "Cracks wider than 1mm but less than 5mm. "
                            "Noticeable spalling with some exposed rebar. "
                            "Moderate corrosion with some material loss. "
                            "Minor deformations or misalignments. "
                            "Some missing pieces of finishing or cladding. "

                        "Severely Damaged: The component exhibits extensive damage that significantly impairs its structural integrity or functionality and poses an immediate risk. This may include: "
                            "Large cracks (wider than 5mm) or significant cracking patterns. "
                            "Extensive spalling with exposed and corroded rebar." 
                            "Significant corrosion with substantial material loss. "
                        "The response should only be a single phrase naming the classification from the following classes: undamaged, lightly damaged, moderately damaged, severely damaged. Regardless of clarity provide a classification. "}]}, #Classify any people you may see as the undamaged class.
                {"role": "user", "content": [
                    {"type": "text", "text": "Classify the damage level of this image in a single phrase."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_image}"}}
                ]}
            ],
            reasoning_effort="high"
        )
    elif prompt_type == "short":
        # Simple prompt
        response = client.chat.completions.create(
            model="o4-mini",
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": "You are an expert in structural damage assessment for buildings. Classify this masked-out "
                    "building component as being either undamaged, lightly damaged, moderately damaged, or "
                    "severely damaged based on the visible damage to the component. Response should be a "
                    "single phrase naming the classification."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_image}"}}
                ]}
            ],
            reasoning_effort="high"
        )
    else:
        raise ValueError("Incorrect prompt type specified")

    classification = response.choices[0].message.content
    classID = classMapping[classification.lower()]
    print(f'{classification}: {classID}')
    return classID