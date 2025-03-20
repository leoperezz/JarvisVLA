from pathlib import Path
import pathlib
from typing import Union,Literal
import base64
from io import BytesIO
import numpy as np
from PIL import Image
import cv2
import requests
import io
import math

def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor

def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor

def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor

def smart_resize(
    height: int, width: int, factor: int, min_pixels: int, max_pixels: int,max_ratio:int
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > max_ratio:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {max_ratio}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar

def fetch_image(image: Image.Image,  factor: int, min_pixels: int, max_pixels: int,max_ratio:int) -> Image.Image:
    width, height = image.size
    resized_height, resized_width = smart_resize(
            height,
            width,
            factor=factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            max_ratio=max_ratio,
        )
    image = image.resize((resized_width, resized_height))
    return image

def pil2base64(image):
    """强制中间结果为jpeg""" 
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str


def encode_image_to_base64(image:Union[str,pathlib.PosixPath,Image.Image,np.ndarray], format='JPEG') -> str:
    """Encode an image to base64 format, supports URL, numpy array, and PIL.Image."""

    # Case 1: If the input is a URL (str)
    image_encode = None
    if isinstance(image, str) and image[:4]=="http":
        try:
            response = requests.get(image)
            response.raise_for_status()
            return base64.b64encode(response.content).decode('utf-8')
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Failed to retrieve the image from the URL: {e}")
    elif isinstance(image, str) and image[0]=='/':
        with image.open('rb') as image_file:
            image_encode =  base64.b64encode(image_file.read()).decode('utf-8')
        return image_encode
    elif isinstance(image,pathlib.PosixPath):
        with image.open('rb') as image_file:
            image_encode =  base64.b64encode(image_file.read()).decode('utf-8')
        return image_encode
    # Case 3: If the input is a numpy array
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        buffer.seek(0)
        image_bytes = buffer.read()
        return base64.b64encode(image_bytes).decode('utf-8')

    # Case 4: If the input is a PIL.Image
    elif isinstance(image, Image.Image):
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        buffer.seek(0)
        image_bytes = buffer.read()
        return base64.b64encode(image_bytes).decode('utf-8')

    # Raise an error if the input type is unsupported
    else:
        raise ValueError("Unsupported input type. Must be a URL (str), numpy array, or PIL.Image.")

def get_suffix(image:Union[list,str,pathlib.PosixPath,np.ndarray,Image.Image]):
    if isinstance(image,np.ndarray|Image.Image):
        image_suffix = 'jpeg'
    elif isinstance(image,str):
        image_suffix = image.split(".")[-1]
    elif isinstance(image,pathlib.PosixPath):
        image_suffix = image.suffix[1:]
    else:
        raise ValueError(f"invalid image type！")
    return image_suffix

def translate_cv2(image: Union[str, pathlib.PosixPath, np.ndarray, Image.Image]) -> np.ndarray:
    if isinstance(image, Image.Image):
        # Convert PIL Image to NumPy array (PIL is in RGB)
        img_array = np.array(image)
        cv2_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    elif isinstance(image, np.ndarray):
        # Check if the NumPy array is in RGB format and has three channels
        if image.shape[2] == 3:  # Only for color images
            cv2_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            cv2_image = image  # No conversion needed for grayscale images
    elif isinstance(image, (str, pathlib.PosixPath)):
        # Read the image using cv2 (assumes BGR format)
        cv2_image = cv2.imread(str(image))  # Convert PosixPath to string if necessary
        if cv2_image is None:
            raise ValueError(f"The image path is incorrect or the file is not accessible: {image}")
    else:
        raise ValueError("Unsupported image format or path type")
    
    return cv2_image

    

class ProcessorWrapper:
    def __init__(self,processor=None,model_name= "qwen2_vl"):
        self.processor = processor
        self.model_name = model_name
        self.image_factor = 28
        self.min_pixels = 4 * 28 * 28
        self.max_pixels = 1024 * 28 * 28  #16384 * 28 * 28
        self.max_ratio = 200

    def get_image_message(self,source_data):
        image_suffix = get_suffix(source_data)
        image_message = {
                "type": "image_url",
                "image_url": { "url": f"data:image/{image_suffix};base64,{encode_image_to_base64(source_data)}"},
            }
        return image_message



    def create_message_vllm(self,
                            role:Literal["user","assistant"]="user",
                            input_type:Literal["image","text"]="image",
                            image:Union[list,str,pathlib.PosixPath,np.ndarray,Image.Image]=None,
                            prompt:Union[list,str]="",):
        if role not in {"user","assistant"}:
            raise ValueError(f"a invalid role {role}")
        if isinstance(prompt,str):
            prompt = [prompt]
        message = {
            "role": role,
            "content": [],
        }
        if input_type=="image":
            if not isinstance(image,list):
                image = [image]
            for idx, text in enumerate(prompt):
                message["content"].append({
                    "type": "text",
                    "text": f"{text}\n"
                })
                if idx < len(image):
                    message["content"].append(self.get_image_message(image[idx]))
            for idx in range(len(prompt), len(image)):
                message["content"].append(self.get_image_message(image[idx])) 
        else:
            for idx, text in enumerate(prompt):
                message["content"].append({
                    "type": "text",
                    "text": f"{text}\n"
                })
        return message

    def create_message(self,role="user",input_type="image",prompt:str="",image=None):

        if self.model_name in {"llava-next","molmo"} :
            if input_type=="image":
                message = {
                    "role":role,
                    "content":[
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ]
                }
            if input_type == "text":
                message = {
                    "role":role,
                    "content":[
                        {"type": "text", "text": prompt},
                    ]
                }
        return message
    
    def create_text_input(self,conversations:list):
        text_prompt = self.processor.apply_chat_template(conversations, add_generation_prompt=True)
        return text_prompt
    
    def create_image_input(self,image_pixels=None,image_path:str=""):
        image = image_pixels
        if image_path:
            image = Image.open(image_path)
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image.astype('uint8'))
        if self.model_name in "qwen2_vl":
            image = fetch_image(image,factor=self.image_factor,min_pixels=self.min_pixels,max_pixels=self.max_pixels,max_ratio=self.max_ratio,)
        return image
