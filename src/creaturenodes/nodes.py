from google import genai
from google.genai import types
from PIL import Image
import io
from io import BytesIO
import numpy as np
from torchvision import transforms
import requests

modelEnum = {
    "Gemini 3 Flash Preview": "gemini-3-flash-preview",
    "Gemini 2.5 Pro": "gemini-2.5-pro",
    "Gemini 2.5 Flash": "gemini-2.5-flash",
    "Gemini 2.0 Flash": "gemini-2.0-flash",
    "Imagen 4": "imagen-4.0-generate-001",
    "Gemini 2.5 Flash Image": "gemini-2.5-flash-image",
    "Gemini 3 Pro Image Preview": "gemini-3-pro-image-preview"
}

"""
Google API Nodes to image generation
"""

class GoogleGenAIClient:
    CATEGORY = "CreatureNodes/Google"
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": "",
                })
            }
        }
    RETURN_TYPES = ("ANY",)
    RETURN_NAMES = ("google_genai_client",)
    FUNCTION = "google_genai_client"
    
    def google_genai_client(self, api_key):
        
        if not api_key:
            raise RuntimeError("Provide an api key")
        
        try:
            client = genai.Client(api_key=api_key)
            return (client,)
        except Exception as e:
            raise RuntimeError(f"Failed to configure Google GenAI API: {e}")

class GoogleT2T:
    CATEGORY = "CreatureNodes/Google"
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "google_genai_client": ("ANY",),
                "model": (["Gemini 3 Flash Preview", "Gemini 2.5 Pro", "Gemini 2.5 Flash", "Gemini 2.0 Flash"],),
                "system_prompt": ("STRING", {
                    "multiline": True
                }),
                "user_prompt": ("STRING", {
                    "multiline": True
                })
            }
        }
    RETURN_TYPES = ("STRING", "ANY",)
    RETURN_NAMES = ("generated_string", "google_genai_client,")
    FUNCTION = "google_t2t"
    
    def google_t2t(self, google_genai_client, model, system_prompt, user_prompt):
        try:
            response = google_genai_client.models.generate_content(
                model=modelEnum[model],
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt
                ),
                contents=user_prompt
            )
            return (response.text, google_genai_client,)
        except Exception as e:
            raise RuntimeError(f"Failed to generate text: {e.message}")

class GoogleT2I:
    CATEGORY = "CreatureNodes/Google"
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "google_genai_client": ("ANY",),
                "model": (["Imagen 4", "Gemini 2.5 Flash Image", "Gemini 3 Pro Image Preview"],),
                "aspect_ratio": (["1:1","2:3","3:2","3:4","4:3","4:5","5:4","9:16","16:9","21:9"],),
                "image_size": (["1K", "2K", "4K"],),
                "prompt": ("STRING", {
                    "multiline": True
                }),
            }
        }
    RETURN_TYPES = ("IMAGE", "ANY")
    RETURN_NAMES = ("generated_image", "google_genai_client")
    FUNCTION = "google_t2i"
    
    def google_t2i(self, google_genai_client, model, aspect_ratio, image_size, prompt):
        try:
            if model in ["Imagen 4"]:
                response = google_genai_client.models.generate_images(
                    model=modelEnum[model],
                    prompt=prompt,
                    config=types.GenerateImagesConfig(
                        number_of_images=1,
                        image_size=image_size,
                        aspect_ratio=aspect_ratio
                    )
                )

                # Extract raw bytes from generated_images
                image_bytes = response.generated_images[0].image.image_bytes

                # Convert bytes → PIL Image
                image = Image.open(io.BytesIO(image_bytes))
                if image.mode != "RGB":
                    image = image.convert("RGB")

            else:
                # Use generate_content for other models
                response = google_genai_client.models.generate_content(
                    model=modelEnum[model],
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_modalities=["IMAGE"],
                        image_config=types.ImageConfig(
                            aspect_ratio=aspect_ratio,
                            image_size=image_size
                        )
                    )
                )

                # Extract the inline_data
                image_parts = [part for part in response.parts if part.inline_data]
                image_bytes = image_parts[0].inline_data.data

                # Convert bytes → PIL Image
                image = Image.open(io.BytesIO(image_bytes))
                if image.mode != "RGB":
                    image = image.convert("RGB")

            # PIL → CHW tensor in [0,1]
            image_tensor = transforms.ToTensor()(image)  # (3, H, W)

            # CHW → HWC
            image_tensor = image_tensor.permute(1, 2, 0)  # (H, W, 3)

            # HWC → BHWC (batch dimension)
            image_tensor = image_tensor.unsqueeze(0)  # (1, H, W, 3)

            # Ensure float32
            image_tensor = image_tensor.float()

            return (image_tensor, google_genai_client,)
        except Exception as e:
            raise RuntimeError(f"Failed to generate image: {e}")

class GoogleI2I:
    CATEGORY = "CreatureNodes/Google"
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "google_genai_client": ("ANY",),
                "images": ("STRING",),
                "model": (["Gemini 3 Pro Image Preview"],),
                "aspect_ratio": (["1:1","2:3","3:2","3:4","4:3","4:5","5:4","9:16","16:9","21:9"],),
                "image_size": (["1K", "2K", "4K"],),
                "prompt": ("STRING", {
                    "multiline": True
                }),
            }
        }
    RETURN_TYPES = ("IMAGE", "ANY",)
    RETURN_NAMES = ("generated_image", "google_genai_client",)
    FUNCTION = "google_i2i"
    
    def google_i2i(self, google_genai_client, images, model, aspect_ratio, image_size, prompt):
        try:
            
            images = [Image.open(image) for image in images]
            
            # Use generate_content for other models
            response = google_genai_client.models.generate_content(
                model=modelEnum[model],
                contents=[prompt, *images],
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    image_config=types.ImageConfig(
                        aspect_ratio=aspect_ratio,
                        image_size=image_size
                    )
                )
            )

            # Extract the inline_data
            image_parts = [part for part in response.parts if part.inline_data]
            image_bytes = image_parts[0].inline_data.data

            # Convert bytes → PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            if image.mode != "RGB":
                image = image.convert("RGB")

            # PIL → CHW tensor in [0,1]
            image_tensor = transforms.ToTensor()(image)  # (3, H, W)

            # CHW → HWC
            image_tensor = image_tensor.permute(1, 2, 0)  # (H, W, 3)

            # HWC → BHWC (batch dimension)
            image_tensor = image_tensor.unsqueeze(0)  # (1, H, W, 3)

            # Ensure float32
            image_tensor = image_tensor.float()

            return (image_tensor,google_genai_client,)
        except Exception as e:
            raise RuntimeError(f"Failed to generate image: {e}")

"""
Model Switch node for image generation
"""

class T2IModelSwitch:
    CATEGORY = "CreatureNodes/Switches"
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "selected_model": (["SD3.5", "SDXL", "Z IMAGE", "Flux2 Klein"],)
            },
            "optional": {
                "sd35": ("IMAGE", {"lazy": True}),
                "sdxl": ("IMAGE", {"lazy": True}),
                "z_image": ("IMAGE", {"lazy": True}),
                "flux2_klein": ("IMAGE", {"lazy": True}),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "t2i_model_switch"
    
    def check_lazy_status(
        self,
        selected_model,
        sd35=None,
        sdxl=None, 
        z_image=None,
        flux2_klein=None
    ):
        needed = []
        if selected_model == "SD3.5":
            if sd35 is None: needed.append("sd35")
        elif selected_model == "SDXL":
            if sdxl is None: needed.append("sdxl")
        elif selected_model == "Z IMAGE":
            if z_image is None: needed.append("z_image")
        elif selected_model == "Flux2 Klein":
            if z_image is None: needed.append("flux2_klein")
        
        return needed
    
    def t2i_model_switch(self, selected_model, sd35=None, sdxl=None, z_image=None, flux2_klein=None):
        
        modelEnum = {
            'SD3.5': sd35,
            'SDXL': sdxl,
            'Z IMAGE': z_image,
            'Flux2 Klein': flux2_klein
        }
        
        return (modelEnum[selected_model], )

class I2IModelSwitch:
    CATEGORY = "CreatureNodes/Switches"
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "selected_model": (["LTX 2.0"],)
            },
            "optional": {
                "ltx_20_image": ("IMAGE", {"lazy": True}),
                "ltx_20_audio": ("AUDIO", {"lazy": True}),
            }
        }
    RETURN_TYPES = ("IMAGE", "AUDIO",)
    RETURN_NAMES = ("image", "audio",)
    FUNCTION = "i2i_model_switch"
    
    def check_lazy_status(
        self,
        selected_model,
        ltx_20_image=None,
        ltx_20_audio=None
    ):
        needed = []
        if selected_model == "LTX 2.0":
            if ltx_20_image is None: needed.append("ltx_20_image")
            if ltx_20_audio is None: needed.append("ltx_20_audio")
        
        return needed
    
    def i2i_model_switch(self, selected_model, ltx_20_image=None, ltx_20_audio=None):
        
        modelEnum = {
            'LTX 2.0': (ltx_20_image, ltx_20_audio,),
        }
        
        return modelEnum[selected_model]

class LMST2T:
    CATEGORY = "CreatureNodes/LM Studio"
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING",),
                "system_prompt": ("STRING",),
                "model": (["IBM Granite 4 Micro", "Gemma 3 1B"],),
                "port": ("INT", {
                    "default": 1234
                })
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output",)
    FUNCTION = "t2t_lms"
    
    def t2t_lms(self, prompt, system_prompt, model, port):
        
        modelEnum = {
            "IBM Granite 4 Micro": "ibm/granite-4-micro",
            "Gemma 3 1B": "google/gemma-3-1b"
        }
        
        response = requests.post(f'http://localhost:{port}/api/v1/chat', json={
            "model": modelEnum[model],
            "system_prompt": system_prompt,
            "input": prompt
        })
        
        return (response.json()['output'][0]['content'],)

NODE_CLASS_MAPPINGS = {
    "Google T2T": GoogleT2T,
    "Google T2I": GoogleT2I,
    "Google I2I": GoogleI2I,
    "Google GenAI Client": GoogleGenAIClient,
    "T2I Model Switch": T2IModelSwitch,
    "I2I Model Switch": I2IModelSwitch,
    "LM Studio T2T": LMST2T
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Google T2T": "Google T2T",
    "Google T2I": "Google T2I",
    "Google I2I": "Google I2I",
    "Google GenAI Client": "Google GenAI Client",
    "T2I Model Switch": "T2I Model Switch",
    "I2I Model Switch": "I2I Model Switch",
    "LM Studio T2T": "LM Studio T2T"
}