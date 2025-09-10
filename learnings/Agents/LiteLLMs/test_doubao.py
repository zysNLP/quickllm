# -*- coding: utf-8 -*-
"""
    @Project : quickllm
    @File    : test_doubao.py
    @Author  : sunday
    @Time    : 2025/9/10 16:56
"""

import os
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Load environment variables from .env file
load_dotenv()

# Retrieve API key and base URL from environment variables
api_key = os.getenv('LITELLM_PROXY_API_KEY')
base_url = os.getenv('LITELLM_PROXY_API_BASE')

# Initialize OpenAI client
client = AsyncOpenAI(
    api_key=api_key,
    base_url=base_url
)

# Define the request payload for image generation
model_name = "volc-engine_doubao-seedream-4-0-250828"
prompt = "给我画个猫咪"

async def generate_image_with_openai():
    """
    Generate image using OpenAI's API with doubao model.

    Returns:
        The generated image response
    """
    try:
        response = await client.images.generate(
            model=model_name,
            prompt=prompt,
            size="1024x1024",
            n=1,
            response_format="url"
        )
        return response
    except Exception as e:
        print(f"Error generating image: {e}")
        return None

# Example usage
if __name__ == "__main__":
    # Run the function
    result = asyncio.run(generate_image_with_openai())
    if result:
        print("Image generated successfully!")
        print("Response:", result)
    else:
        print("Failed to generate image")

