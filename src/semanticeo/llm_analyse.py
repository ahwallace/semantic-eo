import base64
import io
from typing import List, Optional

import numpy as np
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from PIL import Image
from pydantic import BaseModel, Field
from utils import get_image_tiles


# Define Pydantic models for structured output
class LandCoverElement(BaseModel):
    type: str = Field(description="Type of land cover (e.g., forest, urban, water)")
    area_percentage: float = Field(
        description="Estimated percentage of the image covered by this type"
    )
    description: str = Field(description="Brief description of this land cover element")


class LandUseElement(BaseModel):
    type: str = Field(
        description="Type of land use (e.g., agriculture, heavy industrial, residential)"
    )
    confidence: float = Field(
        description="Estimated percentage confidence of the land use element being present in the image"
    )
    description: str = Field(description="Brief description of this land use element")


class ChangeDetection(BaseModel):
    detected: bool = Field(
        description="Whether changes were detected between the two images"
    )
    description: str = Field(description="Description of the changes detected")
    impact_level: str = Field(
        description="Environmental impact level: low, medium, or high"
    )


class ImageAnalysis(BaseModel):
    general_description: str = Field(
        description="General description of the satellite image"
    )
    land_cover: List[LandCoverElement] = Field(
        description="List of land cover elements identified"
    )
    land_use: List[LandUseElement] = Field(description="List of potential land uses")
    notable_features: List[str] = Field(
        description="List of notable features in the image"
    )
    environmental_assessment: str = Field(
        description="Assessment of environmental conditions"
    )
    change_detection: Optional[ChangeDetection] = Field(
        None, description="Change detection results if two images provided"
    )


def encode_image_to_base64(image):
    """Convert PIL image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()


def analyse_image_with_llm(image_data):
    """Use LLM to analyze the satellite image"""
    # Initialize the LLM
    llm = ChatOpenAI(model="gpt-4.1", max_tokens=1500, temperature=0)

    # Setup the parser
    parser = PydanticOutputParser(pydantic_object=ImageAnalysis)

    # Encode images to base64
    base64_image = encode_image_to_base64(image_data["image"])

    # Create prompt for single image analysis
    prompt = f"""
    You are an expert in satellite imagery analysis. I'm showing you a Sentinel-2 satellite image mosaic with satellite images taken during quarter {image_data['metadata']['quarter']} of year {image_data['metadata']['year']}.
    
    Please analyse this image and provide a detailed description, focusing on:
    1. Land cover types and their approximate percentages. Be specific.
    2. Land use types and their approximate probability.
    3. Notable features visible in the image as well as notable omissions.
    4. Environmental assessment based on what you can see
    
    Format your response according to the provided JSON schema.
    {parser.get_format_instructions()}
    """

    # Create message with single image
    messages = [
        HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "high",
                    },
                },
            ]
        )
    ]

    # Get response from the LLM
    response = llm.invoke(messages)

    # Parse the response to structured format
    try:
        structured_analysis = parser.parse(response.content)
        return structured_analysis
    except Exception as e:
        print(e)


if __name__ == "__main__":
    print("Done!")
