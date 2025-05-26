import numpy as np
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from llm_analyse import analyse_image_with_llm
from PIL import Image
from utils import get_image_tiles

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

vector_store = Chroma(
    collection_name="sentinel_2_mosaics",
    embedding_function=embeddings,
    persist_directory="./chroma_semanticeo",
)


def save_to_vector_db(image_data, analysis):
    """Save the image data and analysis to ChromaDB"""

    # Create metadata
    metadata = image_data["metadata"]

    # Create document text from analysis
    text = f"""
    General Description: {analysis.general_description}
    
    Land Cover:
    {', '.join([f"{lc.type} ({lc.area_percentage}%): {lc.description}" for lc in analysis.land_cover])}
    
    Land Use:
    {', '.join([f"{lc.type} ({lc.confidence}%): {lc.description}" for lc in analysis.land_use])}
    
    Notable Features:
    {', '.join(analysis.notable_features)}
    
    Environmental Assessment:
    {analysis.environmental_assessment}
    """

    # Create doc id
    id = f"id_{metadata['grid_id']}_{metadata['year']}_{metadata['quarter']}"

    # Save to ChromaDB
    vector_store.add_texts(texts=[text], metadatas=[metadata], ids=[id])

    return id


def query_vector_db(query_text, n_results=5):
    """Query the vector database for similar entries"""
    results = vector_store.similarity_search_with_score(query=query_text, k=n_results)
    return results


if __name__ == "__main__":
    print("Done!")
