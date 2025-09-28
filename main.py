import numpy as np
from huggingface_hub import InferenceClient
from numpy import dot
from numpy.linalg import norm
import pdfplumber
import os

hf_api_key = os.getenv("HF_TOKEN")

def load_resume(filepath):
    """
    Extracts text from a PDF resume.
    """
    text = "" 

    # Open the PDF file
    with pdfplumber.open(filepath) as pdf:

        # Loop through all pages in the PDF
        for page in pdf.pages: 
            page_text = page.extract_text(x_tolerance=3, x_tolerance_ratio=None, y_tolerance=3, layout=False, x_density=7.25, y_density=13, line_dir_render=None, char_dir_render=None)  
            
            if page_text:
                text += page_text + "\n"
    
    return text.strip()

def compute_embeddings(texts, hf_client):
    """
    Generates embeddings for a list of texts using a sentence-transformer model.
    """
    
    embeddings = hf_client.feature_extraction(
        texts,
        model="sentence-transformers/all-MiniLM-L6-v2"
    )

    return  np.array(embeddings , dtype="float32")

def compute_similarity(resume_text, job_desc, hf_client):
    """
    Computes cosine similarity between resume and job description embeddings.
    """

    # Compute vector embeddings for the resume and job description using the Hugging Face client
    v1, v2 = compute_embeddings([resume_text, job_desc], hf_client)

    similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    return similarity

if __name__ == "__main__":

    hf_client = InferenceClient(token=hf_api_key)

    # Load resume from file
    resume_text = load_resume("resume.pdf")

    # Load job description from file
    with open("JD.txt", "r") as f:
        job_desc = f.read().strip()

    # Compute and print similarity score
    score = compute_similarity(resume_text, job_desc, hf_client)
    print(f"\nResume-Job Match Score: {score:.2f}")



