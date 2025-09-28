import os
import numpy as np
import pdfplumber
from numpy import dot
from numpy.linalg import norm
from huggingface_hub import InferenceClient
from groq import Groq
# pip install pdfplumber huggingface-hub groq

hf_api_key = os.getenv("HF_TOKEN")
groq_api_key = os.getenv("GROQ_TOKEN")

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

def generate_review(groq_client, resume_text, job_desc, score):
    """
    Generate structured resume feedback using Groq LLM.
    """

    prompt = f"""
    You are an expert career coach and ATS optimization concultant
    Job Description:
    {job_desc}

    Resume:
    {resume_text}

    Resume-Job Match Score: {score:.2f}

    Provide a detailed review with:
    1. Strengths of this resume for the given job.
    2. Weaknesses or gaps compared to the job description.
    3. Overall verdict on hiring potential.
    """

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )

    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    
    # Initialize clients
    hf_client = InferenceClient(token=hf_api_key)
    groq_client = Groq(api_key=groq_api_key)

    # Load resume and job description
    resume_text = load_resume("resume.pdf")
    with open("JD.txt", "r") as f:
        job_desc = f.read().strip()

    # Compute Resume–Job similarity
    print("\nEvaluating resume vs job description...")
    score = compute_similarity(resume_text, job_desc, hf_client)
    print(f"Resume-Job Match Score: {score:.2f}")

    # Generate structured review
    print("\nGenerating resume review...")
    review = generate_review(groq_client, resume_text, job_desc, score)
    print("\nResume Review Report:\n")
    print(review)



