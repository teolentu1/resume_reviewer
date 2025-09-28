# Install Required Library: pip install pdfplumber

import pdfplumber

resume_path = "resume.pdf"

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

if __name__ == "__main__":

    # Call the function to extract resume text
    resume_text = load_resume(resume_path)

    # Print the extracted resume content
    print("\nResume Content:\n", resume_text)


