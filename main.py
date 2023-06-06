import argparse
from ResumeParser import ResumeParserClass
import torch 
from BertModel import JobPostingClassifier
import pandas as pd
from tqdm import tqdm
import numpy as np
import utils as u

def main():

    label_map = {
        0: "SCHOOL",
        1: "SKILLS",
        2: "JOBDES",
        3: "NONE"
    }

    parser = argparse.ArgumentParser(description="Extract sections from a resume PDF.")
    parser.add_argument('pdf_path', help='Path to the PDF file')
    args = parser.parse_args()
    pdf_path = 'resume/' + args.pdf_path


    resume_parser = ResumeParserClass(pdf_path)
    resume_sections = resume_parser.parse()

    for section_name, section_content in resume_sections.items():
        print(f"{section_name}:\n\n{section_content}\n")

        resume_experience = u.split_into_sentences(resume_sections['Experience'])
        resume_projects = u.split_into_sentences(resume_sections['Projects'])
        resume_skills = [resume_sections['Skills']]

    

    # Loading the model
    loaded_model = JobPostingClassifier(model_path='trained_models/model_bert_1.pth')

    # Load the jobs data
    job_data = pd.read_csv('job_data/jobs.csv')

    # Define lists to store similarity scores and job descriptions
    similarity_scores = []
    jobdes_similarity_scores = []
    skills_similarity_scores = []  
    job_descriptions = []


    # Iterate over each job
    for _, row in tqdm(job_data.iterrows(), desc='Jobs', total=job_data.shape[0]):
        # Get the job description and split it into sentences
        description = row['description']
        sentences = description.split('.')
        
        jobdes_sentences = []
        skills_sentences = []
        school_sentences = []

        # Classify each sentence and add to the corresponding list
        for sentence in tqdm(sentences, desc='Sentences', leave=False):
            classification = loaded_model.predict(sentence)
            sentence = u.preprocess_sentence(sentence)

            if classification == 2:
                jobdes_sentences.append(sentence)
            elif classification == 1:
                skills_sentences.append(sentence)
            elif classification == 0:
                school_sentences.append(sentence)
        
        # Calculate cosine similarity scores
        # Check if jobdes_sentences and resume_experience are not empty
        if jobdes_sentences and resume_experience:
            jobdes_similarity = u.calculate_cosine_similarity(jobdes_sentences, resume_experience)
        else:
            jobdes_similarity = 0  

        if skills_sentences and resume_skills:
            skills_similarity = u.calculate_cosine_similarity(skills_sentences, resume_skills)
        else:
            skills_similarity = 0
       
        # Save the scores and job description
        jobdes_similarity_scores.append(jobdes_similarity)
        skills_similarity_scores.append(skills_similarity)
        similarity_scores.append((jobdes_similarity + skills_similarity) / 2)  # average of the two scores
        job_descriptions.append(description)

    # After the loop, sort the jobs by similarity score and get the top 10
    top_jobs_indices = np.argsort(similarity_scores)[-10:]
    top_jobs = [job_descriptions[i] for i in top_jobs_indices]
    similarity_scores = [similarity_scores[i] for i in top_jobs_indices]

    print(f'{len(top_jobs)}, {len(similarity_scores)}, {len(jobdes_similarity_scores)}, {len(skills_similarity_scores)}')

    for i in range(0, 10):
        print(top_jobs[i], similarity_scores[i], jobdes_similarity_scores[i], skills_similarity_scores[i])
    


            
if __name__ == "__main__":
    main()