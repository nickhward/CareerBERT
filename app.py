from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from flask_socketio import SocketIO, emit
from ResumeParser import ResumeParserClass
from BertModel import JobPostingClassifier
import pandas as pd
from tqdm import tqdm
import numpy as np
import utils as u
import os
import webbrowser
from threading import Timer

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}

app = Flask(__name__)
socketio = SocketIO(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Check if the UPLOAD_FOLDER exists and create it if not
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Global variable to hold the results
results = {}



@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('start')
def handle_start(filename):
    global results
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

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
    links = []
    i, j = 0, 0
   
    # Iterate over each job
    for _, row in job_data.iterrows():
        # Get the job description and split it into sentences
        description = row['description']
        old_description = row['description']
        link = row['job_link']
        sentences = description.split('.')
        
        jobdes_sentences = []
        skills_sentences = []
        school_sentences = []
        job_progress = (i / job_data.shape[0]) * 100
  
        j = 0
        i += 1
        emit('progress', {'job_progress': job_progress, 'sentence_progress': 0})
        
        
        # Classify each sentence and add to the corresponding list
        for sentence in sentences:
            classification = loaded_model.predict(sentence)
            sentence = u.preprocess_sentence(sentence)

            if classification == 2:
                jobdes_sentences.append(sentence)
            elif classification == 1:
                skills_sentences.append(sentence)
            elif classification == 0:
                school_sentences.append(sentence)

            sentence_progress = (j / len(sentences)) * 100
            j+=1
            emit('progress', {'job_progress': job_progress, 'sentence_progress': sentence_progress})
        
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
        job_descriptions.append(old_description)
        links.append(link)

    # After the loop, sort the jobs by similarity score and get the top 10
    top_jobs_indices = np.argsort(similarity_scores)[-10:]
    top_jobs = [job_descriptions[i] for i in top_jobs_indices]
    top_similarity_scores = [similarity_scores[i] for i in top_jobs_indices]
    top_jobdes_scores = [jobdes_similarity_scores[i] for i in top_jobs_indices]
    top_skill_scores = [skills_similarity_scores[i] for i in top_jobs_indices]
    top_links = [links[i] for i in top_jobs_indices]

            # Store the result in the application context
    results = {
        'jobs': top_jobs,
        'scores': top_similarity_scores,
        'jobdes_scores': top_jobdes_scores,
        'skills_scores': top_skill_scores,
        'links': top_links
    }

    emit('progress', {'job_progress': 100, 'sentence_progress': 100})


@app.route('/results')
def results():
    global results
    # Get your top_jobs and similarity scores here
    # This is just a placeholder, replace with your actual data

    # We're passing the jobs and scores to the template
    return render_template(
        'results.html', 
        zip=zip, 
        jobs=results['jobs'], 
        scores=results['scores'], 
        jobdes_scores=results['jobdes_scores'], 
        skills_scores=results['skills_scores'],
        links=results['links'],
    )


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_file():
    # check if the post request has the file part
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    # if user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        return 'No selected file'
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return filename

def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000')

if __name__ == '__main__':
    Timer(1, open_browser).start()
    app.run(port=5000)
