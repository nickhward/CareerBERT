# 🚀 **CareerBert**: an automated job search project! 🔍🎯

CareerBert is a tool that brings a data-driven approach to your job search. The aim of this project is to develop a different approach for comparing multiple job descriptions with a given resume. 

This system uses the power of **DistilBertSequence Classification** to classify sentences into categories such as `job description`, `skills`, and `education`. The model training and classification are accomplished using *Doccano*, an open-source text annotation tool, to label sentences effectively. 

⚠️ Kindly note, the model has been trained only on `data science`, `data analysis`, and `data engineering` roles. 

![Animation2](https://github.com/nickhward/CareerBERT/blob/main/gifs/Animation2.gif)
![Animation2](https://github.com/nickhward/CareerBERT/blob/main/gifs/Animation.gif)

## 🌐 **How to get Job Data?** 

For data extraction, an API key from *SerpAPI* is required. Once obtained, replace the `api_key` on **line 46** in the `web_scrape.py` file. This will result in a CSV file containing data for 50 jobs.

❗ Run judiciously to avoid ban due to scraping rules.


## 🛠 **Model Training & Usage** 

Model training requires GPU support and is done on *Google Colab* with a GPU runtime. After data scraping, load the data into the Google Colab notebook to train the model.

To train a model on a different role, uncomment the last line in the notebook that saves a model, add the model to the `trained_models` directory, and change the model name in the code accordingly.


## 📝 **Resume Parsing and Matching** 

The system transforms each sentence of a resume and the job description in question into a TF-IDF representation and uses cosine similarity to compare it with job descriptions. Resume parsing is currently supported only for PDF files.

Choose between running in the terminal or launching a Flask app:

- For terminal: `python main.py resume.pdf`
- For Flask app: `python app.py`

The Flask app displays the job description, the link to the webpage, and the similarity scores in total and per section.


## 📄 **Resume Parser Formatting** 

Currently, the resume parser expects the resume sections to be in the following format:

- EDUCATION
- EXPERIENCE
- PERSONAL PROJECTS
- PUBLICATIONS
- TECHNICAL SKILLS

This will expand in the future to accommodate more diverse resume structures.


