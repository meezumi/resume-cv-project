import sys
import spacy
import docx2txt
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import io
import PyPDF2
import pandas as pd
import os
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
# from tqdm.autonotebook import get_ipython

# Check if running in Colab
in_colab = 'google.colab' in sys.modules

# Download NLTK data
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# Load spaCy model
nlp = spacy.load("en_core_web_lg")


# Set up Google Drive service
def get_drive_service():
    creds = None
    SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

    if in_colab:
        from google.colab import auth
        auth.authenticate_user()
        creds, _ = default()
    else:
        if os.path.exists('token.json'):
            creds = Credentials.from_authorized_user_file('token.json', SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    "./client_secret.json", SCOPES)
                creds = flow.run_local_server(port=0)

            with open('token.json', 'w') as token:
                token.write(creds.to_json())

    return build('drive', 'v3', credentials=creds)


drive_service = get_drive_service()


def getMethod(input: str, folder_id: str):
    resumess = []
    resume_names = []
    try:
        # List files in the specified Google Drive folder
        results = drive_service.files().list(
            q=f"'{folder_id}' in parents",
            fields="files(id, name)").execute()
        items = results.get('files', [])

        for item in items:
            file_id = item['id']
            file_name = item['name']

            if file_name.lower().endswith('.pdf'):
                # Download the file content
                request = drive_service.files().get_media(fileId=file_id)
                fh = io.BytesIO()
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while done is False:
                    status, done = downloader.next_chunk()

                # Read the content
                fh.seek(0)
                pdf_reader = PyPDF2.PdfReader(fh)
                Otext = ""
                for page in pdf_reader.pages:
                    Otext += page.extract_text()

            elif file_name.lower().endswith('.docx'):
                request = drive_service.files().get_media(fileId=file_id)
                fh = io.BytesIO()
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while done is False:
                    status, done = downloader.next_chunk()

                fh.seek(0)
                Otext = docx2txt.process(fh)

            elif file_name.lower().endswith('.txt'):
                request = drive_service.files().get_media(fileId=file_id)
                fh = io.BytesIO()
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while done is False:
                    status, done = downloader.next_chunk()

                fh.seek(0)
                Otext = fh.read().decode('utf-8', errors='ignore')

            else:
                continue

            cleaned_text = cleanUp(Otext)
            resumess.append(cleaned_text)
            resume_names.append(file_name)
            print(f"Processed: {file_name}")

        print(f"Total files processed: {len(resumess)}")

        vetorizer = TfidfVectorizer().fit_transform([input] + resumess)
        vectors = vetorizer.toarray()
        job_vector = vectors[0]
        resume_vectors = vectors[1:]
        similarities = []

        for i in range(len(resumess)):
            w = resumess[i]
            similarities.append(similarity(input, w))

        # sort similarities along with indices
        sorted_indices = sorted(range(len(similarities)), key=similarities.__getitem__, reverse=True)
        top_indices = sorted_indices[-5:]
        top_resumes = [resumess[i] for i in top_indices]
        top_resume_names = [resume_names[i] for i in top_indices]
        similarity_scores = [round(similarities[i], 2) * 100 for i in top_indices]

        for name, score in zip(top_resume_names, similarity_scores):
            print(f"Resume: {name}, Similarity Score: {score}")

        return resumess

    except Exception as e:
        print(f"An error occurred: {e}")
        return [], []


def remove_stopwords(sen):
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new


def cleanUp(text):
    # remove punctuations, numbers and special characters
    clean_sentences = pd.Series([text]).str.replace("[^a-zA-Z]", " ")
    # make alphabets lowercase
    clean_sentences = [s.lower() for s in clean_sentences]
    clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]
    return clean_sentences[0]


def similarity(w1, w2):
  w1=nlp(w1)
  w2=nlp(w2)
  return score(w1,w2)


def similarity_nouns(w1_nouns,w2_nouns):
  w1_nouns=nlp(w1_nouns)
  w2_nouns=nlp(w2_nouns)
  similarity_score1=w1_nouns.similarity(w2_nouns)
  return similarity_score1


def similarity_adj(w1_adj, w2_adj):
  w1_adjs=nlp(w1_adj)
  w2_adjs=nlp(w2_adj)
  similarity_score2=w1_adjs.similarity(w2_adjs)
  return similarity_score2


def similarity_verb(w1_verb, w2_verb):
  w1_verbs=nlp(w1_verb)
  w2_verbs=nlp(w2_verb)
  similarity_score3=w1_verbs.similarity(w2_verbs)
  return similarity_score3


def score(w1,w2):
  w1_nouns=" ".join([token.lemma_ for token in w1 if token.pos_=="NOUN"])
  w2_nouns=" ".join([token.lemma_ for token in w2 if token.pos_=="NOUN"])
  w1_adj=" ".join([token.lemma_ for token in w1 if token.pos_=="ADJ"])
  w2_adj=" ".join([token.lemma_ for token in w2 if token.pos_=="ADJ"])
  w1_verb=" ".join([token.lemma_ for token in w1 if token.pos_=="VERB"])
  w2_verb=" ".join([token.lemma_ for token in w2 if token.pos_=="VERB"])
  vector1=similarity_nouns(w1_nouns,w2_nouns)
  vector2=similarity_adj(w1_adj,w2_adj)
  vector3=similarity_verb(w1_verb,w2_verb)
  sum=((vector1 + vector2+ vector3)/3)
  return sum


job_desc = "We are seeking a highly motivated and experienced Software Engineer to join our dynamic team. The ideal candidate will have a strong background in software development, excellent problem-solving skills, and the ability to work effectively in a collaborative environment. Key Responsibilities: - Develop, test, and maintain high-quality software applications. - Collaborate with cross-functional teams to define and design new features. - Troubleshoot and resolve software defects and issues. - Participate in code reviews to ensure adherence to coding standards and best practices. - Continuously learn and apply new technologies to improve software performance and scalability. Requirements: - Bachelor's degree in Computer Science, Engineering, or a related field. - 3+ years of experience in software development. - Proficiency in one or more programming languages (e.g., Python, Java, C++). - Experience with version control systems (e.g., Git). - Strong understanding of software development methodologies (e.g., Agile, Scrum). - Excellent communication and teamwork skills. Preferred Qualifications: - Experience with cloud platforms (e.g., AWS, Azure, Google Cloud). - Familiarity with front-end technologies (e.g., HTML, CSS, JavaScript). - Knowledge of database management systems (e.g., MySQL, PostgreSQL). - Previous experience in a fast-paced startup environment. If you are passionate about software engineering and eager to contribute to the success of a growing company, we encourage you to apply. Please submit your resume and cover letter for consideration."
folder_id = '1W76QMo6XSKRLpHbHHLzgDDJuVGWznuVH'
output = getMethod(job_desc, folder_id)
print(output)