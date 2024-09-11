import PyPDF2
from fastapi import HTTPException
from sklearn.feature_extraction.text import TfidfVectorizer
from cloud_storage import *
import re
import csv
import numpy as np
from io_processing import process_incoming_voice


def get_title(directory):
    files_list = os.listdir(directory)
    stop_words = ["arrangementofsections", "sections", "section", "arrengementofsections", "arrengementofsection",
                  "arrangmentofsections", "arrangementofsection",
                  "arrngementofsections", "arrangamentofsections", "arrangementsofsections", "arrengmentofsections",
                  "arrangmentofsection", "arrangaemntofsections",
                  "arrangementsections", "arramgememtofsections", "arrangementsofsection", "contents", "1shorttitle",
                  "statement"]
    titles_list = []
    for filename in files_list:
        final_title = ""
        with open(directory + "/" + filename, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            page_one = pdf_reader.pages[0].extract_text().strip()
            page_one = page_one.split("\n")
            page_one = [page for page in page_one if page]
            titles = page_one[:10]
            for title in titles:
                new_string = re.sub('[^a-zA-Z0-9]', '', title.strip())
                if new_string.lower() in stop_words:
                    break
                else:
                    final_title += title
        titles_list.append(final_title)
    return titles_list


def tfidf_indexing(directory):
    files_list = os.listdir(directory)
    titles_list = []
    with open("titles.txt", "r") as file:
        for line in file:
            titles_list.append(line.strip())

    headers = ["Document Name", "Document Title", "Document Public Url"]
    with open("Titles.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        for i in range(len(files_list)):
            writer.writerow(
                {"Document Name": files_list[0], "Document Title": titles_list[0], "Document Public Url": ""})


def querying_with_tfidf(query, input_language, audio_file):
    titles_map = {}
    with open("Titles.csv", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            titles_map[row["Document Title"]] = row["Document Public Url"]
    titles_list = list(titles_map.keys())

    if query == "":
        regional_text, english_text, error_message = process_incoming_voice(audio_file, input_language)
        if english_text is None:
            raise HTTPException(status_code=503, detail=error_message)
        else:
            print(english_text)
            query = english_text

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(titles_list)
    title_vector = vectorizer.transform([query])
    cosine_similarities = tfidf_matrix.dot(title_vector.T).toarray().flatten()
    top_3_indices = np.argsort(cosine_similarities)[-3:][::-1]
    answer_list = []
    for i in top_3_indices:
        title = titles_list[i]
        answer_list.append((title, titles_map[title], round(cosine_similarities[i], 2)))
    return answer_list
