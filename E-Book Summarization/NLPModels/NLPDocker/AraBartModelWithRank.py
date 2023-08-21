from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from arabert.preprocess import ArabertPreprocessor
import re
import string
import sys
import argparse
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
import math
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from pyarabic.araby import strip_tashkeel, strip_tatweel
from nltk.tokenize import sent_tokenize


arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''
english_punctuations = string.punctuation
punctuations_list = arabic_punctuations + english_punctuations

arabic_diacritics = re.compile("""
                             ّ    | # Tashdid
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ     # Tatwil/Kashida
                         """, re.VERBOSE)

import nltk
nltk.download('stopwords')

model_name="abdalrahmanshahrour/arabartsummarization"
preprocessor = ArabertPreprocessor(model_name="")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
pipeline = pipeline("text2text-generation",model=model,tokenizer=tokenizer)


#lex rank from skratch, parameter 'text' is text after cleaning,
# parameter 'len' is len(cleanedLst) to make a percentage
def LexRankTech(text,length):
  # Tokenize the text into sentences
  sentences = sent_tokenize(text)

  # Preprocess the sentences by removing diacritics and tatweel
  preprocessed_sentences = [strip_tashkeel(strip_tatweel(sentence)) for sentence in sentences]

  # Convert the preprocessed sentences to lowercase and split into words
  words_list = [[word.lower() for word in sentence.split()] for sentence in preprocessed_sentences]

  # Create a vocabulary from the words
  vocabulary = sorted(set(word for sentence in words_list for word in sentence))

  # Create a dictionary that maps words to indices in the vocabulary
  word_to_index = {word: index for index, word in enumerate(vocabulary)}

  # Create a document-term matrix from the words
  document_term_matrix = csr_matrix((len(sentences), len(vocabulary)), dtype=np.float32)
  for row, words in enumerate(words_list):
      for word in words:
          document_term_matrix[row, word_to_index[word]] += 1

  # Compute the cosine similarity of the sentences
  sentence_similarity_matrix = cosine_similarity(document_term_matrix)

  # Create a graph from the similarity matrix
  graph = sentence_similarity_matrix.copy()

  # Normalize the graph by row
  row_sums = graph.sum(axis=1)
  graph /= row_sums[:, np.newaxis]

  # Set the damping factor for the PageRank algorithm
  damping_factor = 0.85

  # Initialize the PageRank scores
  scores = np.ones(len(sentences), dtype=np.float32) / len(sentences)

  # Run the PageRank algorithm
  for _ in range(100):
      scores = (1 - damping_factor) + damping_factor * np.dot(graph, scores)

  # Sort the sentences by their PageRank score
  ranked_sentences = sorted(((score, index) for index, score in enumerate(scores)), reverse=True)

  # Print the top N sentences as the summary
    
  N =math.floor( (length*20) /100) 
  summary_sentences = [sentences[index] for score, index in ranked_sentences[:N]]
  summary = ' '.join(summary_sentences)

  return summary


def convertStringToList(stringText):
  sentences_tokens = stringText.split(".")
  output = []
  for sen in sentences_tokens:
      if len(sen)>2:
          output.append(sen)
  return output


def remove_diacritics(text):
    text = re.sub(arabic_diacritics, '', text)
    return text


def remove_punctuations(text):

    pattern = r'[^\w\s.]'
    # Remove the punctuation marks using the regular expression pattern
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text


def remove_numbers(text):
    pattern=r'[0-9]'
    return re.sub(pattern, ' ', text)

#remove diacritics,punctuations and numbers from text 
def CleanTheText(text): #4
  text=remove_diacritics(text)
  text=remove_punctuations(text)
  text=remove_numbers(text)
  return text


def applyTheModel(cleanedList):#8
  out=[]
  for i in range(len(cleanedList)):
    result = pipeline(cleanedList[i],
    pad_token_id=tokenizer.eos_token_id,
    num_beams=120,
    repetition_penalty=3.0,
    max_length=4000,
    length_penalty=6.0,
    no_repeat_ngram_size = 3)[0]['generated_text']
    out.append(result)
  return out

def Run(book):
        out=[]

        fileList=book #readTheFileAndCopyIntoList(book)
        print("fileList: ",len(fileList))

        fileList=fileList.decode('utf-8')
        text=CleanTheText(fileList)
        print("text  after cleaning: ",len(text))

        length=convertStringToList(text)
        print("length of list: ",len(length))

        lexRank=LexRankTech(text,len(length))
        print("LexRank: ",len(lexRank))

        summary=convertStringToList(lexRank)
        print("Summary: ",len(summary))

        out=applyTheModel(summary)
        out=applyTheModel(out)
        return  out 

from fastapi import FastAPI, Request, File, UploadFile, Depends
from pydantic import BaseModel
import uvicorn


app = FastAPI()

#Upload a file and return filename as a reponse  
@app.post("/uploadfile")
async def create_upload_file(data: UploadFile = File(...)):
    
    out=Run(await data.read())
    print(len(out))
    print(data.filename)
    
    return {"Summarization":out }

uvicorn.run(app,host="127.0.0.1",port=8000)

