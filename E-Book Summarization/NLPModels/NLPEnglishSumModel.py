from itertools import count
import PyPDF2
import numpy as np
import pandas as pd
import re
import spacy
import nltk
import nltk.corpus
import torch
import math
import uvicorn
import pytorch_pretrained_bert as ppb
assert 'bert-large-cased' in ppb.modeling.PRETRAINED_MODEL_ARCHIVE_MAP
from transformers import BertTokenizerFast, EncoderDecoderModel
from transformers import pipeline
from concurrent.futures import ThreadPoolExecutor
from threading import Thread
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
stemmer = PorterStemmer()
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
from nltk import sent_tokenize, word_tokenize
from transformers import BertTokenizerFast, EncoderDecoderModel
from fastapi import FastAPI, Request, File, UploadFile, Depends
from pydantic import BaseModel
from PyPDF2 import PdfFileReader

outlines=[]
pagesNumber=[]
levels=[]
Details=[]
TextPage=[]



class ThreadWithReturnValue(Thread):    
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return

# get pages number 
# here we will get the pages number of outline
def printBookmarksPageNumbers(pdf):
    def review_and_print_bookmarks(bookmarks, lvl=0):
        for b in bookmarks:
            if type(b) == list:
                review_and_print_bookmarks(b, lvl + 4)
                continue
            pg_num = pdf.get_destination_page_number(b) + 1 #page count starts from 0
            #print(pg_num)
            pagesNumber.append(pg_num)
    
    review_and_print_bookmarks(pdf.outline)


#here we will get outlines from the book
def show_tree(bookmark_list, indent=0):
    count=0
    for item in bookmark_list:
        if isinstance(item, list):
            # recursive call with increased indentation
            show_tree(item, indent + 4)
        else:
            nameOfOutline=" " * indent + item.title.lower()
            level=int(indent/2)
            if(level==0):
              level+=1
            levels.append(str(level))
            outlines.append(nameOfOutline)
            count+=1
           # print(pdf.get_destination_page_number(item) + 1 )
            #print(" " * indent + item.title.lower()) 
            
#merge outlines details [level , outline , number of page]
def mergeOutlinesDetails():
   TheFullOutlines=[]
   count=0
   for outline in outlines:
    TheFullOutlines.append([levels[count],outline,pagesNumber[count]])
    count+=1
   return TheFullOutlines

#to get introduction and abstract
def contains_intro_or_abstract(text):
    # Convert the text to lowercase to make the search case-insensitive
    text = text.lower()
    
    # Define a list of keywords to search for
    keywords = ['introduction', 'abstract']
    
    # Search for each keyword in the text
    for keyword in keywords:
        if keyword in text:
            return True
    
    # If none of the keywords are found, return False
    return False
  
#read from book 
def readPagesFromBook(pdf):
 openPdf = open(pdf,'rb')
 read = PyPDF2.PdfReader(openPdf)
 for page in range(len(read.pages)):
   TextPage.append(read.pages[page].extract_text().lower())
 return TextPage

def ConvertToLower(outlines):
  lower=[]
  for i in outlines:
    lower.append(i.lower())
  return lower

def Cleanning(outlines):
  lowerLst=outlines
  keyWords=["title","content","contents","title page","introduction","about the authors","about"
    ,"permissions","searchable terms","acknowledgments","praise","credits"
    ,"copyright","about the publisher","publisher’s preface","author’s preface"
    ,"preface","credits","Index","Conclusion","Abstract","contact us"]
  keyWords=ConvertToLower(keyWords)
  outputAfterCleaning=[]
  for i in keyWords:
    for j in lowerLst:
      if i in j :
        lowerLst.remove(j)
  return lowerLst
      


def CheckForMoreLevels(CleaningLst):
  Details=[]
  for i in CleaningLst:
      str="1"
      if (i[0] > str):
         return True

def RemoveLevelOne(CleaningLst):

  if(CheckForMoreLevels(CleaningLst)==True):
    for i,j in zip(CleaningLst,CleaningLst[1:] ):
        if (i[0] == "2"):
          Title=i[1]
          From=int(i[2])
          To=(j[2]-1)
          Details.append([Title,From,To])
          
  if(CheckForMoreLevels(CleaningLst)==None):
    for i,j in zip(CleaningLst,CleaningLst[1:] ):
        if (i[0] == "1"):
          Title=i[1]
          From=int(i[2])
          To=(j[2]-1)
          Details.append([Title,From,To])

def getoutlines(pdf2):
 print("get outline processing .......")
 with open(pdf2, "rb") as f:
     pdf = PyPDF2.PdfReader(f)
     printBookmarksPageNumbers(pdf)
 reader = PyPDF2.PdfReader(pdf2)
 show_tree(reader.outline)
 print("finished .......")

def getPagesForAllOutlines():
  print("get text from book .......")
  textOfAllOutLines=[]
  text=""
  for i in Details:
   for j in range(i[1],i[2]):
     text+=TextPage[j-1]
   numOfPages=i[2]-i[1]
   textOfAllOutLines.append([i[0],text,numOfPages])
   text=""
  print("finished .......")
  return textOfAllOutLines


summarizer = pipeline(
     "summarization",
     "pszemraj/long-t5-tglobal-base-16384-book-summary",
     device=0 if torch.cuda.is_available() else -1,
 )

def modelWithT5Version(token):
   long_text =token 
   result = summarizer(long_text)
   print(result[0]["summary_text"])
   return result[0]["summary_text"]


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizerFast.from_pretrained('mrm8488/bert-small2bert-small-finetuned-cnn_daily_mail-summarization')
model = EncoderDecoderModel.from_pretrained('mrm8488/bert-small2bert-small-finetuned-cnn_daily_mail-summarization').to(device)

#Bert model
def generate_summary(text):
    # cut off at BERT max length 512
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizerFast.from_pretrained('mrm8488/bert-small2bert-small-finetuned-cnn_daily_mail-summarization')
    model = EncoderDecoderModel.from_pretrained('mrm8488/bert-small2bert-small-finetuned-cnn_daily_mail-summarization').to(device)
    inputs = tokenizer([text], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    output = model.generate(input_ids, attention_mask=attention_mask)
  
    return tokenizer.decode(output[0], skip_special_tokens=True)

#cleaning method


def clean_text(text):
    # Remove any chapter names (e.g. "Chapter 1")
    text = re.sub(r'chapter \d+', '', text, flags=re.IGNORECASE)

    # Remove any URLs
    text = re.sub(r'http\S+', '', text)

    # Remove any email addresses
    text = re.sub(r'\S+@\S+', '', text)

    # Remove any phone numbers
    text = re.sub(r'\d{3}-\d{3}-\d{4}', '', text)
    text = re.sub(r'\d{3}\.\d{3}\.\d{4}', '', text)
    text = re.sub(r'\d{10}', '', text)

    # Remove any non-alphanumeric characters (except for spaces)
    text = re.sub(r'[^\w\s]', '', text)

    # Convert to lowercase
    text = text.lower()

    # Remove any extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

#join 
def join_paragraphs(textList,size):
   newlist=[]
   count=0
   text=""
   for p in textList:
     if len(p) < 30:
       continue
     count+=1
     text=text+p
     if(count <= size):
         newlist.append(text)
         count=0
         text=""
   if text != "":
    newlist.append(text)  
   return newlist 



#split method
def split_into_paragraphs(text,size=10):
    # Split the text into lines
    lines = text.split('\n')

    # Initialize variables
    paragraphs = []
    current_paragraph = []

    # Loop through the lines of text
    for line in lines:
        # If the line is empty, it marks the end of a paragraph
        if not line.strip():
            # Add the current paragraph to the list of paragraphs
            paragraphs.append(' '.join(current_paragraph))
            # Reset the current paragraph
            current_paragraph = []
        else:
            # Add the line to the current paragraph
            current_paragraph.append(line)

    # Add the final paragraph to the list of paragraphs
    if current_paragraph:
        paragraphs.append(' '.join(current_paragraph))
    
    return join_paragraphs(paragraphs,size)
 

#ranking ------------------------
from nltk.cluster.util import cosine_distance

MULTIPLE_WHITESPACE_PATTERN = re.compile(r"\s+", re.UNICODE)


def normalize_whitespace(text):
    """
    Translates multiple whitespace into single space character.
    If there is at least one new line character chunk is replaced
    by single LF (Unix new line) character.
    """
    return MULTIPLE_WHITESPACE_PATTERN.sub(_replace_whitespace, text)


def _replace_whitespace(match):
    text = match.group()

    if "\n" in text or "\r" in text:
        return "\n"
    else:
        return " "


def is_blank(string):
    """
    Returns `True` if string contains only white-space characters
    or is empty. Otherwise `False` is returned.
    """
    return not string or string.isspace()


def get_symmetric_matrix(matrix):
    """
    Get Symmetric matrix
    :param matrix:
    :return: matrix
    """
    return matrix + matrix.T - np.diag(matrix.diagonal())


def core_cosine_similarity(vector1, vector2):
    """
    measure cosine similarity between two vectors
    :param vector1:
    :param vector2:
    :return: 0 < cosine similarity value < 1
    """
    return 1 - cosine_distance(vector1, vector2)


'''
Note: This is not a summarization algorithm. This Algorithm pics top sentences irrespective of the order they appeared.
'''


class TextRank4Sentences():
    def __init__(self):
        self.damping = 0.85  # damping coefficient, usually is .85
        self.min_diff = 1e-5  # convergence threshold
        self.steps = 100  # iteration steps
        self.text_str = None
        self.sentences = None
        self.pr_vector = None

    def _sentence_similarity(self, sent1, sent2, stopwords=None):
        if stopwords is None:
            stopwords = []

        sent1 = [w.lower() for w in sent1]
        sent2 = [w.lower() for w in sent2]

        all_words = list(set(sent1 + sent2))

        vector1 = [0] * len(all_words)
        vector2 = [0] * len(all_words)

        # build the vector for the first sentence
        for w in sent1:
            if w in stopwords:
                continue
            vector1[all_words.index(w)] += 1

        # build the vector for the second sentence
        for w in sent2:
            if w in stopwords:
                continue
            vector2[all_words.index(w)] += 1

        return core_cosine_similarity(vector1, vector2)

    def _build_similarity_matrix(self, sentences, stopwords=None):
        # create an empty similarity matrix
        sm = np.zeros([len(sentences), len(sentences)])

        for idx1 in range(len(sentences)):
            for idx2 in range(len(sentences)):
                if idx1 == idx2:
                    continue

                sm[idx1][idx2] = self._sentence_similarity(sentences[idx1], sentences[idx2], stopwords=stopwords)

        # Get Symmeric matrix
        sm = get_symmetric_matrix(sm)

        # Normalize matrix by column
        norm = np.sum(sm, axis=0)
        sm_norm = np.divide(sm, norm, where=norm != 0)  # this is ignore the 0 element in norm

        return sm_norm

    def _run_page_rank(self, similarity_matrix):

        pr_vector = np.array([1] * len(similarity_matrix))

        # Iteration
        previous_pr = 0
        for epoch in range(self.steps):
            pr_vector = (1 - self.damping) + self.damping * np.matmul(similarity_matrix, pr_vector)
            if abs(previous_pr - sum(pr_vector)) < self.min_diff:
                break
            else:
                previous_pr = sum(pr_vector)

        return pr_vector

    def _get_sentence(self, index):

        try:
            return self.sentences[index]
        except IndexError:
            return ""

    def get_top_sentences(self, number=5):

        top_sentences = []

        if self.pr_vector is not None:

            sorted_pr = np.argsort(self.pr_vector)
            sorted_pr = list(sorted_pr)
            sorted_pr.reverse()

            index = 0
            for epoch in range(number):
                sent = self.sentences[sorted_pr[index]]
                sent = normalize_whitespace(sent)
                top_sentences.append(sent)
                index += 1

        return top_sentences

    def analyze(self, text, stop_words=None):
        self.text_str = text
        self.sentences = sent_tokenize(self.text_str)
        
        tokenized_sentences = [word_tokenize(sent) for sent in self.sentences]

        similarity_matrix = self._build_similarity_matrix(tokenized_sentences, stop_words)

        self.pr_vector = self._run_page_rank(similarity_matrix)

#ranking end ------------------------
def split(string,segmentSize):
  count=0
  segment=" "
  segmentList=[]
  for char in string:
    segment+=char
    if char ==" ":
      count+=1
    if count >= segmentSize and char==".":
      segmentList.append(segment)
      count=0
      segment=""
  if(count != 0):
     segmentList.append(segment)
     count=0
     segment=""
  return segmentList; 


#to get calc Percantage
def FromPercantageTONumber(number,percantage):
   return math.floor(number*percantage)

#start model one 
def startFirstModel(string,percentage):
  #split data for bert input
  BertOutput=[]
  BertString=""
  count=0
  inputToBert=split_into_paragraphs(string,20)
  inputToBertAftercleaning=inputToBert
  for segment in inputToBertAftercleaning:
      BertOutput.append(ThreadWithReturnValue(target=generate_summary, args=(clean_text(segment),)))
  for Output in BertOutput:
      Output.start()  
      BertString=BertString+" "+Output.join()   
  print(BertString) 
  tr4sh = TextRank4Sentences()
  
  tr4sh.analyze(BertString) 
  size=FromPercantageTONumber(len(tr4sh.sentences),percentage)
  RankOutput=tr4sh.get_top_sentences(size)
  t5InputBeforeCleaning=" ".join(RankOutput)
  summaryText=""
  summaryText+=modelWithT5Version(t5InputBeforeCleaning)
  
  return summaryText

def modeLcontroler(inputForModel,percentage):
   output=[]
   count=0
   for item in inputForModel:
     count+=1
     print("round number "+str(count))
     output.append({"sum":startFirstModel(item[1],percentage),"Chapter":item[0]})
     if count > 4:
        break
   return  output 

result=[]

def firing(url,Percantage):
  TextPage=readPagesFromBook(url)
  print("numper of pages "+str(len(TextPage)))
  getoutlines(url)
  print("get page number processing .......")
  outlines=mergeOutlinesDetails()
  CleaningLst=Cleanning(outlines)
  RemoveLevelOne(CleaningLst)
  print("finished .......")
  inputForModel=getPagesForAllOutlines()
  result=modeLcontroler(inputForModel,Percantage)
  return result
#----------------------------------------
#api 
app = FastAPI() 
@app.post("/uploadfile")
async def create_upload_file(data,percantage):    
    out=firing(data,int(percantage)/100) 
    return out#"Filename": data.filename,
uvicorn.run(app,host="127.0.0.1",port=9000)