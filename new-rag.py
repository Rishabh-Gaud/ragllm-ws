from tqdm.rich import trange, tqdm
from rich import console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.text import Text
import warnings
warnings.filterwarnings(action='ignore')
import datetime
from rich.console import Console
console = Console(width=110)
from transformers import pipeline
import os
from keybert import KeyBERT

from keybert import KeyBERT
kw_model = KeyBERT(model='intfloat/multilingual-e5-base')

logfile = 'KeyBERT-Log.txt'

def writehistory(text):
    with open(logfile, 'a', encoding='utf-8') as f:
        f.write(text)
        f.write('\n')
    f.close()

def extract_keys(text, ngram,dvsity):
    import datetime
    import random
    a = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, ngram), stop_words='english',
                              use_mmr=True, diversity=dvsity, highlight=True)     #highlight=True
    tags = []
    for kw in a:
        tags.append(str(kw[0]))
    timestamped = datetime.datetime.now()
    #LOG THE TEXT AND THE METATAGS
    logging_text = f"LOGGED ON: {str(timestamped)}\nMETADATA: {str(tags)}\nsettings: keyphrase_ngram_range (1,{str(ngram)})  Diversity {str(dvsity)}\n---\nORIGINAL TEXT:\n{text}\n---\n\n"
    writehistory(logging_text)
    return tags

text = """
Eligibility
To qualify for admission to the Master of International Trade Law and Economics, you must have earned your bachelor's degree in any field, within or outside of the United States, by the time you enroll at USC Gould School of Law. Students are required to have a solid foundation in mathematics at the university level, including calculus. If you hold a degree from a university within the United States, USC Graduate Admissions requires that it must be a regionally accredited institution.
If you are currently earning your degree and will complete it prior to the start of your studies, you may be admitted, but will be required to complete certain continuing registration requirements. You must submit official final transcripts showing your degree was awarded before you may begin your studies.
We do not require work experience or an LSAT or a GRE score to be considered for admission to any of our graduate degrees. Should you elect to submit an LSAT or GRE score report to supplement your application, such score will not play a key role in our admission decision-making process to ensure an equitable process for all applicants.
Application Deadlines
USC Gould School of Law offers two starts in fall and spring. Applications become available in September for Fall semester and in June for Spring semester. To receive priority consideration, apply by our Priority Deadline (see date below for each semester). After our Priority Deadline, applications will only be accepted on a rolling basis as space is available until the final application deadline.
Below are the start dates and application deadlines:
Program Start Date
Priority Deadline
Final Application Deadline
Fall 2024: August 26, 2024
February 1, 2024
June 1, 2024*
Spring 2025: January 13, 2025
September 1, 2024
November 1, 2024*

*Final Application Deadline for international students who require initial I-20 documentation for their student visas is May 1, 2024 for Fall semester and October 1, 2024 for Spring semester. We highly encourage you to apply by our Priority Deadline in order to provide sufficient time to process any required documents, including I-20 documentation for student visas, and to apply for USC campus housing.
If you are planning to take the TOEFL or IELTS exam after our Priority Deadline, we highly recommend that you go ahead and submit your application (and any other completed materials) prior to taking the exam and we will accept your test scores after you receive your official results. The Admissions Committee will make a final admissions decision upon receipt of your complete application file.
All applicants will be automatically considered for a scholarship award and do not need to submit any further documents to request an award. In addition, applicants who apply by our Priority Deadline will be automatically considered for a housing stipend.
Application Fee
Applicants will be charged a $90 application fee.
Application Process
To apply for admission, the following information and documents must be submitted:
1. Online USC Graduate Admission Application
Note: Once you start your application by clicking above, you will be directed to complete your online application through the USC Office of Graduate Admission application.
2. Personal Statement
Following the directions in your online application, provide a two or three-page document that includes your personal, academic, and professional background and your reasons for pursuing the Master of International Trade Law and Economics degree. Please include a description of your quantitative education or experience such as coursework or research projects in math or economics.
3. Resume/Curriculum Vitae (CV)
Following the directions in your online application, provide a record of your employment history and list of any distinctions, publications, and licenses/credentials.
4. Official Transcripts and Degree Verification
You must have earned your bachelor's degree by the time you enroll at USC. Following the directions provided in your online application, submit official transcripts from all institutions attended.
Upload scanned copies of your official, up-to-date transcripts into the USC online application system for review for admissions purposes. Please see information on the USC Graduate Admissions website:
Transcripts Requirements
FAQ
Video - Academic History
Country Requirements
If you hold any degree from outside of the U.S., USC does not accept or recognize credential evaluation reports from outside agencies; only records issued by your previous institution will be accepted. You may be required to submit English translations of transcripts and diploma(s)/degree certificate(s) - review country requirements above for further information. If you are admitted to USC, you may receive instructions to verify your degree with the International Education Research Foundation (IERF). For detailed information on how to obtain an IERF verification report, please refer to the Degree Progress website.
5. English Proficiency
If you are not a U.S. citizen or permanent resident, you must take the Test of English as Foreign Language (TOEFL) or International English Language Testing System (IELTS). TOEFL/IELTS waiver requests will only be considered if (1) your native language is English, or (2) you possess an undergraduate degree, master’s degree, or doctoral degree from an institution in which English is the primary language of instruction in accordance with the USC Graduate Admissions Policy.
USC Gould does not have a minimum TOEFL or IELTS score; however, we recommend a 90 TOEFL iBT and a 7.0 IELTS or above.
Please have your official scores sent to USC. Instructions on how to submit scores can be found on the USC Graduate Admissions website.
Test scores are valid within two years (24 months) of the application date. 
USC does NOT accept MyBest TOEFL scores of applicants as valid proof of language proficiency. We will consider the single best total TOEFL or IELTS score you submit along with the sub-scores from that test.
The LSAT or GRE exam is not required for admission. In addition, letters of recommendation are not required.
The Admissions Committee will not review your application until all transcripts and supporting documents are received. It is your responsibility to ensure that all of your documents are submitted by the final application deadline. Be sure to regularly check the email address you have provided on the application in the event that the Admissions Committee requests additional materials or information.
Application Status Check
To ensure you have submitted a complete application, please refer to the USC Graduate Admissions checklists. Our office will contact you via email for any missing or additional requirements. Please contact our office directly at mitle@law.usc.edu with any questions.
Admission Decisions
Admissions decisions are made upon receipt of complete application files. Incomplete applications are not reviewed. We encourage you to apply by our Priority Deadline to provide sufficient time for you to complete your application file. Our Admissions Committee will provide admissions decisions generally within 3-4 weeks upon our receipt of completed applications, beginning in January for Fall semester start and beginning in July for Spring semester start. You will be notified via email and postal mail. No admissions decisions will be released over the phone. We will contact you if the Admissions Committee recommends an interview or additional submissions.
Financial Statement
If you are admitted, you will be required to provide a financial statement that certifies that you have sufficient funds available to meet your living (housing, meals, etc.) and tuition expenses while at USC (unless you are a U.S. citizen or permanent resident, or have been granted political asylum). At that time, you also must submit a copy of the photo page of your passport (and passport copies of any dependent(s) traveling with you).
"""

a = extract_keys(text, 1,0.32)
console.print(f"[bold]Keywords: {a}")

filename = 'Magicoder Source Code Is All You Need.txt'
with open(filename, encoding="utf8") as f:
  fulltext = f.read()
f.close()
console.print("Text has been saved into variable [bold]fulltext")
title = 'Magicoder: Source Code Is All You Need'
filename = '2023-12-03 18.41.12 Governing societies with Artificial Intelligence.txt'
author = 'Yuxiang Wei, Zhe Wang, Jiawei Liu, Yifeng Ding, Lingming Zhang'
url = 'https://arxiv.org/pdf/2312.02120v1.pdf'

from langchain.document_loaders import TextLoader
from langchain.text_splitter import TokenTextSplitter

text_splitter = TokenTextSplitter(chunk_size=350, chunk_overlap=10)
splitted_text = text_splitter.split_text(fulltext)

console.print(len(splitted_text))
console.print("---")
console.print(splitted_text[0])

keys = []
for i in trange(0,len(splitted_text)):
  text = splitted_text[i]
  keys.append({'document' : filename,
              'title' : title,
              'author' : author,
              'url' : url,
              'doc': text,
              'keywords' : extract_keys(text, 1, 0.34)
  })

console.print(keys[1])

############### CREATE cHUnKS DOC DATABASE ##################
from langchain.schema.document import Document
goodDocs = []
for i in range(0,len(keys)):
  goodDocs.append(Document(page_content = keys[i]['doc'],
                          metadata = {'source': keys[i]['document'],
                              'type': 'chunk',
                              'title': keys[i]['title'],
                              'author': keys[i]['author'],
                              'url' : keys[i]['url'],
                              'keywords' : keys[i]['keywords']
                              }))

console.print(goodDocs[1])