from langchain.schema import Document

import os
# Example usage

def create_documents_from_files(file_metadata_list):
    documents = []
    for file_metadata in file_metadata_list:
        filename = file_metadata["filename"]
                
        current_directory = os.getcwd()
        print("Current directory:", current_directory)
        print(filename)
        with open(filename, encoding="utf8") as file:
            page_content = file.read()
        # file.close()
        document = Document(
            page_content=page_content,
            metadata=file_metadata
        )
        documents.append(document)
    
    return documents


dataset =[
            
            {
            'Title':"Career Opportunities - Master of International Trade Law and Economics (MITLE)",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Career Opportunities - Master of International Trade Law and Economics (MITLE).txt',
            "tags" : "ucla"
            },
            {
            'Title':"Curriculum - Alternative Dispute Resolution Certificate",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Curriculum - Alternative Dispute Resolution Certificate.txt',
            "tags" : "ucla"
            },
            {
            'Title':"Curriculum - LLM in ADR",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Curriculum - LLM in ADR.txt',
            "tags" : "ucla"
            },
            {
            'Title':"Curriculum - LLM in International Business and Economic Law",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Curriculum - LLM in International Business and Economic Law.txt',
            "tags" : "ucla"
            },
            {
            'Title':"Curriculum - LLM in Privacy Law and Cybersecurity",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Curriculum - LLM in Privacy Law and Cybersecurity.txt',
            "tags" : "ucla"
            },
            {
            'Title':"Curriculum - Master of Comparative Law (MCL)",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Curriculum - Master of Comparative Law (MCL).txt',
            "tags" : "ucla"
            },
            {
            'Title':"Curriculum - Master of International Trade Law and Economics (MITLE)",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Curriculum - Master of International Trade Law and Economics (MITLE).txt',
            "tags" : "ucla"
            },
            {
            'Title':"Curriculum - Master of Laws (LLM) - Online",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Curriculum - Master of Laws (LLM) - Online.txt',
            "tags" : "ucla"
            },
            {
            'Title':"Curriculum Master of Dispute Resolution (MDR)",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Curriculum Master of Dispute Resolution (MDR).txt',
            "tags" : "ucla"
            },
            {
            'Title':"LLM - 1 year",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/LLM - 1 year.txt',
            "tags" : "ucla"
            },
            {
            'Title':"LLM in International Business and Economic Law",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/LLM in International Business and Economic Law.txt',
            "tags" : "ucla"
            },
            {
            'Title':"Master of Comparative Law (MCL) Degree",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Master of Comparative Law (MCL) Degree.txt',
            "tags" : "ucla"
            },
            {
            'Title':"Master of International Trade Law and Economics (MITLE)",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Master of International Trade Law and Economics (MITLE).txt',
            "tags" : "ucla"
            },
            {
            'Title':"Master of Laws (LLM) - Online",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Master of Laws (LLM) - Online.txt',
            "tags" : "ucla"
            },
            {
            'Title':"Master of Laws (LLM) in Privacy Law and Cybersecurity",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Master of Laws (LLM) in Privacy Law and Cybersecurity.txt',
            "tags" : "ucla"
            },
            {
            'Title':"Master of Laws in Alternative Dispute Resolution (LLM in ADR)",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Master of Laws in Alternative Dispute Resolution (LLM in ADR).txt',
            "tags" : "ucla"
            },
            {
            'Title':"Master of Science in Innovation Economics, Law and Regulation (MIELR)",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Master of Science in Innovation Economics, Law and Regulation (MIELR).txt',
            "tags" : "ucla"
            },
            {
            'Title':"Tuition & Financial Aid - 1 yr Master of Laws (LLM) Degree",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Tuition & Financial Aid - 1 yr Master of Laws (LLM) Degree.txt',
            "tags" : "ucla"
            },
            {
            'Title':"Tuition & Financial Aid - Alternative Dispute Resolution Certificate",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Tuition & Financial Aid - Alternative Dispute Resolution Certificate.txt',
            "tags" : "ucla"
            },
            {
            'Title':"Tuition & Financial Aid - LLM in ADR",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Tuition & Financial Aid - LLM in ADR.txt',
            "tags" : "ucla"
            },
            {
            'Title':"Tuition & Financial Aid - Master of Comparative Law (MCL)",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Tuition & Financial Aid - Master of Comparative Law (MCL).txt',
            "tags" : "ucla"
            },
            {
            'Title':"Tuition & Financial Aid - Master of Dispute Resolution (MDR)",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Tuition & Financial Aid - Master of Dispute Resolution (MDR).txt',
            "tags" : "ucla"
            },
            {
            'Title':"Tuition & Financial Aid LLM in International Business and Economic Law",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Tuition & Financial Aid LLM in International Business and Economic Law.txt',
            "tags" : "ucla"
            },
            {
            'Title':"Tuition and Financial Aid - Master of International Trade Law and Economics (MITLE)",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Tuition and Financial Aid - Master of International Trade Law and Economics (MITLE).txt',
            "tags" : "ucla"
            },
            {
            'Title':"Tuition and Financial Aid Master of Laws (LLM) - Online",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Tuition and Financial Aid Master of Laws (LLM) - Online.txt',
            "tags" : "ucla"
            },
            {
            'Title':"Two-Year Extended Master of Laws (LLM)",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Two-Year Extended Master of Laws (LLM).txt',
            "tags" : "ucla"
            },
            {
            'Title':"Types of Aid - LLM in Privacy Law and Cybersecurity",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Types of Aid - LLM in Privacy Law and Cybersecurity.txt',
            "tags" : "ucla"
            },
            {
            'Title':"USC Gould -  Master of Dispute Resolution (MDR)",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/USC Gould -  Master of Dispute Resolution (MDR).txt',
            "tags" : "ucla"
            },
            {
            'Title':"USC Gould - Centre for Dispute Resolution_",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/USC Gould - Centre for Dispute Resolution_.txt',
            "tags" : "ucla"
            },
            {
            'Title':"USC Gould - Housing Flyer",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/USC Gould - Housing Flyer.txt',
            "tags" : "ucla"
            },
            {
            'Title':"USC Gould - Master of Comparative Law (MCL)",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/USC Gould - Master of Comparative Law (MCL).txt',
            "tags" : "ucla"
            },
            {
            'Title':"Alternative Dispute Resolution (ADR) Certificate",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Alternative Dispute Resolution (ADR) Certificate.txt',
            "tags" : "ucla"
            },
            {
            'Title':"Application Instructions - Alternate Dispute Resolution (ADR) Certificate",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Application Instructions - Alternate Dispute Resolution (ADR) Certificate.txt',
            "tags" : "ucla"
            },
            {
            'Title':"Application Instructions - Master of Comparative Law (MCL)",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Application Instructions - Master of Comparative Law (MCL).txt',
            "tags" : "ucla"
            },
            {
            'Title':"Application Instructions - Master of Dispute Resolution (MDR)",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Application Instructions - Master of Dispute Resolution (MDR).txt',
            "tags" : "ucla"
            },
            {
            'Title':"Application Instructions - Master of International Trade Law and Economics (MITLE)",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Application Instructions - Master of International Trade Law and Economics (MITLE).txt',
            "tags" : "ucla"
            },
            {
            'Title':"Application Instructions Extended Master of Laws (LLM)",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Application Instructions Extended Master of Laws (LLM).txt',
            "tags" : "ucla"
            },
            {
            'Title':"Application Instructions LLM in ADR",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Application Instructions LLM in ADR.txt',
            "tags" : "ucla"
            },
            {
            'Title':"Application Instructions LLM in International Business and Economic Law",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Application Instructions LLM in International Business and Economic Law.txt',
            "tags" : "ucla"
            },
            {
            'Title':"Application Instructions LLM in Privacy Law and Cybersecurity",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Application Instructions LLM in Privacy Law and Cybersecurity.txt',
            "tags" : "ucla"
            },
            {
            'Title':"Application Instructions Master of Comparative Law (MCL)",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Application Instructions Master of Comparative Law (MCL).txt',
            "tags" : "ucla"
            },
            {
            'Title':"Application Instructions Master of Laws (LLM) - Online",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Application Instructions Master of Laws (LLM) - Online.txt',
            "tags" : "ucla"
            },
            {
            'Title':"Application Instructions Master of Laws (LLM) Degree",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Application Instructions Master of Laws (LLM) Degree.txt',
            "tags" : "ucla"
            },
            {
            'Title':"Bar and Certificate Tracks Master of Laws (LLM) - Online",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Bar and Certificate Tracks Master of Laws (LLM) - Online.txt',
            "tags" : "ucla"
            },
            {
            'Title':"Campus Housing - 1 yr Master of Laws (LLM) Degree",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Campus Housing - 1 yr Master of Laws (LLM) Degree.txt',
            "tags" : "ucla"
            },
            {
            'Title':"Campus Housing - LLM in ADR",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Campus Housing - LLM in ADR.txt',
            "tags" : "ucla"
            },
            {
            'Title':"Campus Housing - LLM in International Business and Economic Law",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Campus Housing - LLM in International Business and Economic Law.txt',
            "tags" : "ucla"
            },
            {
            'Title':"Campus Housing - LLM in Privacy Law and Cybersecurity",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Campus Housing - LLM in Privacy Law and Cybersecurity.txt',
            "tags" : "ucla"
            },
            {
            'Title':"Campus Housing - Master of Comparative Law (MCL)",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Campus Housing - Master of Comparative Law (MCL).txt',
            "tags" : "ucla"
            },
            {
            'Title':"Campus Housing - Master of Dispute Resolution (MDR)",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Campus Housing - Master of Dispute Resolution (MDR).txt',
            "tags" : "ucla"
            },
            {
            'Title':"Campus Housing - Master of International Trade Law and Economics (MITLE)",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Campus Housing - Master of International Trade Law and Economics (MITLE).txt',
            "tags" : "ucla"
            },
            {
            'Title':"Career and Bar - 1 yr Master of Laws (LLM) Degree",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Career and Bar - 1 yr Master of Laws (LLM) Degree.txt',
            "tags" : "ucla"
            },
            {
            'Title':"Career and Bar - LLM in ADR",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Career and Bar - LLM in ADR.txt',
            "tags" : "ucla"
            },
            {
            'Title':"Career and Bar - LLM in International Business and Economic Law",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Career and Bar - LLM in International Business and Economic Law.txt',
            "tags" : "ucla"
            },
            {
            'Title':"Career and Bar - LLM in Privacy Law and Cybersecurity",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Career and Bar - LLM in Privacy Law and Cybersecurity.txt',
            "tags" : "ucla"
            },
            {
            'Title':"Career and Bar Master of Comparative Law (MCL)",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Career and Bar Master of Comparative Law (MCL).txt',
            "tags" : "ucla"
            },
            {
            'Title':"Career Opportunities - Master of Dispute Resolution (MDR)",
            'URL': "https://gould.usc.edu/academics/degrees/online-llm/application/",
            'filename' :'files/Career Opportunities - Master of Dispute Resolution (MDR).txt',
            "tags" : "ucla"
            },
        ]
docs = create_documents_from_files(dataset)