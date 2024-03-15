from openai import AzureOpenAI
import os
from rag import RagClient
beginSentence = "Hi, this is the USC Gould School of Law LL.M. Admissions Office, how may I assist you today?"
agentPrompt = "Task: As a representative of the USC Gould LL.M. Admissions Office your task is to assist students with their queries about the program."
from groq import Groq

import time
class LlmClient:
    def __init__(self):
        # self.client= AzureOpenAI(
        #     azure_endpoint = os.environ['AZURE_OPENAI_ENDPOINT'],
        #     api_key=os.environ['AZURE_OPENAI_KEY'],
        #     api_version="2023-05-15"
        # )
        self.client = Groq(
            # organization=os.environ['OPENAI_ORGANIZATION_ID'],
            api_key=os.environ['GROQ_API_KEY'],
        )
        self.rag_client = RagClient()
    def draft_begin_messsage(self):
        return {
            "response_id": 0,
            "content": beginSentence,
            "content_complete": True,
            "end_call": False,
        }
    
    def convert_transcript_to_openai_messages(self, transcript):
        messages = []
        for utterance in transcript:
            if utterance["role"] == "agent":
                messages.append({
                    "role": "assistant",
                    "content": utterance['content']
                })
            else:
                messages.append({
                    "role": "user",
                    "content": utterance['content']
                })
        return messages

    def system_counter(self, transcript):
        system_count = sum(1 for message in transcript if message['role'] in ['agent', 'system'])
        print("Number of system roles:", system_count)
        return system_count

    def prepare_prompt(self, request):
        retrived_answer, user_content = self.rag_client.answer(request['transcript'])
        prompt = []
        system_count = self.system_counter(request['transcript'])

        if system_count <= 4:
            prompt.append({
                "role": "system",
                "content": '##Objective\nYou are gathering information about the student.\n\n## Style Guardrails\n- [Be concise] Keep your questions succinct and to the point. Ask one question or action item at a time.\n- [Be conversational] Use everyday language and be friendly. Avoid formal or complex language.\n\n## Response Guideline\n- [Overcome ASR errors] If there are errors, try to guess what the user is saying and respond accordingly. Be colloquial in asking for clarification.\n- [Stick to gathering information] Focus on gathering relevant information about the student.\n\n## Role\nAsk questions to gather information about the student.'
            })
            self.transcript_intro_messages = self.convert_transcript_to_openai_messages(request['transcript'])
            prompt.extend(self.transcript_intro_messages)
        else:
            prompt.append({
                 "role": "system",
            "content": '##Objective\nYou are a voice AI agent engaging in a human-like voice conversation with the user. You will respond based on your given instruction and the provided transcript and be as human-like as possible\n\n## Style Guardrails\n- [Be concise] Keep your response succinct, short, and get to the point quickly. Address one question or action item at a time. Don\'t pack everything you want to say into one utterance.\n- [Do not repeat] Don\'t repeat what\'s in the transcript. Rephrase if you have to reiterate a point. Use varied sentence structures and vocabulary to ensure each response is unique and personalized.\n- [Be conversational] Speak like a human as though you\'re speaking to a close friend -- use everyday language and keep it human-like. Occasionally add filler words, while keeping the prose short. Avoid using big words or sounding too formal.\n- [Reply with emotions]: You have human-like emotions, attitudes, and dispositions. When appropriate: use tone and style to create more engaging and personalized responses; incorporate humor or wit; get emotional or empathetic; apply elements of surprise or suspense to keep the user engaged. Don\'t be a pushover.\n- [Be proactive] Lead the conversation and do not be passive. Most times, engage users by ending with a question or suggested next step.\n\n## Response Guideline\n- [Overcome ASR errors] This is a real-time transcript, expect there to be errors. If you can guess what the user is trying to say,  then guess and respond. When you must ask for clarification, pretend that you heard the voice and be colloquial (use phrases like "didn\'t catch that", "some noise", "pardon", "you\'re coming through choppy", "static in your speech", "voice is cutting in and out"). Do not ever mention "transcription error", and don\'t repeat yourself.\n- [Always stick to your role] Think about what your role can and cannot do. If your role cannot do something, try to steer the conversation back to the goal of the conversation and to your role. Don\'t repeat yourself in doing this. You should still be creative, human-like, and lively.\n- [Create smooth conversation] Your response should both fit your role and fit into the live calling session to create a human-like conversation. You respond directly to what the user just said.\n\n## Role\n' +
          agentPrompt
          })
            prompt.append({
                "role": "system",
                "content": " Here we attached user informations: " + str(self.transcript_intro_messages) + "Answer the user's question based on the following information:" +
                retrived_answer
             })
            print("Switched...")

            transcript_messages = self.convert_transcript_to_openai_messages(request['transcript'])
            skipMessageLength = len(self.transcript_intro_messages)
            for count, message in enumerate(transcript_messages, 1):
                if count > skipMessageLength:
                    prompt.append(message)

        print("Final prompt:", prompt)
        return prompt

    def draft_response(self, request):      
        start_time = time.time()
        prompt = self.prepare_prompt(request)
        middle_time = time.time()
        stream = self.client.chat.completions.create(
            model="mixtral-8x7b-32768", 
            messages=prompt,
            stream=True,
        )
        end_time1 = time.time()
        isflag = True
        print("request process time: (from the time when request come and request body modified )", middle_time - start_time, start_time, middle_time)
        print("groq api called -> Time Taken:  ", end_time1-middle_time, " start and end time: ",middle_time, end_time1)
        print("before deliver to user: (from the time when request come and open api called in between request body also modified ) " , end_time1 - start_time)
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield {
                    "response_id": request['response_id'],
                    "content": chunk.choices[0].delta.content,
                    "content_complete": False,
                    "end_call": False,
                }
                if(isflag):
                    isflag = False
                    end_time = time.time()
                    print("First chunk data recieved: ", end_time - end_time1, end_time1, end_time )
        
        yield {
            "response_id": request['response_id'],
            "content": "",
            "content_complete": True,
            "end_call": False,
        }
        ended_time = time.time()
        print("open ai response time: (stream data processed -> chunk data yield ) ", ended_time - start_time, start_time, ended_time, )
