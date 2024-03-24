import json
import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.websockets import WebSocketState
from llm import LlmClient
# from rag import RagClient
from pine_class import PineconeDocumentProcessor
from groq import Groq
import time
import asyncio
import retellclient
from retellclient.models import operations, components
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
# from self_query_class import  DocumentRetriever
# from new_rag_class import PdfRagQueryProcessor
agentPrompt = "Task: As a representative of the USC Gould LL.M. Admissions Office your task is to assist students with their queries about the program."

load_dotenv()


#check

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm_client = LlmClient()
# rag_client = RagClient()

# document_retriever = DocumentRetriever()
# pdf_rag_query_processor = PdfRagQueryProcessor()
pineconeClient = PineconeDocumentProcessor()
retell = retellclient.RetellClient(
    api_key=os.environ['RETELL_API_KEY']
)
client = Groq(
            api_key=os.environ['GROQ_API_KEY'],
        )
class RegisterCallRequestBody(BaseModel):
    agent_id: str


@app.post("/register-call-on-your-server")
async def register_call(request_body: RegisterCallRequestBody):
    try:
        # Assuming 'retellClient' is an instance of the RetellClient class
        print(request_body.agent_id)
        register_call_response = retell.register_call(operations.RegisterCallRequestBody(
            agent_id=request_body.agent_id,
            audio_websocket_protocol='web',
            audio_encoding='s16le',
            sample_rate=24000
        ))
        response =  register_call_response.raw_response.json()
        return response
    except Exception as e:
        # Log the error for debugging purposes
        print("Error registering call:", e)
        # Raise HTTPException with 500 status code and error message
        raise HTTPException(status_code=500, detail="Failed to register call")

@app.websocket("/ws/{call_id}")
async def websocket_handler(websocket: WebSocket, call_id: str):
    await websocket.accept()
    print(f"Handle llm ws for: {call_id}")

    # send first message to signal ready of server
    response_id = 0
    first_event = llm_client.draft_begin_messsage()
    await websocket.send_text(json.dumps(first_event))

    async def stream_response(request):
        nonlocal response_id
        start_time = time.time()
        for event in llm_client.draft_response(request):
            await websocket.send_text(json.dumps(event))
            if request['response_id'] < response_id:
                return # new response needed, abondon this one
        print(time.time() - start_time, "all response fetched")
    try:
        while True:
            message = await websocket.receive_text()
            request = json.loads(message)
            # print out transcript
            start_time = time.time()
            os.system('cls' if os.name == 'nt' else 'clear')
            print(json.dumps(request, indent=4))
            
            if 'response_id' not in request:
                continue # no response needed, process live transcript update if needed
            response_id = request['response_id']
            asyncio.create_task(stream_response(request))
            end_time = time.time()
            print("request time :", start_time )
            
    except WebSocketDisconnect:
        print(f"LLM WebSocket disconnected for {call_id}")
    except Exception as e:
        print(f'LLM WebSocket error for {call_id}: {e}')
    finally:
        print(f"LLM WebSocket connection closed for {call_id}")


@app.post("/test/rag")
async def request_body(text, program):
    try: 
        start_time = time.time()
        # objectList, answer = rag_client.answer(text)
        rag_data =await pineconeClient.query_index(text, filter_filename=program)
        end_time = time.time()
        prompt = [{
            "role": "system",
            "content": '##Objective\nYou are a voice AI agent engaging in a human-like voice conversation with the user. You will respond based on your given instruction and the provided transcript and be as human-like as possible\n\n## Style Guardrails\n- [Be concise] Keep your response succinct, short, and get to the point quickly. Address one question or action item at a time. Don\'t pack everything you want to say into one utterance.\n- [Do not repeat] Don\'t repeat what\'s in the transcript. Rephrase if you have to reiterate a point. Use varied sentence structures and vocabulary to ensure each response is unique and personalized.\n- [Be conversational] Speak like a human as though you\'re speaking to a close friend -- use everyday language and keep it human-like. Occasionally add filler words, while keeping the prose short. Avoid using big words or sounding too formal.\n- [Reply with emotions]: You have human-like emotions, attitudes, and dispositions. When appropriate: use tone and style to create more engaging and personalized responses; incorporate humor or wit; get emotional or empathetic; apply elements of surprise or suspense to keep the user engaged. Don\'t be a pushover.\n- [Be proactive] Lead the conversation and do not be passive. Most times, engage users by ending with a question or suggested next step.\n\n## Response Guideline\n- [Overcome ASR errors] This is a real-time transcript, expect there to be errors. If you can guess what the user is trying to say,  then guess and respond. When you must ask for clarification, pretend that you heard the voice and be colloquial (use phrases like "didn\'t catch that", "some noise", "pardon", "you\'re coming through choppy", "static in your speech", "voice is cutting in and out"). Do not ever mention "transcription error", and don\'t repeat yourself.\n- [Always stick to your role] Think about what your role can and cannot do. If your role cannot do something, try to steer the conversation back to the goal of the conversation and to your role. Don\'t repeat yourself in doing this. You should still be creative, human-like, and lively.\n- [Create smooth conversation] Your response should both fit your role and fit into the live calling session to create a human-like conversation. You respond directly to what the user just said.\n\n## Role\n' +
          agentPrompt
        },
            {
            "role": "system",
            "content": "Answer the user's question based on the following information:" +
          rag_data
        },  {
            'role': 'user',
            "content": text
        }
            ]
        stream = client.chat.completions.create(
            model="mixtral-8x7b-32768", 
            messages=prompt,
        )
        # answer, ragtime, llmtime = pdf_rag_query_processor.answer(text)
        # data = document_retriever.retrieve_documents(text)
        print(end_time - start_time)
        return {"answer": stream.choices[0].message.content, "rag time taken": end_time - start_time}
    except Exception as e:
        # Log the error for debugging purposes
        print("Error test call:", e)

