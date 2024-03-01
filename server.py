import json
import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.websockets import WebSocketState
from llm import LlmClient
import asyncio
import retellclient
from retellclient.models import operations, components
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

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
retell = retellclient.RetellClient(
    api_key=os.environ['RETELL_API_KEY']
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

@app.websocket("/llm-websocket/{call_id}")
async def websocket_handler(websocket: WebSocket, call_id: str):
    await websocket.accept()
    print(f"Handle llm ws for: {call_id}")

    # send first message to signal ready of server
    response_id = 0
    first_event = llm_client.draft_begin_messsage()
    await websocket.send_text(json.dumps(first_event))

    async def stream_response(request):
        nonlocal response_id
        for event in llm_client.draft_response(request):
            await websocket.send_text(json.dumps(event))
            if request['response_id'] < response_id:
                return # new response needed, abondon this one
    try:
        while True:
            message = await websocket.receive_text()
            request = json.loads(message)
            # print out transcript
            os.system('cls' if os.name == 'nt' else 'clear')
            print(json.dumps(request, indent=4))
            
            if 'response_id' not in request:
                continue # no response needed, process live transcript update if needed
            response_id = request['response_id']
            asyncio.create_task(stream_response(request))
            print("ws called done")
    except WebSocketDisconnect:
        print(f"LLM WebSocket disconnected for {call_id}")
    except Exception as e:
        print(f'LLM WebSocket error for {call_id}: {e}')
    finally:
        print(f"LLM WebSocket connection closed for {call_id}")


html = """
<!DOCTYPE html>
<html>
    <head>
        <title>Chat</title>
    </head>
    <body>
        <h1>WebSocket Chat</h1>
        <form action="" onsubmit="sendMessage(event)">
            <input type="text" id="messageText" autocomplete="off"/>
            <button>Send</button>
        </form>
        <ul id='messages'>
        </ul>
        <script>
            var ws = new WebSocket("ws://rag.collegeit.in/ws");
            ws.onmessage = function(event) {
                var messages = document.getElementById('messages')
                var message = document.createElement('li')
                var content = document.createTextNode(event.data)
                message.appendChild(content)
                messages.appendChild(message)
            };
            function sendMessage(event) {
                var input = document.getElementById("messageText")
                ws.send(input.value)
                input.value = ''
                event.preventDefault()
            }
        </script>
    </body>
</html>
"""


@app.get("/")
async def get():
    return HTMLResponse(html)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        await websocket.send_text(f"Message text was: {data}")

