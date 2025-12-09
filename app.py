"""
Multi-Agent Backend Orchestrator
This FastAPI server orchestrates interactions between multiple agents, each with their own LLM and TTS capabilities.
It exposes an endpoint for the frontend to request the next turn in the conversation.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
from pathlib import Path

from dotenv import load_dotenv

flag = load_dotenv(dotenv_path=Path(__file__).parent / ".env")
if flag:
    print(".env file loaded successfully.")

ENV = os.getenv('ENV')

# Import the services
if ENV != 'DUMMY':
    from services.llm_service import generate_llm_response, GEMINI_API_MODELS, HF_SERVERLESS_MODELS
    from services.tts_service import generate_audio_base64, basic_clean_text, DEVICE # The model is loaded on import
    from services.storage_service import Agent, AgentStorage

    agent_storage = AgentStorage()

from schemas.response_schemas import AgentMessage, AgentList, AgentItem, BaseResponse, ModelItem
from schemas.request_schemas import ConversationRequest, AgentUpdateRequest

PROJ_DIR = Path(__file__).parent

# --- FastAPI App Setup ---
app = FastAPI(title="Multi-Agent Backend Orchestrator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.head("/health_check")
def check():
    return {
        "status": "success",
        "message": "works well"
    }, 200


@app.post("/next_turn", response_model=BaseResponse[AgentMessage])
def next_turn(request: ConversationRequest):
    """
    Orchestrates the next turn for the specified agent (LLM + TTS).
    """
    current_agent_id = request.next_speaker_id

    if ENV == 'DUMMY':
        def load_base64_audio(file_name: str) -> str:
            path = Path(file_name)

            if not path.exists():
                raise FileNotFoundError(f"{file_name} not found")

            # Read and strip whitespace/newlines
            return path.read_text(encoding="utf-8").strip()

        audio_base64 = load_base64_audio("base64_sample.txt")

        return BaseResponse(
            status="success",
            message="WARNING: THIS IS DUMMY RESPONSE",
            data=AgentMessage(
                speaker_id=current_agent_id,
                text="<think>So here is the example of how LLM thinking will be shown in the response schema. Please do not show it in the response box</think> This is the sample response that you may get from the LLM response",
                audio_base64=audio_base64
            )
        )
    
    # 1. Get the LLM response
    agent = agent_storage.get_agent(current_agent_id)
    llm_text = generate_llm_response(
        agent=agent,
        conversation_history=request.conversation_history,
        user_prompt=request.user_prompt
    )

    if llm_text.startswith("ERROR"):
        raise HTTPException(status_code=500, detail=llm_text)

    print(f"LLM Response from {current_agent_id}:\n\n {llm_text}")
    if '<think>' in llm_text:
        cleaned_text = basic_clean_text(llm_text)
    else:
        cleaned_text = llm_text
    print(cleaned_text)

    try:
        # Map the agent ID to the speaker file name (e.g., AGENT_ME -> me)
        audio_base64 = generate_audio_base64(
            text=cleaned_text,
            agent=agent,
            language="ru" if current_agent_id == 'DRAGUNOV' else 'en'
        )
    except Exception as e:
        print(f"TTS Error: {e}")
        raise HTTPException(status_code=500, detail=f"TTS Generation failed: {e}")
    
    return BaseResponse(
        status="success",
        data=AgentMessage(
            speaker_id=current_agent_id,
            text=llm_text,
            audio_base64=audio_base64
        )
    )


def dummy_agent_handler(agent_id: str = "NASSEER", model: str = "gemini-2.5-flash") -> AgentItem:
    return AgentItem(
        agent_id=agent_id,
        model_name=model
    )


@app.get("/agent/{agent_id}", response_model=BaseResponse[AgentItem])
def get_agent(agent_id: str = None):
    """
    Retrieve details of a specific persona by ID.
    """

    if ENV == 'DUMMY':
        return BaseResponse(
            status="success",
            message="WARNING: THIS IS DUMMY RESPONSE",
            data=dummy_agent_handler()
        )
    agent: Agent = agent_storage.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found.")
    
    return BaseResponse(
        status="success",
        data=AgentItem(
            **agent.to_dict()
        )
    )


@app.get("/agents", response_model=BaseResponse[AgentList])
def list_agents():
    """
    List all available agents.
    """
    if ENV == 'DUMMY':
        agent_items = [
            dummy_agent_handler(agent_id='HONG'),
            dummy_agent_handler(agent_id='DRAGUNOV', model='zai-org/GLM-4.5')
        ]

        return BaseResponse(
            status="success",
            message="WARNING: THIS IS DUMMY RESPONSE",
            data=AgentList(
                total=len(agent_items),
                agents=agent_items
            )
        )
    agent_items = [
        AgentItem(
            **agent.to_dict()
        )
        for agent in agent_storage.get_all_agents().values()
    ]
    return BaseResponse(
        status="success",
        data=AgentList(
            total=len(agent_items),
            agents=agent_items
        )
    )


@app.put('/agents/{agent_id}', response_model=BaseResponse[AgentItem])
def update_agent(agent_id: str, request: AgentUpdateRequest):
    """
    Update an existing agent's details.
    """
    if ENV == 'DUMMY':
        return BaseResponse(
            status='success',
            message='WARNING: THIS IS DUMMY RESPONSE',
            data=dummy_agent_handler()
        )
    try:
        agent_storage.update_agent(
            agent_id=agent_id,
            model_name=request.model_name
        )
    except KeyError as ke:
        raise HTTPException(status_code=404, detail=str(ke))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    updated_agent = agent_storage.get_agent(agent_id)
    return BaseResponse(
        status="success",
        data=AgentItem(
            **updated_agent.to_dict()
        )
    )


@app.get("/models", response_model=BaseResponse[list[ModelItem]])
def get_model_list(provider_name=None):
    """
    Retrieve the list of available models from the specified provider.
    If no provider is specified, return models from all providers.
    """
    if ENV == 'DUMMY':
        return BaseResponse(
            status='success',
            message='WARNING: THIS IS DUMMY RESPONSE',
            data=[
                ModelItem(
                    model_name='gemini-2.5-flash',
                    provider='gemini'
                ),
                ModelItem(
                    model_name='zai-org/GLM-4.5',
                    provider='hf_serverless'
                )
            ]
        )
    models = []
    if provider_name is not None and provider_name not in ["gemini", "hf_serverless"]:
        raise HTTPException(status_code=400, detail=f"Invalid provider_name '{provider_name}'. Must be 'gemini' or 'hf_serverless'.")
    if provider_name is None or provider_name == "gemini":
        for model_name in GEMINI_API_MODELS:
            models.append(ModelItem(
                model_name=model_name,
                provider="gemini"
            ))
    if provider_name is None or provider_name == "hf_serverless":
        for model_name in HF_SERVERLESS_MODELS:
            models.append(ModelItem(
                model_name=model_name,
                provider="hf_serverless"
            ))
    
    return BaseResponse(
        status="success",
        data=models
    )   
