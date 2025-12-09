import os 

if os.getenv('ENV') != 'DUMMY':
    from google import genai
    from google.genai import types

    import requests

    from services.storage_service import Agent


    try: 
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    except Exception as e:
        print("WARNING: Could not initialize Gemini Client. Check GEMINI_API_KEY.")
        client = None

    HF_TOKEN = os.getenv("HF_API_KEY")
    if not HF_TOKEN:
        print("WARNING: HF_API_KEY environment variable not set. HF Serverless models may fail.")

    GEMINI_API_MODELS = [
        'gemini-2.5-flash'
    ]

    HF_SERVERLESS_MODELS = [
        "deepseek-ai/DeepSeek-R1:sambanova", 
        "zai-org/GLM-4.5"
    ]

    HF_API_URL = f"https://router.huggingface.co/v1/chat/completions"

    def _call_gemini_api(agent: Agent, contents: list[types.Content]) -> str:
        if client is None: return "ERROR: Gemini client not initialized."
        
        sys_prompt = agent.get_system_prompt()
        config = types.GenerateContentConfig(system_instruction=sys_prompt)
        response = client.models.generate_content(
            model=agent.model_name,
            contents=contents, # Pre-converted list of types.Content
            config=config
        )
        return response.text


    def _call_hf_api(agent: Agent, messages: list[dict[str, str]]) -> str:
        if not HF_TOKEN: return "ERROR: HF_API_KEY not set."

        headers = {
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json"
        }

        payload = {
            "messages": messages,
            "model": agent.model_name,
            "temperature": 0.8
        }

        response = requests.post(HF_API_URL, headers=headers, json=payload)
        response.raise_for_status()

        data = response.json()
        return data['choices'][0]['message']['content']


    def generate_llm_response(agent: Agent, conversation_history: list[dict[str, str]], user_prompt: str) -> str:
        """
        Generates a text response from the specified LLM agent.
        conversation_history is a list of {'role': 'user'/'model', 'content': ''}.
        """
        if client is None:
            return "ERROR: LLM API not available."

        if not agent:
            raise ValueError("No agent provided.")

        model_name = agent.model_name
        if model_name is None:
            raise ValueError("Agent model_name is not defined.")
        if model_name in GEMINI_API_MODELS:
            contents = [
                types.Content(
                    role=msg.role,
                    parts=[types.Part.from_text(text=msg.content)]
                ) for msg in conversation_history
            ]
            print(user_prompt)
            contents.append(
                types.Content(
                    role='user', parts=[types.Part.from_text(text=user_prompt)]
                )
            )
            
            try:
                return _call_gemini_api(agent, contents)
            except Exception as e:
                return f"ERROR: Gemini API error - {e}"
        elif model_name in HF_SERVERLESS_MODELS:
            messages = [
                {
                    "role": "system",
                    "content": agent.get_system_prompt()
                }
            ]

            for msg in conversation_history:
                role = "assistant" if msg.role == 'model' else "user"
                messages.append({
                    "role": role,
                    "content": msg.content
                })

            messages.append({"role": "user", "content": user_prompt})

            try:
                return _call_hf_api(agent, messages)
            except Exception as e:
                return f"ERROR: HF Serverless API error - {e}"
        else:
            return f"ERROR: Model '{model_name}' not configured for API routing."
