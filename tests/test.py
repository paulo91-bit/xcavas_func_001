# function_app.py
# This is the main file for the v2 programming model.
# UPDATED: This version now includes a 'negotiate' endpoint for WebSocket connections.

import logging
import json
import asyncio
import os
import base64
import openai
import azure.functions as func
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.aio import SearchClient
from azure.search.documents.models import VectorizedQuery
from pypdf import PdfReader
import io
from azure.messaging.webpubsubservice import WebPubSubServiceClient

# --- Create a FunctionApp instance ---
app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

# --- Helper function for robust chunking ---
def chunk_text(text: str, chunk_size_chars: int = 6000, overlap_chars: int = 200) -> list[str]:
    """
    Splits text into fixed-size chunks with overlap.
    """
    if not text or not text.strip():
        return []
    
    chunks = []
    start_index = 0
    text_length = len(text)
    
    while start_index < text_length:
        end_index = start_index + chunk_size_chars
        chunk = text[start_index:end_index]
        chunks.append(chunk)
        start_index += chunk_size_chars - overlap_chars
        
    return [c for c in chunks if c.strip()]

# --- HTTP Trigger 1: Document Indexing ---
@app.route(route="UploadAndIndex", methods=['post'])
async def UploadAndIndex(req: func.HttpRequest) -> func.HttpResponse:
    """
    Handles the document upload, chunking, embedding, and indexing process.
    """
    logging.info('Python UploadAndIndex trigger function processed a request.')
    try:
        # --- Get Configuration ---
        search_endpoint = os.environ["AZURE_SEARCH_ENDPOINT"]
        search_key = os.environ["AZURE_SEARCH_KEY"]
        search_index_name = os.environ["AZURE_SEARCH_INDEX_NAME"]
        azure_openai_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
        azure_openai_key = os.environ["AZURE_OPENAI_KEY"]
        azure_embedding_deployment = os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]
        azure_api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01")

        if 'document' not in req.files:
            return func.HttpResponse("Missing 'document' file in the request.", status_code=400)

        # --- Read and Parse Document ---
        document_file = req.files['document']
        document_text = ""
        if document_file.filename.lower().endswith('.pdf'):
            pdf_bytes = document_file.read()
            pdf_stream = io.BytesIO(pdf_bytes)
            reader = PdfReader(pdf_stream)
            for page in reader.pages:
                document_text += page.extract_text() + "\n"
        else:
            document_text = document_file.read().decode('utf-8', errors='replace')

        safe_filename = base64.urlsafe_b64encode(document_file.filename.encode('utf-8')).decode('utf-8')
        chunks = chunk_text(document_text)

        # --- Create Embeddings and Index ---
        ai_client = openai.AsyncAzureOpenAI(api_key=azure_openai_key, api_version=azure_api_version, azure_endpoint=azure_openai_endpoint)
        
        documents_to_index = []
        for i, chunk in enumerate(chunks):
            if chunk.strip():
                embedding_response = await ai_client.embeddings.create(input=chunk, model=azure_embedding_deployment)
                embedding = embedding_response.data[0].embedding
                documents_to_index.append({
                    "id": f"{safe_filename}-{i}",
                    "content": chunk,
                    "contentVector": embedding,
                    "source": document_file.filename
                })

        search_client = SearchClient(endpoint=search_endpoint, index_name=search_index_name, credential=AzureKeyCredential(search_key))
        async with search_client:
            await search_client.upload_documents(documents=documents_to_index)
        
        return func.HttpResponse(json.dumps({"status": "success", "indexedChunks": len(documents_to_index)}), mimetype="application/json")

    except KeyError as e:
        return func.HttpResponse(f"Server configuration error: Missing setting for {e}.", status_code=500)
    except Exception as e:
        return func.HttpResponse(f"An internal server error occurred: {e}", status_code=500)

# --- HTTP Trigger 2: Module Generation (RAG-enabled) ---
@app.route(route="GenerateTrainingModule", methods=['post'])
async def GenerateTrainingModule(req: func.HttpRequest) -> func.HttpResponse:
    """
    Generates a training module using the RAG pattern.
    """
    logging.info('Python GenerateTrainingModule trigger function processed a request.')
    try:
        # --- Get Configuration ---
        search_endpoint = os.environ["AZURE_SEARCH_ENDPOINT"]
        search_key = os.environ["AZURE_SEARCH_KEY"]
        search_index_name = os.environ["AZURE_SEARCH_INDEX_NAME"]
        azure_openai_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
        azure_openai_key = os.environ["AZURE_OPENAI_KEY"]
        azure_openai_deployment = os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]
        azure_embedding_deployment = os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]
        azure_api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01")

        # --- Parse & Validate Request ---
        form_data = req.form
        req_body = {key: form_data.get(key) for key in form_data}
        training_request = TrainingRequest(req_body)
        validation_error = training_request.validate()
        if validation_error: return func.HttpResponse(validation_error, status_code=400)

        # --- RAG Retrieval Step ---
        ai_client = openai.AsyncAzureOpenAI(api_key=azure_openai_key, api_version=azure_api_version, azure_endpoint=azure_openai_endpoint)
        query_embedding_response = await ai_client.embeddings.create(input=training_request.topic, model=azure_embedding_deployment)
        query_vector = query_embedding_response.data[0].embedding

        search_client = SearchClient(endpoint=search_endpoint, index_name=search_index_name, credential=AzureKeyCredential(search_key))
        retrieved_context = ""
        async with search_client:
            vector_query = VectorizedQuery(vector=query_vector, k_nearest_neighbors=3, fields="contentVector")
            results = await search_client.search(search_text=None, vector_queries=[vector_query])
            async for result in results:
                retrieved_context += result['content'] + "\n\n"
        
        training_request.deepContext = retrieved_context

        # --- Generation Step ---
        ai_service = GenerativeAiService(api_key=azure_openai_key, azure_endpoint=azure_openai_endpoint, api_version=azure_api_version, azure_deployment=azure_openai_deployment, logger=logging)
        training_module = await ai_service.generate_module_async(training_request)
        
        training_module['requestDetails'] = training_request.to_dict()
        return func.HttpResponse(json.dumps(training_module), mimetype="application/json")

    except KeyError as e:
        return func.HttpResponse(f"Server configuration error: Missing setting for {e}.", status_code=500)
    except Exception as e:
        return func.HttpResponse(f"An internal server error occurred: {e}", status_code=500)

# --- HTTP Trigger 3: Contextual Q&A Chat ---
@app.route(route="Chat", methods=['post'])
async def ChatHandler(req: func.HttpRequest) -> func.HttpResponse:
    """
    Handles a context-question-answer request.
    """
    logging.info('Python Chat trigger function processed a request.')
    try:
        # --- Get Configuration & Parse Request ---
        azure_openai_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
        azure_openai_key = os.environ["AZURE_OPENAI_KEY"]
        azure_openai_deployment = os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]
        azure_api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01")
        req_body = req.get_json()
        chat_request = ChatRequest(req_body)
        validation_error = chat_request.validate()
        if validation_error: return func.HttpResponse(validation_error, status_code=400)
        
        # --- Get AI Response ---
        ai_service = GenerativeAiService(api_key=azure_openai_key, azure_endpoint=azure_openai_endpoint, api_version=azure_api_version, azure_deployment=azure_openai_deployment, logger=logging)
        response_json = await ai_service.contextual_chat_async(chat_request)
        
        return func.HttpResponse(json.dumps(response_json), mimetype="application/json")

    except KeyError as e:
        return func.HttpResponse(f"Server configuration error: Missing setting for {e}.", status_code=500)
    except Exception as e:
        return func.HttpResponse(f"An internal server error occurred: {e}", status_code=500)

# --- HTTP Trigger 4: WebSocket Connection Negotiator ---
@app.route(route="negotiate")
def negotiate(req: func.HttpRequest) -> func.HttpResponse:
    """
    Generates a secure URL for the client to connect to the WebSocket service.
    """
    logging.info("Negotiate function processed a request.")
    try:
        connection_string = os.environ["WEB_PUBSUB_CONNECTION_STRING"]
        hub_name = "xcanvas-hub" # You can make this an environment variable if needed

        service_client = WebPubSubServiceClient.from_connection_string(
            connection_string=connection_string,
            hub=hub_name
        )

        # Generate a token that is valid for 1 hour
        token = service_client.get_client_access_token(expires_after=3600)

        # Return the URL in a JSON object for easy client-side parsing
        return func.HttpResponse(
            body=json.dumps({"url": token["url"]}),
            status_code=200,
            mimetype="application/json"
        )
    except KeyError as e:
        return func.HttpResponse(f"Server configuration error: Missing setting for {e}.", status_code=500)
    except Exception as e:
        return func.HttpResponse(f"An internal server error occurred: {e}", status_code=500)


# --- Data Models ---
class TrainingRequest:
    def __init__(self, data: dict): self.topic=data.get('topic'); self.subTopic=data.get('subTopic'); self.industry=data.get('industry'); self.audience=data.get('audience'); self.desiredLength=data.get('desiredLength'); self.deepContext=None
    def validate(self):
        if not self.topic: return "Please provide a training topic."
        return None
    def to_dict(self): return {k: v for k, v in self.__dict__.items()}

class ChatRequest:
    def __init__(self, data: dict): self.context = data.get('context'); self.question = data.get('question')
    def validate(self):
        if not self.context: return "Request must include a 'context' string."
        if not self.question: return "Request must include a 'question' string."
        return None

# --- Services ---
class GenerativeAiService:
    def __init__(self, api_key: str, azure_endpoint: str, api_version: str, azure_deployment: str, logger: logging.Logger): self.azure_deployment=azure_deployment; self.client=openai.AsyncAzureOpenAI(api_key=api_key, api_version=api_version, azure_endpoint=azure_endpoint); self._logger=logger
    
    async def generate_module_async(self, request: TrainingRequest) -> dict:
        self._logger.info(f"Generating module for topic: '{request.topic}'")
        system_prompt = """
        You are the XCanvas Generative Core. Your sole purpose is to take a user's request and generate a single, complete, and valid JSON object. Your final response MUST be ONLY the JSON object and nothing else.
        To generate the final JSON object, you MUST follow these steps in this exact order:
        1. ANALYZE THE REQUEST: First, carefully analyze the user request to understand the core topic, industry, and target audience.
        2. GENERATE 'training_script': Assume the persona of an expert UK-based corporate trainer. The output for this part MUST be a JSON object with these keys in this order: 'introduction', 'key_points' (an array with a maximum of 3 items), and 'conclusion'.
        3. GENERATE 'puzzle_data': Assume the persona of an expert instructional designer. Based exclusively on the concepts from 'key_points', design a drag-and-drop puzzle. The output for this part MUST be a JSON object with these keys in this order: 'type' (must be "drag_and_drop_categorisation"), 'instructions', and 'items' (an array of objects with 'text' and 'category' keys).
        4. GENERATE 'exercise_scenario': Assume the persona of an expert role-play writer. Based strictly on the 'training_script', create a role-play exercise. The 'opening_line' MUST require the user to apply knowledge from the 'key_points'. The output for this part MUST be a JSON object with these keys in this strict order: 'title', 'character_persona', and 'opening_line'.
        5. ASSEMBLE AND VALIDATE: Combine the three generated JSON objects ('training_script', 'puzzle_data', 'exercise_scenario') into a single parent JSON object.
        """
        user_prompt_parts = [f"<UserRequest>"]
        user_prompt_parts.append(f"- Topic: \"{request.topic}\"")
        if request.subTopic: user_prompt_parts.append(f"- Sub-Topic: \"{request.subTopic}\"")
        if request.industry: user_prompt_parts.append(f"- Industry: \"{request.industry}\"")
        if request.audience: user_prompt_parts.append(f"- Audience: \"{request.audience}\"")
        if request.deepContext: user_prompt_parts.extend(["- Deep Context:", f"\"{request.deepContext}\""])
        user_prompt_parts.append("</UserRequest>")
        user_prompt = "\n".join(user_prompt_parts)
        
        response = await self.client.chat.completions.create(model=self.azure_deployment, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], response_format={"type": "json_object"})
        return json.loads(response.choices[0].message.content)

    async def contextual_chat_async(self, request: ChatRequest) -> dict:
        self._logger.info("Handling contextual chat request...")
        system_prompt = "You are an expert assistant... Return a JSON object with a single key, 'answer'."
        user_prompt = json.dumps({"context": request.context, "question": request.question})
        response = await self.client.chat.completions.create(model=self.azure_deployment, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], response_format={"type": "json_object"})
        ai_response = json.loads(response.choices[0].message.content)
        return {"context": request.context, "question": request.question, "answer": ai_response.get("answer", "No answer found.")}
