# function_app.py
# This is the main file for the v2 programming model.
# UPDATED: The Chat endpoint now uses a context-question-answer format.

import logging
import json
import asyncio
import os
import openai
import azure.functions as func
import fitz # PyMuPDF
from docx import Document
import io

# --- Logger setup ---
logger = logging.getLogger(__name__)

# --- Azure Function App instance ---
app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

# Helper function for robust text decoding
def _safely_decode_text(file_bytes: bytes, filename: str, content_type: str) -> str:
    """
    Safely decodes file bytes into a string, handling various file types and encodings.
    Raises ValueError if decoding fails or file type is unsupported.
    """
    text = ""
    
    if filename.endswith('.pdf') or content_type == "application/pdf":
        logger.info("PDF file detected. Extracting text...")
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            text = "\n".join([page.get_text() for page in doc])
    
    elif filename.endswith('.docx') or content_type in [
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/octet-stream"  # fallback for Word uploads
    ]:
        logger.info("DOCX file detected. Extracting text...")
        docx_stream = io.BytesIO(file_bytes)
        doc = Document(docx_stream)
        text = "\n".join([para.text for para in doc.paragraphs])

    elif filename.endswith('.txt') and content_type.startswith("text/"):
        logger.info("Plain text file detected. Attempting UTF-8 decode...")
        try:
            text = file_bytes.decode('utf-8')
            logger.info("Text file successfully decoded with UTF-8.")
        except UnicodeDecodeError:
            logger.warning("UTF-8 decode failed. Attempting decode with latin-1 as fallback.")
            try:
                text = file_bytes.decode('latin-1')
                logger.info("Text file successfully decoded with latin-1.")
            except Exception as decode_e:
                logger.exception(f"Text file could not be decoded with UTF-8 or latin-1: {decode_e}")
                raise ValueError("Text file could not be decoded. Please ensure it's a valid text encoding (UTF-8 recommended).") from decode_e
    else:
        logger.warning("Unsupported file type.")
        raise ValueError("Unsupported file type. Please upload a .pdf, .docx, or UTF-8 .txt file.")
        
    return text

# --- HTTP Trigger 1: Module Generation ---
@app.route(route="GenerateTrainingModule", methods=['post'])
async def GenerateTrainingModule(req: func.HttpRequest) -> func.HttpResponse:
    """
    Main function entry point for generating a complete training module.
    """
    logger.info('Python GenerateTrainingModule trigger function processed a request.')
    try:
        # --- Get Configuration & Parse Request ---
        azure_openai_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
        azure_openai_key = os.environ["AZURE_OPENAI_KEY"]
        azure_openai_deployment = os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]
        azure_api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01")
        form_data = req.form
        req_body = {key: form_data.get(key) for key in form_data}
        
        deep_context_content = ""

        # --- Handle deepContextFile upload safely ---
        if 'deepContextFile' in req.files:
            file = req.files['deepContextFile']
            filename = file.filename.lower()
            file_bytes = file.read()
            content_type = file.content_type

            logger.info(f"Received file: {filename} ({content_type})")

            try:
                deep_context_content = _safely_decode_text(file_bytes, filename, content_type)
            except ValueError as ve:
                return func.HttpResponse(str(ve), status_code=400)
            except Exception as e:
                logger.exception("File processing failed.")
                return func.HttpResponse(f"Could not process uploaded file: {e}", status_code=400)
        else:
            # If no file is uploaded, use the deepContext from the form field directly
            deep_context_content = form_data.get('deepContext', '')
            logger.info("No file uploaded. Using 'deepContext' form field.")
            
            # Additional sanitization for form field deepContext to ensure valid UTF-8
            if deep_context_content:
                try:
                    # This step attempts to re-encode and decode the string.
                    # If the string already contains "mojibake" (characters that are not valid Unicode),
                    # encoding to latin-1 will treat each character as a byte.
                    # Then decoding to utf-8 with 'replace' will substitute invalid sequences.
                    deep_context_content = deep_context_content.encode('latin-1', errors='replace').decode('utf-8', errors='replace')
                    logger.info("deepContext from form field successfully sanitized to UTF-8.")
                except Exception as sanitize_e:
                    logger.exception(f"Failed to sanitize deepContext from form field: {sanitize_e}")
                    return func.HttpResponse("Provided deepContext text in form field could not be processed due to encoding issues.", status_code=400)

        req_body['deepContext'] = deep_context_content
        logger.debug(f"Extracted deepContext (preview): {deep_context_content[:300]}")
        
        # --- Validate & Generate ---
        training_request = TrainingRequest(req_body)
        validation_error = training_request.validate()
        if validation_error: return func.HttpResponse(validation_error, status_code=400)
        
        # Pass the correct logger instance
        ai_service = GenerativeAiService(api_key=azure_openai_key, azure_endpoint=azure_openai_endpoint, api_version=azure_api_version, azure_deployment=azure_openai_deployment, logger=logger)
        training_module = await ai_service.generate_module_async(training_request)
        
        # --- Final Response ---
        training_module['requestDetails'] = training_request.to_dict()
        return func.HttpResponse(json.dumps(training_module), mimetype="application/json", status_code=200)

    except KeyError as e:
        logger.exception("Missing environment variable.") # Added logging for KeyError
        return func.HttpResponse(f"Server configuration error: Missing setting for {e}.", status_code=500)
    except Exception as e:
        logger.exception("An internal server error occurred.") # Added logging for generic Exception
        return func.HttpResponse(f"An internal server error occurred: {e}", status_code=500)

# --- HTTP Trigger 2: Contextual Q&A Chat (Text-based) ---
@app.route(route="Chat", methods=['post'])
async def ChatHandler(req: func.HttpRequest) -> func.HttpResponse:
    """
    Handles a context-question-answer request.
    """
    logger.info('Python Chat trigger function processed a request.') # Changed logging to use 'logger'
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
        # Pass the correct logger instance
        ai_service = GenerativeAiService(api_key=azure_openai_key, azure_endpoint=azure_openai_endpoint, api_version=azure_api_version, azure_deployment=azure_openai_deployment, logger=logger)
        response_json = await ai_service.contextual_chat_async(chat_request)
        
        # --- Final Response ---
        return func.HttpResponse(json.dumps(response_json), mimetype="application/json", status_code=200)

    except KeyError as e:
        logger.exception("Missing environment variable.") # Added logging for KeyError
        return func.HttpResponse(f"Server configuration error: Missing setting for {e}.", status_code=500)
    except Exception as e:
        logger.exception("An internal server error occurred.") # Added logging for generic Exception
        return func.HttpResponse(f"An internal server error occurred: {e}", status_code=500)

# --- HTTP Trigger 3: Structured Role-Play (Text-based) ---
@app.route(route="StructuredRolePlay", methods=['post'])
async def StructuredRolePlayHandler(req: func.HttpRequest) -> func.HttpResponse:
    """
    Handles a structured, text-based role-play simulation turn.
    Accepts and returns a complex JSON object with messages and objectives.
    """
    logger.info('Python StructuredRolePlay trigger function processed a request.') # Changed logging to use 'logger'
    try:
        # --- Get Configuration & Parse Request ---
        azure_openai_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
        azure_openai_key = os.environ["AZURE_OPENAI_KEY"]
        azure_openai_deployment = os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]
        azure_api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01")
        req_body = req.get_json()
        roleplay_request = StructuredRolePlayRequest(req_body)
        validation_error = roleplay_request.validate()
        if validation_error: return func.HttpResponse(validation_error, status_code=400)
        
        # --- Get AI Response ---
        # Pass the correct logger instance
        ai_service = GenerativeAiService(api_key=azure_openai_key, azure_endpoint=azure_openai_endpoint, api_version=azure_api_version, azure_deployment=azure_openai_deployment, logger=logger)
        response_json = await ai_service.structured_roleplay_async(roleplay_request)
        
        # --- Final Response ---
        return func.HttpResponse(json.dumps(response_json), mimetype="application/json", status_code=200)

    except KeyError as e:
        logger.exception("Missing environment variable.") # Added logging for KeyError
        return func.HttpResponse(f"Server configuration error: Missing setting for {e}.", status_code=500)
    except Exception as e:
        logger.exception("An internal server error occurred.") # Added logging for generic Exception
        return func.HttpResponse(f"An internal server error occurred: {e}", status_code=500)


# --- Data Models ---
class TrainingRequest:
    def __init__(self, data: dict):
        self.topic = data.get('topic')
        self.subTopic = data.get('subTopic')
        self.industry = data.get('industry')
        self.audience = data.get('audience')
        self.desiredLength = data.get('desiredLength')
        self.deepContext = data.get('deepContext')
    def validate(self):
        if not self.topic: return "Please provide a training topic."
        if not self.industry: return "Please provide the industry."
        if not self.audience: return "Please provide the target audience."
        return None
    def to_dict(self): return {k: v for k, v in self.__dict__.items()}

class ChatRequest:
    """Represents a request for the contextual Q&A chat."""
    def __init__(self, data: dict):
        self.context = data.get('context')
        self.question = data.get('question')
    def validate(self):
        if not self.context: return "Request must include a 'context' string."
        if not self.question: return "Request must include a 'question' string."
        return None

class StructuredRolePlayRequest:
    """Represents the state of a structured role-play simulation."""
    def __init__(self, data: dict):
        self.messages = data.get('messages', [])
        self.objectives = data.get('objectives', [])
        self.trainingContext = data.get('trainingContext')
    def validate(self):
        if not self.messages: return "Request must include a 'messages' array."
        if not self.objectives: return "Request must include an 'objectives' array."
        return None

# --- Services ---
class GenerativeAiService:
    def __init__(self, api_key: str, azure_endpoint: str, api_version: str, azure_deployment: str, logger: logging.Logger):
        self.azure_deployment = azure_deployment
        self.client = openai.AsyncAzureOpenAI(api_key=api_key, api_version=api_version, azure_endpoint=azure_endpoint)
        self._logger = logger
    
    async def generate_module_async(self, request: TrainingRequest) -> dict:
        self._logger.info(f"Generating module for topic: '{request.topic}'")
        system_prompt = """
        You are an expert instructional designer for XCanvas. Your task is to generate a complete, interactive training module based on detailed specifications.
        Your response must be a single, valid JSON object that adheres to the following JSON schema:
        {
          "title": "...",
          "training": {"introduction": "...", "keyConcepts": ["..."], "exampleScenario": "..."},
          "puzzles": [{"description": "...", "question": "...", "options": ["..."], "correctOptionIndex": 0}],
          "exercises": [{"description": "...", "scenario": "...", "tasks": ["..."]}]
        }
        Do not include any other text or explanation outside of the JSON object.
        """
        user_prompt_parts = [f"Generate a training module with these specs:", f"- Topic: \"{request.topic}\""]
        if request.deepContext: user_prompt_parts.extend(["\n--- DEEP CONTEXT ---", f"Context: \"{request.deepContext}\""])
        user_prompt = "\n".join(user_prompt_parts)
        response = await self.client.chat.completions.create(model=self.azure_deployment, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], response_format={"type": "json_object"})
        return json.loads(response.choices[0].message.content)

    async def contextual_chat_async(self, request: ChatRequest) -> dict:
        """Handles a contextual question and answer request."""
        self._logger.info("Handling contextual chat request...")
        system_prompt = """
        You are an expert assistant trained to provide clear, accurate, and concise answers based on the provided context and question.
        You will receive a JSON object with a 'context' and a 'question'.
        Your task is to analyze the context and provide the best possible answer to the question.
        Return a JSON object with a single key, "answer", containing your response string.
        Example: {"answer": "This is the answer."}
        Do not include any other text or explanation.
        """
        
        user_prompt = json.dumps({
            "context": request.context,
            "question": request.question
        })

        response = await self.client.chat.completions.create(
            model=self.azure_deployment,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        ai_response = json.loads(response.choices[0].message.content)
        
        # Construct the final response object
        final_response = {
            "context": request.context,
            "question": request.question,
            "answer": ai_response.get("answer", "No answer found.")
        }
        return final_response

    async def structured_roleplay_async(self, request: StructuredRolePlayRequest) -> dict:
        """Handles a structured role-play turn."""
        self._logger.info("Handling structured role-play turn.")
        system_prompt = """
        You are an AI simulation engine inside a corporate training platform. Your job is to simulate the next turn in a realistic training conversation based on prior context and update the objective status accordingly.
        You will receive a JSON object with a list of previous messages, a list of objectives, and optional training context.
        You must return a complete JSON object in the same structure. In your response, you must:
        1. Add only one new message (an AI-generated message object) to the 'messages' array that continues the scenario.
        2. Modify the 'status' of any objective in the 'objectives' array based on the user's latest response.
        The new message's role must be "ai" or "system".
        Only output a valid JSON object. Do not add any explanation or commentary — JSON only.
        """
        
        user_prompt = json.dumps(request.__dict__)

        response = await self.client.chat.completions.create(
            model=self.azure_deployment,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        updated_simulation_state = json.loads(response.choices[0].message.content)

        if 'messages' in updated_simulation_state and updated_simulation_state['messages']:
            updated_simulation_state['newMessage'] = updated_simulation_state['messages'][-1]

        return updated_simulation_state
