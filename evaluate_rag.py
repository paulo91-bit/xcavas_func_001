# evaluate_rag_negative_tests.py

import os
import json
import requests
import asyncio
import openai
from dotenv import load_dotenv
from typing import Tuple, List
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.aio import SearchClient
from azure.search.documents.models import VectorizedQuery

# --- Configuration ---
# Load environment variables from a .env file for local development
load_dotenv()

# Your deployed Azure Function App URL
FUNCTION_APP_URL = os.getenv("FUNCTION_APP_URL") 

# Azure OpenAI settings (for embeddings and the "judge" model)
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_JUDGE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

# Azure AI Search settings (for the retrieval step)
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")

def validate_env_vars():
    """Checks if all required environment variables are set."""
    required_vars = [
        "FUNCTION_APP_URL", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_KEY",
        "AZURE_OPENAI_JUDGE_DEPLOYMENT", "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
        "AZURE_SEARCH_ENDPOINT", "AZURE_SEARCH_KEY", "AZURE_SEARCH_INDEX_NAME"
    ]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"Missing required environment variables in your .env file: {', '.join(missing_vars)}")

# --- Negative Test Golden Dataset ---
# The "correct_answer" is what we expect the AI to say when it doesn't know.
NEGATIVE_TEST_DATASET = [
    {"question": "What was Enbros's exact revenue in the last fiscal quarter?", "correct_answer": "The answer to this question cannot be found in the provided context."},
    {"question": "What is the current Enbros stock price or market capitalization?", "correct_answer": "The answer to this question cannot be found in the provided context."},
    {"question": "Which individual sales representative sold the most client contracts in the first half of the year?", "correct_answer": "The answer to this question cannot be found in the provided context."},
    {"question": "What is the specific IP address of the primary production server for the Enbros platform?", "correct_answer": "The answer to this question cannot be found in the provided context."},
    {"question": "What are the detailed strategic objectives of our top three competitors for the next two years?", "correct_answer": "The answer to this question cannot be found in the provided context."},
    {"question": "According to the document, what were the main conclusions of the most recent Gartner Magic Quadrant report?", "correct_answer": "The answer to this question cannot be found in the provided context."},
    {"question": "What new products does Enbros plan to launch in 2028?", "correct_answer": "The answer to this question cannot be found in the provided context."},
    {"question": "Which markets does Enbros plan to expand into after 2026?", "correct_answer": "The answer to this question cannot be found in the provided context."},
    {"question": "Who is the head of the Human Resources department at Enbros?", "correct_answer": "The answer to this question cannot be found in the provided context."},
    {"question": "What is the approved company holiday schedule for the upcoming calendar year?", "correct_answer": "The answer to this question cannot be found in the provided context."}
]

# --- Helper Functions ---

async def retrieve_context(question: str, ai_client: openai.AsyncAzureOpenAI, search_client: SearchClient) -> str:
    """Performs the RAG retrieval step."""
    print(f"\n--- Retrieving context for: '{question}' ---")
    query_embedding_response = await ai_client.embeddings.create(input=question, model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT)
    query_vector = query_embedding_response.data[0].embedding
    retrieved_context = ""
    vector_query = VectorizedQuery(vector=query_vector, k_nearest_neighbors=3, fields="contentVector")
    results = await search_client.search(search_text=None, vector_queries=[vector_query])
    async for result in results:
        retrieved_context += result['content'] + "\n\n"
    print(f"Retrieved Context: {retrieved_context[:150]}...")
    return retrieved_context

async def get_qna_response(context: str, question: str) -> str:
    """Calls the live /api/Chat endpoint to get a direct answer."""
    endpoint = f"{FUNCTION_APP_URL}/api/Chat"
    payload = {"context": context, "question": question}
    print(f"--- Getting Q&A response from: '{endpoint}' ---")
    try:
        response = requests.post(endpoint, json=payload)
        response.raise_for_status()
        response_json = response.json()
        generated_answer = response_json.get("answer", "No answer found.")
        print(f"Generated Answer: {generated_answer[:100]}...")
        return generated_answer
    except requests.exceptions.RequestException as e:
        print(f"Error calling the Azure Function: {e}")
        return None

async def evaluate_with_ai_judge(question, correct_answer, generated_answer, retrieved_context, ai_client: openai.AsyncAzureOpenAI) -> dict:
    """Uses a powerful AI model to score the RAG system's response."""
    print("--- Evaluating response with AI Judge ---")
    judge_prompt = f"""
    You are an impartial AI evaluator. Your task is to score an AI-generated answer based on its adherence to instructions.
    The system was instructed to respond with a specific phrase if the answer was not in the context.

    **Evaluation Criteria:**
    - **Adherence Score (1-5):** Did the 'Generated Answer' correctly state that the answer could not be found, as specified in the 'Correct Answer'? (5=Perfectly Adherent, 1=Failed, Hallucinated an answer)

    **Evaluation Data:**
    - **Question:** "{question}"
    - **Retrieved Context:** "{retrieved_context}"
    - **Correct Answer (Expected System Response):** "{correct_answer}"
    - **Generated Answer:** "{generated_answer}"

    Please provide your evaluation in a valid JSON object with the keys "adherence_score" and "reason".
    """
    try:
        response = await ai_client.chat.completions.create(
            model=AZURE_OPENAI_JUDGE_DEPLOYMENT,
            messages=[{"role": "user", "content": judge_prompt}],
            response_format={"type": "json_object"}
        )
        evaluation = json.loads(response.choices[0].message.content)
        print(f"Evaluation: {evaluation}")
        return evaluation
    except Exception as e:
        print(f"Error calling the AI Judge: {e}")
        return {"adherence_score": 0, "reason": str(e)}

# --- Main Execution ---

async def main():
    """Main function to run the evaluation pipeline."""
    validate_env_vars()
    results = []
    total_adherence = 0

    ai_client = openai.AsyncAzureOpenAI(api_key=AZURE_OPENAI_KEY, api_version=AZURE_API_VERSION, azure_endpoint=AZURE_OPENAI_ENDPOINT)
    search_client = SearchClient(endpoint=AZURE_SEARCH_ENDPOINT, index_name=AZURE_SEARCH_INDEX_NAME, credential=AzureKeyCredential(AZURE_SEARCH_KEY))

    async with search_client:
        for item in NEGATIVE_TEST_DATASET:
            question = item["question"]
            correct_answer = item["correct_answer"]
            
            retrieved_context = await retrieve_context(question, ai_client, search_client)
            generated_answer = await get_qna_response(retrieved_context, question)
            
            if generated_answer:
                evaluation = await evaluate_with_ai_judge(question, correct_answer, generated_answer, retrieved_context, ai_client)
                results.append({"question": question, "evaluation": evaluation})
                total_adherence += evaluation.get("adherence_score", 0)

    # --- Print Final Report ---
    print("\n\n--- RAG NEGATIVE TEST REPORT ---")
    if results:
        avg_adherence = total_adherence / len(results)
        print(f"\nAverage Adherence Score: {avg_adherence:.2f} / 5.0")
        print("\nDetailed Results:")
        for res in results:
            print(f"- Q: {res['question']}")
            print(f"  Score: {res['evaluation']}")
    else:
        print("No results to report.")

if __name__ == "__main__":
    asyncio.run(main())
