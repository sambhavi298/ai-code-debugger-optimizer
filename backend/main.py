import os
import logging
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from starlette.requests import Request
from cachetools import TTLCache

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Validate API Key
if not api_key:
    raise ValueError("🚨 OpenAI API Key is missing! Please check your .env file.")

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# FastAPI instance
app = FastAPI()

# Setup CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this in production to restrict domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup Logging
logging.basicConfig(filename="backend_errors.log", level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")

# Setup Rate Limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(HTTPException, _rate_limit_exceeded_handler)

# Setup Caching (store responses for 5 minutes)
cache = TTLCache(maxsize=100, ttl=300)

# Root route (for testing)
@app.get("/")
def home():
    return {"message": "Hello, FastAPI backend is running successfully!"}

# Define request model for debugging
class CodeRequest(BaseModel):
    code: str
    language: str  # Supports multiple languages

@app.post("/debug-code")
@limiter.limit("5/minute")  # Limit to 5 requests per minute per user
async def debug_code(request: CodeRequest, req: Request):
    # Check if result is in cache
    cache_key = f"{request.language}-{request.code}"
    if cache_key in cache:
        return cache[cache_key]

    try:
        # Send request to OpenAI API
        response = client.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an AI that debugs and optimizes code with explanations."},
                {"role": "user", "content": f"Language: {request.language}\n\nDebug and optimize this code:\n{request.code}"}
            ],
            max_tokens=500
        )
        result = {
            "optimized_code": response.choices[0].message["content"].strip(),
            "explanation": "This response includes fixes and optimizations."
        }
        cache[cache_key] = result  # Store in cache
        return result
    except Exception as e:
        logging.error(f"Error debugging code: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while processing the request.")
