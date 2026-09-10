import os
from dotenv import load_dotenv
from google import genai

# Load .env
load_dotenv()

# Get API key
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found")

# Create Gemini client
client = genai.Client(api_key=api_key)

# Test Gemini
response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents="Reply with exactly: Gemini OCR connection successful"
)

print(response.text)