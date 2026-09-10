import os
import json
import mimetypes
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=api_key)


def extract_text_from_image(image_path):
    """
    Extract medical report information from an image
    using Gemini Vision.
    """

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")

    # Detect image type automatically
    mime_type, _ = mimetypes.guess_type(image_path)

    if mime_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise ValueError("Unsupported image format. Use JPG, PNG, or WEBP.")

    # Read image
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    prompt = """
You are a medical report OCR assistant.

Read the uploaded medical report carefully.

Extract ONLY information that is clearly visible.

DO NOT guess missing values.

Return ONLY valid JSON in this format:

{
    "patient_name": "",
    "age": null,
    "gender": "",
    "medical_values": {},
    "other_information": ""
}

Rules:
1. Extract only clearly readable information.
2. Never guess or estimate values.
3. Keep numerical values exactly as shown.
4. Put medical measurements inside "medical_values".
5. If a value is missing or unreadable, use null or "".
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=[
            prompt,
            {
                "inline_data": {
                    "mime_type": mime_type,
                    "data": image_bytes
                }
            }
        ]
    )

    result = response.text

    result = result.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(result)

    except json.JSONDecodeError:
        return {
            "raw_text": result
        }


if __name__ == "__main__":
    print("Gemini OCR reader loaded successfully.")