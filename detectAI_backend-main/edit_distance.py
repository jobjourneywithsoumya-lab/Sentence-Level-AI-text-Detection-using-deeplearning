import os
import Levenshtein
from groq import Groq
from fastapi import HTTPException


def generate_rewritten_text(input_text):
    """
    Generate a rewritten version of the input text using the Groq API.
    Handles the interaction with the Groq API for generating a rewritten version of the input text.
    """
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        return input_text

    client = Groq(api_key=api_key)
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": f"Rewrite the following text (Don't Write anything extra, the answer Should Contain only the rewritten text):\n\n{input_text}",
                }
            ],
            model="llama3-8b-8192",
        )
        # Extract the rewritten content from the API response
        rewritten_text = chat_completion.choices[0].message.content
        return rewritten_text

    except Exception as e:
        # Handle errors from the Groq API or network issues
        raise HTTPException(status_code=500, detail=f"Error in rewriting text: {str(e)}")


def calculate_edit_distance(original_text, rewritten_text):
    """
    Calculate the Levenshtein distance (edit distance) between the original and rewritten texts.
    """
    return Levenshtein.distance(original_text, rewritten_text)


def Edit_distance(input_text):
    """
    Complete function to generate a rewritten text and calculate the edit distance
    between the original and rewritten versions.
    """
    # Generate rewritten text
    rewritten_text = generate_rewritten_text(input_text)

    # Calculate edit distance between original and rewritten texts
    distance = calculate_edit_distance(input_text, rewritten_text)

    return distance

# Example usage:
# text = "The quick brown fox jumps over the lazy dog"
# print(edit_distance(text))
