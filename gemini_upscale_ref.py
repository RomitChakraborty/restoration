#!/usr/bin/env python3
import os, base64
from pathlib import Path
from io import BytesIO
from PIL import Image
from google import genai
from google.genai import types
import argparse

MODEL = "gemini-2.5-flash-image-preview"

PROMPT= """
    Restore and 4× upscale a vintage broadcast cricket frame (1980s). 
    Keep composition, broadcast color, and natural grain. Remove analog noise and interlace artifacts. 
    Preserve on-screen captions exactly (OCR-safe). 
    Use the following reference images only for maintaining the batter’s facial identity. Do not change pose, clothing, bat, background, or text. Blend identity cues across all references.
    match facial structure and texture but DO NOT change pose, clothing, equipment, background, or text. 
    No new elements; documentary realism.
"""


NEG_PROMPT = (
    "no hallucinated objects, no added players/crowd"
    "no heavy denoise that erases grain, no text redraw or font substitution"
)

def save_first_image(response, out_path: Path) -> bool:
    cand = response.candidates[0]
    for part in cand.content.parts:
        if getattr(part, "text", None):
            print("[model text]", part.text)
        elif getattr(part, "inline_data", None):
            data = part.inline_data.data
            # handle bytes or base64 transparently
            if isinstance(data, (bytes, bytearray)):
                payload = data
            else:
                try:
                    payload = base64.b64decode(data)
                except Exception:
                    payload = bytes(data)
            img = Image.open(BytesIO(payload))
            out_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(out_path)
            return True
    return False

def main():
    parser = argparse.ArgumentParser(description="Upscale a vintage cricket still using Gemini image model.")
    
    parser.add_argument("-i", "--input", required=True, help="Absolute path to input image file")
    parser.add_argument("-o", "--output", required=True, help="Absolute path to output image file")
    parser.add_argument("-p", "--prompt", required=False, help="Prompt used for text + image to image conversion", default=PROMPT)
    args = parser.parse_args()

    in_path = Path(args.input).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve()

    print("Default prompt:\n", PROMPT, "\n")
    print("Negative prompt:\n", NEG_PROMPT, "\n")

    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

    # Read and normalize to PNG bytes for stable upload
    pil = Image.open(in_path).convert("RGB")
    buf = BytesIO(); pil.save(buf, format="PNG")
    image_bytes = buf.getvalue()

    # Reference image for identity reference
    ref_paths = [
    "/Users/romitchakraborty/QAI/Imran Khan Batting/Face/Pataudi_Interview/face_1_upscaled.webp",
    "/Users/romitchakraborty/QAI/Imran Khan Batting/Face/Pataudi_Interview/face_2_upscaled.webp",
    ]
    ref_images = [Image.open(p).convert("RGB") for p in ref_paths]

    # Your SDK build prefers strings + Blob for images
    contents = [
        PROMPT,
        f"Negative prompt: {NEG_PROMPT}",
        pil,
    ] + ref_images
     
    

    response = client.models.generate_content(
        model=MODEL, 
        contents=contents,
        )
    ok = save_first_image(response, out_path)
    print("Saved image to:", out_path if ok else "No image returned.")

if __name__ == "__main__":
    main()
