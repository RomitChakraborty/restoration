#!/Users/romitchakraborty/venvs/realesrgan311/bin/python

import os, base64
from pathlib import Path
from io import BytesIO
from PIL import Image
from google import genai
from google.genai import types
import argparse

MODEL = "gemini-2.5-flash-image-preview"

PROMPT = """
    Restore and 4× upscale a vintage broadcast cricket frame (1980s). Keep the original 4:3 composition,
    Make the Final Still photorealistic,
    Authentic color cast (slightly warm TV), and natural fine grain. Remove analog noise, ringing/haloing, 
    And interlace combing. Sharpen modestly and recover real detail in faces, kit texture, grass, and ball seams,
    Colourise wherever you detect black and white frames,
    Make the fields lush green wherever possible,    
    Without inventing new elements. Preserve all logos and on-screen captions exactly—OCR-safe: keep glyph shapes,
    Don’t retype or replace fonts, no paraphrasing. Upscale all fonts. No watermarks. No background changes, no stylization,
    Documentary realism.
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
    parser.add_argument("-pa", "--prompt-append", required=False,help="Extra instruction added to the default prompt", default="")
    args = parser.parse_args()

    in_path = Path(args.input).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve()

    # Build final prompt
    final_prompt = PROMPT
    if args.prompt_append:
        final_prompt = f"{final_prompt.strip()} {args.prompt_append.strip()}"

    print("Final prompt:\n", final_prompt, "\n")
    print("Negative prompt:\n", NEG_PROMPT, "\n")
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

    # Read and normalize to PNG bytes for stable upload
    pil = Image.open(in_path).convert("RGB")
    buf = BytesIO(); pil.save(buf, format="PNG")
    image_bytes = buf.getvalue()

    # Your SDK build prefers strings + Blob for images
    contents = [
        final_prompt,
        f"Negative prompt: {NEG_PROMPT}",
        pil,
    ]
    
    response = client.models.generate_content(
        model=MODEL, 
        contents=contents,
        )
    ok = save_first_image(response, out_path)
    print("Saved image to:", out_path if ok else "No image returned.")

if __name__ == "__main__":
    main()
