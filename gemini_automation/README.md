# 💍 AI Ring Stone Modifier

An AI-powered tool that uses Google's Gemini model to generate custom ring designs by transferring stone appearances to ring models.

## How It Works

This tool takes **two input images** and combines them using AI:

1. **IMAGE 1 (Ring Model)** - The base ring design that will be preserved
2. **IMAGE 2 (Stone Reference)** - The stone whose color/appearance should be transferred

**Result:** A generated ring image with the stone's appearance applied to the ring's center stone

## Example

See the `generated_rings` folder for example outputs:

| Input: Ring Model | → | Output: Generated Ring |
|-------------------|---|------------------------|
| ![Ring Model](generated_rings/Gemini_Generated_Image_riw6cwriw6cwriw6.png) | ➔ | ![Generated](generated_rings/Gemini_Generated_Image_4ksx304ksx304ksx.png) |

> **Note:** The arrow shows the generation process - the ring model on the left is used as a base, and the AI generates a new ring with the stone's appearance (as shown in the generated image on the right).

## Features

- **Folder Upload Support** - Upload multiple ring models and stone references at once
- **Batch Processing** - Automatically generates all combinations
- **URL Support** - Load images from URLs (S3, Google Drive direct links, etc.)
- **High Resolution** - 1024×1024 pixel output
- **Custom Prompts** - Optional custom instructions for the AI

## Installation

```bash
pip install google-genai pillow requests gradio
```

## Configuration

Edit `NewFile_1.py` and replace the API key:

```python
API_KEY = "YOUR_GEMINI_API_KEY"
```

Get your API key from: https://aistudio.google.com/app/apikey

## Usage

1. Run the app:
   ```bash
   python NewFile_1.py
   ```

2. Open browser at `http://localhost:7860`

3. **Choose input method:**
   - **Upload Folder** - Drag & drop multiple images
   - **Image URL(s)** - Paste URLs (one per line)

4. Click **"Generate All Combinations"**

## Example Input/Output Flow

```
Ring Model Images          Stone Reference Images       Generated Results
┌──────────────┐           ┌──────────────┐            ┌──────────────┐
│              │           │              │            │              │
│   [ring1]    │    +      │   [stone]    │     =      │  [new ring]  │
│              │           │              │            │              │
└──────────────┘           └──────────────┘            └──────────────┘
    IMAGE 1                    IMAGE 2                     OUTPUT
```

## Requirements

- Python 3.8+
- Google Gemini API key
- Internet connection

## License

MIT
