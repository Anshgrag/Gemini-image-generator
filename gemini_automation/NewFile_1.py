"""
Ring Design Modifier - Folder Upload Support (Gradio 6.0)
=========================================================
Features:
- Upload FOLDERS of images OR paste URLs
- Batch process all combinations automatically
- AI transfers stone appearance to ring models
- 1024x1024 high resolution output

Requirements:
    pip install google-genai pillow requests gradio
"""

import os
import io
import time
import requests
from PIL import Image
from google import genai
from google.genai import types
import gradio as gr
from datetime import datetime
import urllib.parse
import glob

# ========== CONFIGURATION ==========
API_KEY = "API_KEY"
MODEL_NAME = "gemini-2.5-flash-image"
OUTPUT_DIR = "generated_rings"
os.makedirs(OUTPUT_DIR, exist_ok=True)

client = genai.Client(api_key=API_KEY)


# ========== CORE FUNCTIONS ==========

def load_image_from_url(url):
    """Download image from URL with proper handling"""
    try:
        url = url.strip()
       
        if not url:
            raise Exception("Empty URL")
       
        if not url.startswith('http://') and not url.startswith('https://'):
            raise Exception("URL must start with http:// or https://")
       
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'image/avif,image/webp,image/apng,image/*,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': url.rsplit('/', 1)[0] + '/',
        }
       
        response = requests.get(url, headers=headers, timeout=30, stream=True, allow_redirects=True)
        response.raise_for_status()
       
        img = Image.open(io.BytesIO(response.content)).convert("RGB")
        return img
       
    except requests.exceptions.HTTPError as e:
        raise Exception(f"HTTP {e.response.status_code}: {str(e)}")
    except Exception as e:
        raise Exception(f"Failed: {str(e)}")


def load_images_from_files(files):
    """Load images from uploaded files"""
    images = []
    names = []
   
    if not files:
        return images, names
   
    for file in files:
        try:
            # Get file path
            if isinstance(file, str):
                file_path = file
            elif hasattr(file, 'name'):
                file_path = file.name
            else:
                continue
           
            # Check if it's an image
            ext = os.path.splitext(file_path)[1].lower()
            if ext not in ['.jpg', '.jpeg', '.png', '.webp', '.bmp']:
                continue
           
            # Load image
            img = Image.open(file_path).convert("RGB")
            images.append(img)
            names.append(os.path.basename(file_path).replace(ext, ''))
           
        except Exception as e:
            print(f"Failed to load {file}: {e}")
            continue
   
    return images, names


def parse_urls(url_text):
    """Parse multiple URLs from text (one per line)"""
    if not url_text:
        return []
   
    urls = []
    for line in url_text.strip().split('\n'):
        line = line.strip()
        if line and line.startswith('http'):
            urls.append(line)
   
    return urls


def extract_image_from_response(response):
    """Extract image from Gemini response"""
    for part in response.parts:
        if part.inline_data is not None:
            return part.as_image()
    raise ValueError("No image generated in response")


def generate_ring_with_stone(ring_model, stone_reference, custom_prompt=""):
    """Generate ring with stone - FIXED prompt and image order"""
   
    # Create chat
    chat = client.chats.create(
        model=MODEL_NAME,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
            temperature=0.4
        )
    )
   
    # FIXED: Better structured prompt with clear instructions
    if custom_prompt.strip():
        # User provided custom prompt
        final_prompt = f"""I will show you two images in this order:
IMAGE 1 (RING MODEL): The base ring design that must be preserved
IMAGE 2 (STONE REFERENCE): The stone whose appearance should be transferred

{custom_prompt}

CRITICAL OUTPUT REQUIREMENTS:
- Result must be based on IMAGE 1 (ring model) structure
- Only the center stone should reflect IMAGE 2 (stone reference) appearance
- Pure white background
- 1024x1024 pixels
- Photorealistic studio lighting"""
    else:
        # Default prompt
        final_prompt = """”"""

    # CRITICAL: Send images in correct order
    response = chat.send_message([
        final_prompt,
        ring_model,      # IMAGE 1
        stone_reference  # IMAGE 2
    ])
   
    img = extract_image_from_response(response)
    return img


# ========== GUI INTERFACE ==========

def generate_rings_batch(
    model_source, model_folder, model_urls_text,
    stone_source, stone_folder, stone_urls_text,
    custom_prompt, num_variations,
    progress=gr.Progress()
):
    """Main generation function with folder support"""
   
    progress(0, desc="Processing inputs...")
   
    # Load ring models
    ring_models = []
    ring_model_names = []
   
    if model_source == "Upload Folder":
        if not model_folder:
            return "❌ Please upload ring model images", [], None, None
       
        ring_models, ring_model_names = load_images_from_files(model_folder)
        if not ring_models:
            return "❌ No valid images found in uploaded files", [], None, None
        print(f"✓ Loaded {len(ring_models)} ring models from folder")
       
    elif model_source == "Image URL(s)":
        model_urls = parse_urls(model_urls_text)
        if not model_urls:
            return "❌ Please enter at least one ring model URL", [], None, None
       
        print(f"Found {len(model_urls)} ring model URL(s)")
        for idx, url in enumerate(model_urls):
            try:
                img = load_image_from_url(url)
                ring_models.append(img)
                ring_model_names.append(f"model_{idx+1}")
                print(f"✓ Loaded model {idx+1}/{len(model_urls)}")
            except Exception as e:
                print(f"✗ Failed model {idx+1}: {e}")
   
    if not ring_models:
        return "❌ No ring models loaded", [], None, None
   
    # Load stone references
    stone_references = []
    stone_names = []
   
    if stone_source == "Upload Folder":
        if not stone_folder:
            return "❌ Please upload stone reference images", [], ring_models[0], None
       
        stone_references, stone_names = load_images_from_files(stone_folder)
        if not stone_references:
            return "❌ No valid images found in uploaded files", [], ring_models[0], None
        print(f"✓ Loaded {len(stone_references)} stone references from folder")
       
    elif stone_source == "Image URL(s)":
        stone_urls = parse_urls(stone_urls_text)
        if not stone_urls:
            return "❌ Please enter at least one stone reference URL", [], ring_models[0], None
       
        print(f"Found {len(stone_urls)} stone reference URL(s)")
        for idx, url in enumerate(stone_urls):
            try:
                img = load_image_from_url(url)
                stone_references.append(img)
                stone_names.append(f"stone_{idx+1}")
                print(f"✓ Loaded stone {idx+1}/{len(stone_urls)}")
            except Exception as e:
                print(f"✗ Failed stone {idx+1}: {e}")
   
    if not stone_references:
        return "❌ No stone references loaded", [], ring_models[0], None
   
    # Generate combinations
    total_combinations = len(ring_models) * len(stone_references) * num_variations
    print(f"\nGenerating {total_combinations} images ({len(ring_models)} models × {len(stone_references)} stones × {num_variations} variations)")
   
    progress(0.1, desc=f"Generating {total_combinations} images...")
   
    generated_paths = []
    results_log = []
    completed = 0
    start_time = time.time()
   
    # Process all combinations
    for model_idx, (ring_model, model_name) in enumerate(zip(ring_models, ring_model_names)):
        for stone_idx, (stone_ref, stone_name) in enumerate(zip(stone_references, stone_names)):
            for var_num in range(1, num_variations + 1):
                try:
                    # Generate
                    img = generate_ring_with_stone(ring_model, stone_ref, custom_prompt)
                   
                    # Save
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{model_name}_{stone_name}_v{var_num}_{timestamp}.png"
                    output_path = os.path.join(OUTPUT_DIR, filename)
                    img.save(output_path)
                   
                    generated_paths.append(output_path)
                    completed += 1
                    results_log.append(f"✅ {filename}")
                   
                    # Update progress
                    progress(0.1 + (0.9 * completed / total_combinations),
                            desc=f"Generated {completed}/{total_combinations}")
                   
                except Exception as e:
                    results_log.append(f"❌ Failed: {model_name}+{stone_name} v{var_num} - {str(e)}")
                    print(f"Error: {e}")
   
    elapsed = time.time() - start_time
   
    # Create summary
    summary = f"""
╔══════════════════════════════════════════════════╗
║          BATCH GENERATION COMPLETE               ║
╚══════════════════════════════════════════════════╝

✅ Successfully generated: {len(generated_paths)}/{total_combinations}
⏱️  Total time: {elapsed:.1f} seconds
⚡ Average: {elapsed/len(generated_paths) if generated_paths else 0:.1f} sec/image
📐 Resolution: 1024×1024 pixels
💾 Output folder: {OUTPUT_DIR}/

📊 Input Summary:
   - Ring models: {len(ring_models)}
   - Stone references: {len(stone_references)}
   - Variations per combo: {num_variations}

📊 Cost Estimate:
   - Per image: ~$0.134
   - Total: ~${0.134 * len(generated_paths):.2f}

Results:
"""
    summary += "\n".join(results_log[-20:])
   
    if len(results_log) > 20:
        summary += f"\n... and {len(results_log) - 20} more"
   
    progress(1.0, desc="Complete!")
   
    return summary, generated_paths, ring_models[0] if ring_models else None, stone_references[0] if stone_references else None


def toggle_model_input(source):
    return (
        gr.update(visible=(source == "Upload Folder")),
        gr.update(visible=(source == "Image URL(s)"))
    )


def toggle_stone_input(source):
    return (
        gr.update(visible=(source == "Upload Folder")),
        gr.update(visible=(source == "Image URL(s)"))
    )


# ========== GRADIO UI ==========

def create_gui():
   
    with gr.Blocks(title="Ring Stone Modifier - Folder Upload") as app:
       
        gr.Markdown("""
        # 💍 AI Ring Stone Modifier - Folder Upload Support
       
        **Upload entire folders of images OR paste URLs**
       
        ✨ Drag & drop folders of ring models and stone references  
        🎨 Batch process all combinations automatically  
        🖼️ High-quality 1024×1024 product images
        """)
       
        with gr.Row():
            # LEFT COLUMN
            with gr.Column(scale=1):
               
                gr.Markdown("### 🔷 Ring Models - IMAGE 1")
                gr.Markdown("*Base designs that will be preserved*")
               
                model_source = gr.Radio(
                    choices=["Upload Folder", "Image URL(s)"],
                    value="Upload Folder",
                    label="Ring Model Source"
                )
               
                model_folder = gr.File(
                    label="Upload Ring Model Images (Multiple Files)",
                    file_count="multiple",
                    file_types=["image"],
                    visible=True
                )
               
                model_urls_text = gr.Textbox(
                    label="Or paste Ring Model URLs (one per line)",
                    placeholder="https://example.com/ring1.jpg\nhttps://example.com/ring2.jpg",
                    visible=False,
                    lines=5
                )
               
                model_source.change(
                    fn=toggle_model_input,
                    inputs=[model_source],
                    outputs=[model_folder, model_urls_text]
                )
               
                gr.Markdown("---")
               
                gr.Markdown("### 💎 Stone References - IMAGE 2")
                gr.Markdown("*Color/patterns to extract and transfer*")
               
                stone_source = gr.Radio(
                    choices=["Upload Folder", "Image URL(s)"],
                    value="Upload Folder",
                    label="Stone Reference Source"
                )
               
                stone_folder = gr.File(
                    label="Upload Stone Reference Images (Multiple Files)",
                    file_count="multiple",
                    file_types=["image"],
                    visible=True
                )
               
                stone_urls_text = gr.Textbox(
                    label="Or paste Stone Reference URLs (one per line)",
                    placeholder="https://example.com/stone1.jpg\nhttps://example.com/stone2.jpg",
                    visible=False,
                    lines=5
                )
               
                stone_source.change(
                    fn=toggle_stone_input,
                    inputs=[stone_source],
                    outputs=[stone_folder, stone_urls_text]
                )
               
                gr.Markdown("---")
               
                custom_prompt = gr.Textbox(
                    label="Custom Prompt (Optional)",
                    placeholder="Leave empty for default behavior",
                    lines=3,
                    value=""
                )
               
                num_variations = gr.Slider(
                    minimum=1,
                    maximum=5,
                    value=1,
                    step=1,
                    label="Variations per Combination"
                )
               
                generate_btn = gr.Button("🎨 Generate All Combinations", variant="primary", size="lg")
           
            # RIGHT COLUMN
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Generation Results")
               
                output_log = gr.Textbox(
                    label="Processing Log",
                    lines=15,
                    max_lines=20
                )
               
                gr.Markdown("### 🖼️ Input Previews & Generated Results")
               
                with gr.Row():
                    model_preview = gr.Image(label="First Ring Model", type="pil")
                    stone_preview = gr.Image(label="First Stone Reference", type="pil")
               
                output_gallery = gr.Gallery(
                    label="Generated Rings (1024×1024)",
                    columns=3,
                    height=500
                )
       
        generate_btn.click(
            fn=generate_rings_batch,
            inputs=[
                model_source, model_folder, model_urls_text,
                stone_source, stone_folder, stone_urls_text,
                custom_prompt, num_variations
            ],
            outputs=[output_log, output_gallery, model_preview, stone_preview]
        )
       
        gr.Markdown("""
        ---
        ### 💡 How to Use Folder Upload:
       
        **Method 1: Drag & Drop Folder**
        1. Click "Upload Ring Model Images" area
        2. Select multiple files from your folder (Ctrl+A or Cmd+A)
        3. Or drag and drop files directly
        4. Repeat for Stone References
        5. Click Generate
       
        **Method 2: Use URLs**
        1. Switch to "Image URL(s)" mode
        2. Paste one URL per line
        3. Works with S3, Google Drive (direct links), or any public URL
       
        ### 📂 Folder Structure Example:
       
        ```
        ring_models/
        ├── ring_001.jpg
        ├── ring_002.jpg
        └── ring_003.jpg
       
        stone_references/
        ├── malachite.jpg
        ├── ruby.jpg
        └── sapphire.jpg
        ```
       
        **Result:** 3 ring models × 3 stones = 9 unique combinations
       
        ### 🎯 Tips:
        - Upload as many images as you want (100s supported)
        - Supported formats: JPG, PNG, WEBP, BMP
        - Images are automatically filtered and loaded
        - Failed images are skipped with error messages
        - Processing time: ~10-15 seconds per combination
        """)
   
    return app


if __name__ == "__main__":
    app = create_gui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
