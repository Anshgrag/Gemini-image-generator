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

# ========== CONFIGURATION ==========
API_KEY = "YOUR_API"
MODEL_NAME = "gemini-2.5-flash-image"
OUTPUT_DIR = "generated_rings"
os.makedirs(OUTPUT_DIR, exist_ok=True)

client = genai.Client(api_key=API_KEY)


# ========== CORE FUNCTIONS ==========

def load_image_from_url(url):
    """Download image from URL"""
    try:
        url = url.strip()
        if not url or not url.startswith('http'):
            raise Exception("Invalid URL")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'image/*',
        }
        
        response = requests.get(url, headers=headers, timeout=30, stream=True)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content)).convert("RGB")
        
    except Exception as e:
        raise Exception(f"URL load failed: {str(e)}")


def load_images_from_files(files):
    """Load images from uploaded files"""
    images = []
    names = []
    
    if not files:
        return images, names
    
    for file in files:
        try:
            if isinstance(file, str):
                file_path = file
            elif hasattr(file, 'name'):
                file_path = file.name
            else:
                continue
            
            ext = os.path.splitext(file_path)[1].lower()
            if ext not in ['.jpg', '.jpeg', '.png', '.webp', '.bmp']:
                continue
            
            img = Image.open(file_path).convert("RGB")
            images.append(img)
            names.append(os.path.splitext(os.path.basename(file_path))[0])
            
        except Exception as e:
            print(f"Failed to load {file}: {e}")
            continue
    
    return images, names


def parse_urls(url_text):
    """Parse multiple URLs from text"""
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
    raise ValueError("No image generated")


def validate_generated_image(img, ring_model, stone_ref):
    """Basic quality check for generated image"""
    try:
        # Check resolution
        if img.size[0] < 800 or img.size[1] < 800:
            return False, "Resolution too low"
        
        # Check if image is not blank/corrupted
        import numpy as np
        img_array = np.array(img)
        if img_array.std() < 10:  # Too uniform, likely blank
            return False, "Image appears blank"
        
        return True, "OK"
    except:
        return False, "Validation error"


def generate_ring_with_stone_optimized(ring_model, stone_reference, custom_prompt="", max_retries=2):
    """
    Generate ring with stone - OPTIMIZED FOR QUALITY
    - Fresh chat per generation (no context pollution)
    - Higher temperature for better quality
    - Retry logic if quality check fails
    """
    
    for attempt in range(max_retries):
        try:
            # Create FRESH chat for each generation (prevents context pollution)
            chat = client.chats.create(
                model=MODEL_NAME,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE", "TEXT"],
                    temperature=0.6,  # Slightly higher for better quality
                    top_p=0.95,
                    top_k=40
                )
            )
            
            # Build optimized prompt
            if custom_prompt.strip():
                final_prompt = f"""I will show you two separate images:

IMAGE 1: Ring model (base structure to preserve)
IMAGE 2: Stone reference (appearance to transfer)

{custom_prompt}

CRITICAL REQUIREMENTS:
- Base your output on IMAGE 1's ring structure EXACTLY
- Transfer ONLY the stone appearance from IMAGE 2
- Pure white background (RGB 255,255,255)
- Output: 1024x1024 pixels, maximum detail
- Professional jewelry photography quality"""
            else:
                final_prompt = """ENTER_YOUR_PROMPT"""

            # Send message with both images
            response = chat.send_message([
                final_prompt,
                ring_model,      # IMAGE 1
                stone_reference  # IMAGE 2
            ])
            
            img = extract_image_from_response(response)
            
            # Validate quality
            is_valid, reason = validate_generated_image(img, ring_model, stone_reference)
            
            if is_valid or attempt == max_retries - 1:
                return img, is_valid, reason
            else:
                print(f"Quality check failed (attempt {attempt + 1}): {reason}, retrying...")
                time.sleep(1)  # Brief pause before retry
                
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"Generation error (attempt {attempt + 1}): {e}, retrying...")
            time.sleep(1)
    
    raise Exception("Failed after max retries")


# ========== GUI INTERFACE ==========

def generate_rings_batch(
    model_source, model_folder, model_urls_text,
    stone_source, stone_folder, stone_urls_text,
    custom_prompt, num_variations,
    enable_validation,
    progress=gr.Progress()
):
    """Main generation function with quality optimization"""
    
    progress(0, desc="Loading images...")
    
    # Load ring models
    ring_models = []
    ring_model_names = []
    
    if model_source == "Upload Folder":
        if not model_folder:
            return "❌ Please upload ring model images", [], None, None
        ring_models, ring_model_names = load_images_from_files(model_folder)
        if not ring_models:
            return "❌ No valid images found", [], None, None
        print(f"✓ Loaded {len(ring_models)} ring models")
        
    elif model_source == "Image URL(s)":
        model_urls = parse_urls(model_urls_text)
        if not model_urls:
            return "❌ Please enter ring model URLs", [], None, None
        
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
            return "❌ Please upload stone images", [], ring_models[0], None
        stone_references, stone_names = load_images_from_files(stone_folder)
        if not stone_references:
            return "❌ No valid images found", [], ring_models[0], None
        print(f"✓ Loaded {len(stone_references)} stones")
        
    elif stone_source == "Image URL(s)":
        stone_urls = parse_urls(stone_urls_text)
        if not stone_urls:
            return "❌ Please enter stone URLs", [], ring_models[0], None
        
        for idx, url in enumerate(stone_urls):
            try:
                img = load_image_from_url(url)
                stone_references.append(img)
                stone_names.append(f"stone_{idx+1}")
                print(f"✓ Loaded stone {idx+1}/{len(stone_urls)}")
            except Exception as e:
                print(f"✗ Failed stone {idx+1}: {e}")
    
    if not stone_references:
        return "❌ No stones loaded", [], ring_models[0], None
    
    # Calculate total
    total_combinations = len(ring_models) * len(stone_references) * num_variations
    print(f"\nGenerating {total_combinations} images ({len(ring_models)} × {len(stone_references)} × {num_variations})")
    
    progress(0.1, desc=f"Starting generation of {total_combinations} images...")
    
    generated_paths = []
    results_log = []
    completed = 0
    failed = 0
    quality_issues = 0
    start_time = time.time()
    
    # Process each combination with fresh context
    for model_idx, (ring_model, model_name) in enumerate(zip(ring_models, ring_model_names)):
        for stone_idx, (stone_ref, stone_name) in enumerate(zip(stone_references, stone_names)):
            for var_num in range(1, num_variations + 1):
                try:
                    # Generate with quality optimization
                    max_retries = 2 if enable_validation else 1
                    img, is_valid, reason = generate_ring_with_stone_optimized(
                        ring_model, stone_ref, custom_prompt, max_retries
                    )
                    
                    # Save
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:17]
                    filename = f"{model_name}_{stone_name}_v{var_num}_{timestamp}.png"
                    output_path = os.path.join(OUTPUT_DIR, filename)
                    img.save(output_path, quality=95, optimize=True)
                    
                    generated_paths.append(output_path)
                    completed += 1
                    
                    status = "✅" if is_valid else "⚠️"
                    results_log.append(f"{status} {filename}")
                    if not is_valid:
                        quality_issues += 1
                    
                    # Update progress
                    progress_val = 0.1 + (0.9 * completed / total_combinations)
                    progress(progress_val, desc=f"Generated {completed}/{total_combinations}")
                    
                    # Small delay to avoid rate limits
                    time.sleep(0.5)
                    
                except Exception as e:
                    failed += 1
                    results_log.append(f"❌ {model_name}+{stone_name}_v{var_num}: {str(e)[:50]}")
                    print(f"Error: {e}")
    
    elapsed = time.time() - start_time
    
    # Create detailed summary
    summary = f"""
╔══════════════════════════════════════════════════╗
║       QUALITY-OPTIMIZED GENERATION COMPLETE      ║
╚══════════════════════════════════════════════════╝

✅ Successfully generated: {completed}/{total_combinations}
❌ Failed: {failed}
⚠️  Quality warnings: {quality_issues}
⏱️  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)
⚡ Average: {elapsed/completed if completed else 0:.1f} sec/image
📐 Resolution: 1024×1024 pixels
💾 Saved to: {OUTPUT_DIR}/

📊 Batch Summary:
   • Ring models: {len(ring_models)}
   • Stone references: {len(stone_references)}
   • Variations/combo: {num_variations}
   • Fresh context per image: YES ✓
   • Quality validation: {'ON' if enable_validation else 'OFF'}

💰 Cost Estimate:
   • Per image: ~$0.134
   • Total batch: ~${0.134 * completed:.2f}

📋 Recent Results:
"""
    summary += "\n".join(results_log[-15:])
    
    if len(results_log) > 15:
        summary += f"\n... and {len(results_log) - 15} more (see console)"
    
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
    
    with gr.Blocks(title="Ring Stone Modifier - Quality Optimized") as app:
        
        gr.Markdown("""
        # 💍 AI Ring Stone Modifier - Quality Optimized
        
        **Improved quality for batch processing - Each image gets fresh context**
        
        ✨ Fresh AI context per image (prevents quality degradation)  
        🎯 Optional quality validation with auto-retry  
        🖼️ Professional 1024×1024 output
        """)
        
        with gr.Row():
            # LEFT COLUMN
            with gr.Column(scale=1):
                
                gr.Markdown("### 🔷 Ring Models (IMAGE 1)")
                
                model_source = gr.Radio(
                    choices=["Upload Folder", "Image URL(s)"],
                    value="Upload Folder",
                    label="Source"
                )
                
                model_folder = gr.File(
                    label="Upload Ring Images (Multiple)",
                    file_count="multiple",
                    file_types=["image"],
                    visible=True
                )
                
                model_urls_text = gr.Textbox(
                    label="Or Paste URLs (one per line)",
                    placeholder="https://example.com/ring1.jpg\nhttps://example.com/ring2.jpg",
                    visible=False,
                    lines=4
                )
                
                model_source.change(
                    fn=toggle_model_input,
                    inputs=[model_source],
                    outputs=[model_folder, model_urls_text]
                )
                
                gr.Markdown("---")
                
                gr.Markdown("### 💎 Stone References (IMAGE 2)")
                
                stone_source = gr.Radio(
                    choices=["Upload Folder", "Image URL(s)"],
                    value="Upload Folder",
                    label="Source"
                )
                
                stone_folder = gr.File(
                    label="Upload Stone Images (Multiple)",
                    file_count="multiple",
                    file_types=["image"],
                    visible=True
                )
                
                stone_urls_text = gr.Textbox(
                    label="Or Paste URLs (one per line)",
                    placeholder="https://example.com/stone1.jpg\nhttps://example.com/stone2.jpg",
                    visible=False,
                    lines=4
                )
                
                stone_source.change(
                    fn=toggle_stone_input,
                    inputs=[stone_source],
                    outputs=[stone_folder, stone_urls_text]
                )
                
                gr.Markdown("---")
                gr.Markdown("### ⚙️ Generation Settings")
                
                custom_prompt = gr.Textbox(
                    label="Custom Prompt (Optional)",
                    placeholder="Leave empty for optimized default prompt",
                    lines=2,
                    value=""
                )
                
                num_variations = gr.Slider(
                    minimum=1,
                    maximum=3,
                    value=1,
                    step=1,
                    label="Variations per Combination"
                )
                
                enable_validation = gr.Checkbox(
                    label="Enable Quality Validation (auto-retry low quality)",
                    value=True
                )
                
                generate_btn = gr.Button("🎨 Generate All (Quality Optimized)", variant="primary", size="lg")
            
            # RIGHT COLUMN
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Generation Log")
                
                output_log = gr.Textbox(
                    label="Real-time Progress",
                    lines=12,
                    max_lines=20
                )
                
                gr.Markdown("### 🖼️ Previews & Results")
                
                with gr.Row():
                    model_preview = gr.Image(label="Sample Ring Model", type="pil")
                    stone_preview = gr.Image(label="Sample Stone", type="pil")
                
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
                custom_prompt, num_variations, enable_validation
            ],
            outputs=[output_log, output_gallery, model_preview, stone_preview]
        )
        
        gr.Markdown("""
        ---
        ### 🎯 Quality Improvements in This Version:
        
        **1. Fresh Context Per Image** ✅
        - Each combination gets a NEW chat session
        - Prevents "context pollution" from previous generations
        - Maintains consistent quality across 100s of images
        
        **2. Optimized Prompt** ✅
        - More detailed step-by-step instructions to AI
        - Explicit separation of IMAGE 1 vs IMAGE 2
        - Clear quality requirements
        
        **3. Quality Validation** ✅
        - Checks resolution and image quality
        - Auto-retries if quality is low
        - Flags warnings in results
        
        **4. Better Generation Parameters** ✅
        - Temperature: 0.6 (balanced quality/consistency)
        - Top-p sampling for better coherence
        - Rate limiting to avoid API throttling
        
        ### 💡 Tips for Best Quality:
        
        ✅ Use high-resolution input images (1024px+)  
        ✅ Keep custom prompts simple or use default  
        ✅ Enable quality validation for important batches  
        ✅ Process in smaller batches (50-100 combos) for monitoring  
        ✅ Use 1-2 variations per combo (more = longer processing)
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
