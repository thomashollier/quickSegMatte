import torch
from PIL import Image
import os
import argparse
from tqdm import tqdm
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import torchvision
from torchvision.transforms import functional as F
from torch.nn.functional import interpolate

# --- STAGE 1: People Removal with DeepLabV3 ---
def remove_people_deeplab(image, model, device):
    """
    Takes a PIL image and returns a new RGBA image with people removed.
    """
    img_tensor = F.to_tensor(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor)['out']
    
    predictions = output.argmax(1).squeeze(0).cpu().numpy()
    # The 'person' class in COCO is 15. Mask is opaque (255) where there is NO person.
    mask_np = (predictions != 15).astype('uint8') * 255
    
    mask = Image.fromarray(mask_np, mode='L').resize(image.size, Image.NEAREST)
    
    image_with_alpha = image.copy()
    image_with_alpha.putalpha(mask)
    return image_with_alpha


# --- STAGE 2: Custom Object Removal with CLIP-Seg ---
def generate_clip_mask(processor, model, device, image, prompts, thresholds):
    """
    Helper function to generate a combined boolean mask from CLIP-Seg prompts.
    Returns a PyTorch boolean tensor where True means "select this pixel".
    """
    if not prompts:
        return None

    individual_masks = []
    for prompt, threshold in zip(prompts, thresholds):
        inputs = processor(
            text=[prompt], images=[image], padding="max_length",
            truncation=True, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
        
        logit = outputs.logits.squeeze()
        resized_logit = interpolate(
            logit.unsqueeze(0).unsqueeze(0),
            size=image.size[::-1], mode="bilinear", align_corners=False
        ).squeeze()
        normalized_heatmap = (resized_logit - resized_logit.min()) / (resized_logit.max() - resized_logit.min())
        individual_masks.append(normalized_heatmap > threshold)

    stacked_masks = torch.stack(individual_masks, dim=0)
    combined_mask, _ = torch.max(stacked_masks, dim=0)
    return combined_mask


def run_two_stage_pipeline(deeplab_model, clip_processor, clip_model, device,
                           input_path, output_path,
                           prompts, thresholds,
                           negative_prompts, negative_thresholds):
    try:
        # Load the original RGB image for processing
        original_image = Image.open(input_path).convert("RGB")

        # --- STAGE 1 ---
        # Get an RGBA image with people already removed
        image_stage1 = remove_people_deeplab(original_image, deeplab_model, device)
        
        # --- STAGE 2 (Optional) ---
        if not prompts:
            # If no further prompts, just save the result from stage 1
            image_stage1.save(output_path, "PNG")
            return

        # Generate the positive removal mask from CLIP-Seg
        clip_removal_mask = generate_clip_mask(clip_processor, clip_model, device,
                                               original_image, prompts, thresholds)
        
        # Generate the negative protection mask from CLIP-Seg
        clip_protection_mask = generate_clip_mask(clip_processor, clip_model, device,
                                                  original_image, negative_prompts, negative_thresholds)
        
        # Determine the final removal mask for Stage 2
        if clip_protection_mask is not None:
            final_clip_mask = clip_removal_mask & ~clip_protection_mask
        else:
            final_clip_mask = clip_removal_mask
            
        # --- COMBINE MASKS ---
        # Get the alpha mask from Stage 1 (people removed)
        alpha_stage1 = torch.from_numpy(
            np.array(image_stage1.getchannel('A'))
        ).to(device).float() / 255.0 # Normalize to 0-1

        # The alpha mask for Stage 2 is the inverse of its removal mask
        alpha_stage2 = (~final_clip_mask).float()

        # The final alpha is the minimum of both stages. A pixel is transparent
        # if EITHER stage made it transparent.
        final_alpha = torch.minimum(alpha_stage1, alpha_stage2)
        
        # Convert final combined alpha back to a Pillow mask
        final_alpha_np = (final_alpha * 255).cpu().numpy().astype('uint8')
        final_mask_img = Image.fromarray(final_alpha_np, mode='L')
        
        # Apply the final combined mask to the original image
        final_image = original_image.copy()
        final_image.putalpha(final_mask_img)
        final_image.save(output_path, "PNG")

    except Exception as e:
        print(f"\nError processing {os.path.basename(input_path)}: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Two-stage pipeline: auto-remove people, then remove custom objects.")
    # Stage 2 (CLIP-Seg) arguments are now optional
    parser.add_argument("--prompts", type=str, nargs='*', default=[], help="[Stage 2] Objects to remove (e.g., 'a garden hose').")
    parser.add_argument("--thresholds", type=float, nargs='*', default=[], help="[Stage 2] Threshold for each positive prompt.")
    parser.add_argument("--negative-prompts", type=str, nargs='*', default=[], help="[Stage 2] Objects to protect from removal.")
    parser.add_argument("--negative-thresholds", type=float, nargs='*', default=[], help="[Stage 2] Threshold for each negative prompt.")
    
    parser.add_argument("--input-folder", type=str, required=True, help="Path to input images.")
    parser.add_argument("--output-folder", type=str, required=True, help="Path for output images.")
    args = parser.parse_args()

    # --- Input Validation ---
    if args.prompts and len(args.prompts) != len(args.thresholds):
        print("Error: The number of positive prompts must match the number of positive thresholds.")
        return
    if args.negative_prompts and len(args.negative_prompts) != len(args.negative_thresholds):
        print("Error: The number of negative prompts must match the number of negative thresholds.")
        return

    # --- Setup and Logging ---
    print("--- Two-Stage Removal Pipeline ---")
    print("Stage 1 (Automatic): Removing 'person' using DeepLabV3.")
    if args.prompts:
        print("Stage 2 (Custom):")
        print("  Positive Prompts (to remove):")
        for p, t in zip(args.prompts, args.thresholds): print(f"    - '{p}' (Threshold: {t})")
        if args.negative_prompts:
            print("  Negative Prompts (to protect):")
            for p, t in zip(args.negative_prompts, args.negative_thresholds): print(f"    - '{p}' (Threshold: {t})")
    else:
        print("Stage 2 (Custom): Skipped (no prompts provided).")
    print(f"\nResults will be saved in: '{args.output_folder}'")

    # --- Model Loading ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    print("Loading Stage 1 model (DeepLabV3)...")
    deeplab_model = torchvision.models.segmentation.deeplabv3_resnet101(weights=torchvision.models.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT)
    deeplab_model.to(device).eval()

    # Only load the heavy CLIP-Seg model if it's needed
    clip_processor, clip_model = None, None
    if args.prompts:
        print("Loading Stage 2 model (CLIP-Seg)...")
        clip_model_name = "CIDAS/clipseg-rd64-refined"
        clip_processor = CLIPSegProcessor.from_pretrained(clip_model_name)
        clip_model = CLIPSegForImageSegmentation.from_pretrained(clip_model_name).to(device)
    
    print("Models loaded successfully.\n")

    # --- Image Processing Loop ---
    # Need numpy for mask combination
    global np
    import numpy as np
    
    image_files = [f for f in os.listdir(args.input_folder) if f.lower().endswith(('.jpg', '.jpeg'))]
    if not image_files:
        print(f"No JPG images found in '{args.input_folder}'.")
        return

    for filename in tqdm(image_files, desc="Processing Images"):
        input_path = os.path.join(args.input_folder, filename)
        output_filename = os.path.splitext(filename)[0] + ".png"
        output_path = os.path.join(args.output_folder, output_filename)
        
        run_two_stage_pipeline(
            deeplab_model, clip_processor, clip_model, device,
            input_path, output_path,
            args.prompts, args.thresholds,
            args.negative_prompts, args.negative_thresholds
        )

    print("\nProcessing complete!")

if __name__ == "__main__":
    main()