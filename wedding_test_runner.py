"""
===============================================================================
WEDDING DECOR VISUALIZATION - COMPREHENSIVE TEST RUNNER
===============================================================================
Runs all 3 tests from the Breakdown PDF:
- Test 1: Blue pintuck + ghost chairs → green/gold chargers → blue/white plates → green centerpiece
- Test 2: Gold chiavari + gold chargers → red plates + pink napkins → pink/gold centerpiece  
- Test 3: Champagne pintuck + phoenix chairs → purple chargers → black cutlery → white plates → champagne napkins → purple centerpiece → wine glasses

Operations supported:
1. Element Edit - Change specific element using reference image
2. Table Removal - Remove table, keep products floating
3. Fusion - Merge floating elements onto table scene

All intermediary images are saved for testing purposes.
===============================================================================
"""

import os
import gc
import time
import math
import torch
import requests
from PIL import Image
from datetime import datetime
from io import BytesIO
from diffusers import QwenImageEditPlusPipeline, FlowMatchEulerDiscreteScheduler
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from enum import Enum

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = "/workspace/wedding_decor"
IMAGES_DIR = os.path.join(BASE_DIR, "images")
TESTS_DIR = os.path.join(IMAGES_DIR, "tests")
RESULTS_DIR = os.path.join(TESTS_DIR, "Results")

FIXED_WIDTH = 1024
FIXED_HEIGHT = 1024
REF_SIZE = 384

LORA_WEIGHTS = "Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-8steps-V1.0-fp32.safetensors"

TRUE_CFG_SCALE = 1.0
GUIDANCE_SCALE = 1.0
BASE_SEED = 42

# Default inference steps for different operation types
DEFAULT_STEPS = {
    "edit": 8,
    "table_removal": 8,
    "fusion": 8,
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class OperationType(Enum):
    EDIT = "edit"                    # Change element using reference
    TABLE_REMOVAL = "table_removal"  # Remove table, keep floating elements
    FUSION = "fusion"                # Merge two images together


@dataclass
class Operation:
    """Single operation in the test workflow"""
    op_type: OperationType
    name: str
    prompt: str
    input_image: str                 # Path or "previous" for chained ops
    ref_image: Optional[str] = None  # Path to reference image (for edit/fusion)
    ref_url: Optional[str] = None    # URL to download reference from
    output_name: str = None          # Name for output file
    steps: int = 8
    visible_to_user: bool = True     # Whether this is shown to user in final flow


@dataclass  
class TestConfig:
    """Configuration for a complete test"""
    name: str
    test_dir: str
    base_image: str
    operations: List[Operation]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def print_banner(text, char="="):
    line = char * 70
    print(f"\n{line}")
    print(f"  {text}")
    print(f"{line}\n")


def resize_to_fixed(img, width=FIXED_WIDTH, height=FIXED_HEIGHT):
    """Resize image to fixed dimensions"""
    return img.resize((width, height), Image.LANCZOS)


def resize_reference(img, size=REF_SIZE):
    """Resize reference image to square"""
    return img.resize((size, size), Image.LANCZOS)


def ensure_square(img, target_size=FIXED_WIDTH):
    """Ensure image is square by center-cropping or padding"""
    w, h = img.size
    if w == h:
        return img.resize((target_size, target_size), Image.LANCZOS)
    
    # Center crop to square
    min_dim = min(w, h)
    left = (w - min_dim) // 2
    top = (h - min_dim) // 2
    img = img.crop((left, top, left + min_dim, top + min_dim))
    return img.resize((target_size, target_size), Image.LANCZOS)


def download_image(url: str, save_path: str = None) -> Image.Image:
    """Download image from URL and optionally save locally"""
    print(f"   📥 Downloading: {url[:60]}...")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            img.save(save_path)
            print(f"   💾 Saved to: {save_path}")
        
        return img
    except Exception as e:
        print(f"   ❌ Failed to download: {e}")
        return None


def load_or_download_image(path: str = None, url: str = None, cache_dir: str = None) -> Image.Image:
    """Load image from path or download from URL"""
    if path and os.path.exists(path):
        return Image.open(path).convert("RGB")
    
    if url:
        # Create cache path from URL
        if cache_dir:
            filename = url.split("/")[-1].split("?")[0]
            cache_path = os.path.join(cache_dir, filename)
            if os.path.exists(cache_path):
                return Image.open(cache_path).convert("RGB")
            return download_image(url, cache_path)
        return download_image(url)
    
    raise ValueError(f"Cannot load image - path: {path}, url: {url}")


def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.2f}s"
    return f"{int(seconds // 60)}m {seconds % 60:.1f}s"


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_pipeline():
    print_banner("LOADING MODEL")
    
    gc.collect()
    torch.cuda.empty_cache()
    
    scheduler_config = {
        "base_image_seq_len": 256,
        "base_shift": math.log(3),
        "invert_sigmas": False,
        "max_image_seq_len": 8192,
        "max_shift": math.log(3),
        "num_train_timesteps": 1000,
        "shift": 1.0,
        "shift_terminal": None,
        "stochastic_sampling": False,
        "time_shift_type": "exponential",
        "use_beta_sigmas": False,
        "use_dynamic_shifting": True,
        "use_exponential_sigmas": False,
        "use_karras_sigmas": False,
    }
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
    
    print("📦 Loading Qwen-Image-Edit-2509...")
    load_start = time.time()
    
    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2509",
        scheduler=scheduler,
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    
    print(f"✅ Base model loaded in {time.time() - load_start:.1f}s")
    
    print("⚡ Loading Lightning 8-step LoRA...")
    pipeline.load_lora_weights(
        "lightx2v/Qwen-Image-Lightning",
        weight_name=LORA_WEIGHTS
    )
    print("✅ LoRA loaded")
    
    pipeline.set_progress_bar_config(disable=True)
    
    return pipeline


def warmup(pipeline):
    """Run warmup inference"""
    print("\n🔥 Warmup run...")
    dummy = Image.new('RGB', (FIXED_WIDTH, FIXED_HEIGHT), 'white')
    dummy_ref = Image.new('RGB', (REF_SIZE, REF_SIZE), 'gray')
    
    with torch.inference_mode():
        _ = pipeline(
            image=[dummy, dummy_ref],
            prompt="warmup",
            num_inference_steps=4,
            true_cfg_scale=1.0,
            guidance_scale=1.0,
        )
    
    gc.collect()
    torch.cuda.empty_cache()
    print("✅ Warmup complete")


# =============================================================================
# INFERENCE FUNCTIONS
# =============================================================================

def run_edit(pipeline, base_img: Image.Image, ref_img: Image.Image, 
             prompt: str, steps: int, seed: int) -> Image.Image:
    """Run element edit operation"""
    with torch.inference_mode():
        output = pipeline(
            image=[base_img, ref_img],
            prompt=prompt,
            negative_prompt="blurry, distorted, low quality, deformed, artifacts, missing items, wrong count",
            num_inference_steps=steps,
            true_cfg_scale=TRUE_CFG_SCALE,
            guidance_scale=GUIDANCE_SCALE,
            generator=torch.Generator("cuda").manual_seed(seed),
        )
    
    result = output.images[0]
    if result.size != (FIXED_WIDTH, FIXED_HEIGHT):
        result = resize_to_fixed(result)
    
    return result


def run_table_removal(pipeline, base_img: Image.Image, steps: int, seed: int) -> Image.Image:
    """Remove table from image, keep products floating"""
    prompt = "Remove the table from this image. Leave the products floating in the air, do not change anything about the products (size or orientation or anything). Keep the white/gray background."
    
    # For table removal, we don't need a reference image
    # We'll use a blank reference
    ref_img = Image.new('RGB', (REF_SIZE, REF_SIZE), 'white')
    
    with torch.inference_mode():
        output = pipeline(
            image=[base_img, ref_img],
            prompt=prompt,
            negative_prompt="table, tablecloth, fabric, draped cloth, products on surface",
            num_inference_steps=steps,
            true_cfg_scale=TRUE_CFG_SCALE,
            guidance_scale=GUIDANCE_SCALE,
            generator=torch.Generator("cuda").manual_seed(seed),
        )
    
    result = output.images[0]
    if result.size != (FIXED_WIDTH, FIXED_HEIGHT):
        result = resize_to_fixed(result)
    
    return result


def run_fusion(pipeline, base_img: Image.Image, overlay_img: Image.Image, 
               prompt: str, steps: int, seed: int) -> Image.Image:
    """Fuse two images together"""
    with torch.inference_mode():
        output = pipeline(
            image=[base_img, resize_reference(overlay_img)],
            prompt=prompt,
            negative_prompt="blurry, distorted, low quality, deformed, artifacts, misaligned, wrong position",
            num_inference_steps=steps,
            true_cfg_scale=TRUE_CFG_SCALE,
            guidance_scale=GUIDANCE_SCALE,
            generator=torch.Generator("cuda").manual_seed(seed),
        )
    
    result = output.images[0]
    if result.size != (FIXED_WIDTH, FIXED_HEIGHT):
        result = resize_to_fixed(result)
    
    return result


# =============================================================================
# TEST CONFIGURATIONS
# =============================================================================

def get_test1_config() -> TestConfig:
    """Test 1: Blue pintuck tablecloth with ghost chairs flow"""
    test_dir = os.path.join(TESTS_DIR, "Test 1")
    ref_cache = os.path.join(test_dir, "ref_cache")
    
    return TestConfig(
        name="Test 1",
        test_dir=test_dir,
        base_image=os.path.join(test_dir, "Set_1_RoundTable_WhiteTablecloth_GoldChiavariChairsWhiteCushion.png"),
        operations=[
            # === SET 1: Tablecloth & Chairs ===
            Operation(
                op_type=OperationType.EDIT,
                name="tablecloth_blue_pintuck",
                prompt="Change the tablecloth on the table to a blue pintuck tablecloth matching the reference. Keep the gold chiavari chairs.",
                input_image="base",
                ref_url="https://www.illusionsrentals.com/wp-content/uploads/2024/07/Royal-Blue-Pintuck-Tablecloth-2837-Edit.jpg",
                output_name="Set_1_BluePintuckTablecloth",
                visible_to_user=False,  # User must change chairs first
            ),
            Operation(
                op_type=OperationType.EDIT,
                name="chairs_ghost",
                prompt="Replace all chairs with clear ghost chiavari chairs with white cushions matching the reference. 8 chairs evenly spaced around the round table with blue pintuck tablecloth.",
                input_image="previous",
                ref_url="https://allcargos.com/wp-content/uploads/2018/12/Acrylic-Ghost-Clear-Wedding-Chiavari-Chairs-Rental-Toronto.png",
                output_name="Set_1_PERMUTATION_1_BluePintuck_GhostChairs",
                visible_to_user=True,
            ),
            
            # === SET 2: Charger Plates & Cutlery ===
            Operation(
                op_type=OperationType.EDIT,
                name="charger_plates_green_gold",
                prompt="Add green and gold charger plates at each place setting on the blue pintuck tablecloth, matching the reference. 8 place settings around the table. Ghost chairs remain.",
                input_image="previous",
                ref_url="https://m.media-amazon.com/images/I/71nBtmlp6L.jpg",
                output_name="Set_2_ChargerPlates_NoTable",
                visible_to_user=False,
            ),
            Operation(
                op_type=OperationType.TABLE_REMOVAL,
                name="table_removal_set2",
                prompt="Remove the table from this image. Leave the charger plates floating in position.",
                input_image="previous",
                output_name="Set_2.5_ChargerPlates_Floating",
                visible_to_user=False,
            ),
            Operation(
                op_type=OperationType.EDIT,
                name="cutlery_rose_gold",
                prompt="Add rose gold cutlery beside each charger plate - 2 forks on left, knife and spoon on right, matching the reference.",
                input_image="Set_2_ChargerPlates_NoTable",  # Go back to pre-removal
                ref_url="https://5.imimg.com/data5/SELLER/Default/2023/4/301820642/WL/JJ/DQ/113440186/stainless-steel-rose-gold-cutlery-set-500x500.png",
                output_name="Set_2_ChargerPlates_RoseGoldCutlery",
                visible_to_user=False,
            ),
            Operation(
                op_type=OperationType.TABLE_REMOVAL,
                name="table_removal_set2_cutlery",
                prompt="Remove the table from this image. Leave the charger plates and cutlery floating.",
                input_image="previous",
                output_name="Set_2.5_ChargerPlates_Cutlery_Floating",
                visible_to_user=False,
            ),
            
            # === FUSION A: Set 1 + Set 2.5 ===
            Operation(
                op_type=OperationType.FUSION,
                name="fusion_A",
                prompt="Place the floating charger plates and cutlery onto the table with blue pintuck tablecloth and ghost chairs. Align the place settings properly around the table.",
                input_image="Set_1_PERMUTATION_1_BluePintuck_GhostChairs",
                ref_image="Set_2.5_ChargerPlates_Cutlery_Floating",
                output_name="Image_A",
                visible_to_user=True,
            ),
            
            # === SET 3: Dinner Plates & Napkins ===
            Operation(
                op_type=OperationType.EDIT,
                name="dinner_plates_blue_white",
                prompt="Add blue and white floral dinner plates on top of each charger plate, matching the reference.",
                input_image="Image_A",
                ref_url="https://tabletales.ca/cdn/shop/files/vintage_floral_charger_plate_blue.png?v=1759417908",
                output_name="Set_3_DinnerPlates",
                visible_to_user=False,
            ),
            Operation(
                op_type=OperationType.EDIT,
                name="napkins_rose_gold",
                prompt="Add rose gold satin napkins folded elegantly on each dinner plate, matching the reference.",
                input_image="previous",
                ref_url="https://www.amazon.ca/Horbaunal-Napkins-Decoration-Weddings-Banquets/dp/B0BL3NFYDJ",
                output_name="Set_3_DinnerPlates_Napkins",
                visible_to_user=False,
            ),
            Operation(
                op_type=OperationType.TABLE_REMOVAL,
                name="table_removal_set3",
                prompt="Remove the table. Leave the dinner plates and napkins floating in position.",
                input_image="previous",
                output_name="Set_3.5_DinnerPlates_Napkins_Floating",
                visible_to_user=False,
            ),
            
            # === FUSION B: Image A + Set 3.5 ===
            Operation(
                op_type=OperationType.FUSION,
                name="fusion_B",
                prompt="Place the floating dinner plates and napkins on top of the charger plates on the table. Stack them properly: charger plate, then dinner plate, then napkin on top.",
                input_image="Image_A",
                ref_image="Set_3.5_DinnerPlates_Napkins_Floating",
                output_name="Image_B",
                visible_to_user=True,
            ),
            
            # === SET 4: Centerpiece & Glassware ===
            Operation(
                op_type=OperationType.EDIT,
                name="centerpiece_green",
                prompt="Add a green crystal vase centerpiece to the center of the table, matching the reference.",
                input_image="Image_B",
                ref_url="https://ajka-crystal.com/cdn/shop/files/5995747767123_1c59d2c8-5073-4bdc-9b13-0eeb16f6151a.png?v=1761055410",
                output_name="Set_4_Centerpiece",
                visible_to_user=False,
            ),
            Operation(
                op_type=OperationType.EDIT,
                name="glassware_champagne_flutes",
                prompt="Add champagne flutes at each place setting above the knife, matching the reference.",
                input_image="previous",
                output_name="Set_4_Centerpiece_Glassware",
                visible_to_user=False,
            ),
            Operation(
                op_type=OperationType.TABLE_REMOVAL,
                name="table_removal_set4",
                prompt="Remove the table. Leave the centerpiece and glasses floating.",
                input_image="previous",
                output_name="Set_4.5_Centerpiece_Glassware_Floating",
                visible_to_user=False,
            ),
            
            # === FUSION C: Image B + Set 4.5 ===
            Operation(
                op_type=OperationType.FUSION,
                name="fusion_C",
                prompt="Place the floating centerpiece in the center of the table and the champagne flutes at each place setting.",
                input_image="Image_B",
                ref_image="Set_4.5_Centerpiece_Glassware_Floating",
                output_name="Image_C_FINAL",
                visible_to_user=True,
            ),
        ]
    )


def get_test2_config() -> TestConfig:
    """Test 2: Gold chiavari chairs with gold/red theme"""
    test_dir = os.path.join(TESTS_DIR, "Test 2")
    
    return TestConfig(
        name="Test 2",
        test_dir=test_dir,
        base_image=os.path.join(test_dir, "Set_1_RoundTable_WhiteTablecloth_GoldChiavariChairsWhiteCushion.png"),
        operations=[
            # === SET 1 & 2.5 FUSION → Image A ===
            Operation(
                op_type=OperationType.EDIT,
                name="charger_plates_gold",
                prompt="Add gold charger plates at each place setting on the white tablecloth. 8 place settings around the table. Gold chiavari chairs remain.",
                input_image="base",
                output_name="Set_2_GoldChargerPlates",
                visible_to_user=False,
            ),
            Operation(
                op_type=OperationType.EDIT,
                name="cutlery_gold",
                prompt="Add gold cutlery beside each charger plate - fork on left, knife and spoon on right.",
                input_image="previous",
                output_name="Set_2_GoldChargerPlates_GoldCutlery",
                visible_to_user=False,
            ),
            Operation(
                op_type=OperationType.TABLE_REMOVAL,
                name="table_removal_set2",
                prompt="Remove the table. Leave the charger plates and cutlery floating.",
                input_image="previous",
                output_name="Set_2.5_GoldChargerPlates_Floating",
                visible_to_user=False,
            ),
            Operation(
                op_type=OperationType.FUSION,
                name="fusion_A",
                prompt="Place the floating gold charger plates and cutlery onto the table with white tablecloth and gold chiavari chairs.",
                input_image="base",
                ref_image="Set_2.5_GoldChargerPlates_Floating",
                output_name="Image_A",
                visible_to_user=True,
            ),
            
            # === SET 3.5 FUSION → Image B ===
            Operation(
                op_type=OperationType.EDIT,
                name="dinner_plates_red",
                prompt="Add red dinner plates on top of each gold charger plate.",
                input_image="Image_A",
                output_name="Set_3_RedDinnerPlates",
                visible_to_user=False,
            ),
            Operation(
                op_type=OperationType.EDIT,
                name="napkins_pink",
                prompt="Add pink satin napkins folded elegantly on each red dinner plate.",
                input_image="previous",
                output_name="Set_3_RedDinnerPlates_PinkNapkins",
                visible_to_user=False,
            ),
            Operation(
                op_type=OperationType.TABLE_REMOVAL,
                name="table_removal_set3",
                prompt="Remove the table. Leave the dinner plates and napkins floating.",
                input_image="previous",
                output_name="Set_3.5_RedDinnerPlates_PinkNapkins_Floating",
                visible_to_user=False,
            ),
            Operation(
                op_type=OperationType.FUSION,
                name="fusion_B",
                prompt="Place the floating red dinner plates and pink napkins on top of the gold charger plates.",
                input_image="Image_A",
                ref_image="Set_3.5_RedDinnerPlates_PinkNapkins_Floating",
                output_name="Image_B",
                visible_to_user=True,
            ),
            
            # === SET 4.5 FUSION → Image C ===
            Operation(
                op_type=OperationType.EDIT,
                name="centerpiece_pink_gold",
                prompt="Add a pink rose ball centerpiece on a gold stand to the center of the table.",
                input_image="Image_B",
                output_name="Set_4_PinkGoldCenterpiece",
                visible_to_user=False,
            ),
            Operation(
                op_type=OperationType.EDIT,
                name="glassware_wine",
                prompt="Add wine glasses at each place setting.",
                input_image="previous",
                output_name="Set_4_PinkGoldCenterpiece_Glassware",
                visible_to_user=False,
            ),
            Operation(
                op_type=OperationType.TABLE_REMOVAL,
                name="table_removal_set4",
                prompt="Remove the table. Leave the centerpiece and glasses floating.",
                input_image="previous",
                output_name="Set_4.5_Centerpiece_Glassware_Floating",
                visible_to_user=False,
            ),
            Operation(
                op_type=OperationType.FUSION,
                name="fusion_C",
                prompt="Place the floating pink/gold centerpiece in the center of the table and the wine glasses at each place setting.",
                input_image="Image_B",
                ref_image="Set_4.5_Centerpiece_Glassware_Floating",
                output_name="Image_C_FINAL",
                visible_to_user=True,
            ),
        ]
    )


def get_test3_config() -> TestConfig:
    """Test 3: Champagne pintuck + ghost phoenix chairs - full 8 sets"""
    test_dir = os.path.join(TESTS_DIR, "Test 3")
    
    return TestConfig(
        name="Test 3",
        test_dir=test_dir,
        base_image=os.path.join(test_dir, "Set_1_RoundTable_WhiteTablecloth_GoldChiavariChairsWhiteCushion.png"),
        operations=[
            # === SET 1: Tablecloth ===
            Operation(
                op_type=OperationType.EDIT,
                name="tablecloth_champagne_pintuck",
                prompt="Change the tablecloth on the table to a champagne/gold pintuck tablecloth matching the reference. Keep the gold chiavari chairs.",
                input_image="base",
                ref_url="https://affordableelegance.ca/wp-content/uploads/2018/02/champagne-pintuck-tablecloth.png",
                output_name="Set_1_PERMUTATION_1_ChampagnePintuckTablecloth",
                visible_to_user=True,
            ),
            
            # === SET 2: Chairs ===
            Operation(
                op_type=OperationType.EDIT,
                name="chairs_ghost_phoenix",
                prompt="Replace all chairs with clear ghost phoenix chairs matching the reference. 8 chairs evenly spaced around the round table with champagne pintuck tablecloth.",
                input_image="previous",
                ref_url="https://m.media-amazon.com/images/I/71oTqoID7zL.jpg",
                output_name="Set_2_GhostPhoenixChair",
                visible_to_user=True,
            ),
            
            # === SET 3: Charger Plates ===
            Operation(
                op_type=OperationType.EDIT,
                name="charger_plates_gold",
                prompt="Add gold charger plates at each place setting on the champagne pintuck tablecloth. 8 place settings around the table.",
                input_image="previous",
                output_name="Set_3_GoldChargerPlates",
                visible_to_user=False,
            ),
            Operation(
                op_type=OperationType.EDIT,
                name="charger_plates_purple",
                prompt="Change the charger plates to purple reef acrylic charger plates matching the reference.",
                input_image="previous",
                ref_url="https://www.cvlinens.com/cdn/shop/products/Reef-Acrylic-Plastic-Charger-Plate-Purple.jpg?v=1744804968",
                output_name="Set_3_PERMUTATION_1_PurpleChargerPlates",
                visible_to_user=False,
            ),
            Operation(
                op_type=OperationType.TABLE_REMOVAL,
                name="table_removal_set3",
                prompt="Remove the table. Leave the purple charger plates floating.",
                input_image="previous",
                output_name="Set_3.5_PERMUTATION_1_PurpleChargerPlates_Floating",
                visible_to_user=False,
            ),
            
            # === FUSION: Set 2 + Set 3.5 → Fused Image A ===
            Operation(
                op_type=OperationType.FUSION,
                name="fusion_A",
                prompt="Place the floating purple charger plates onto the table with champagne pintuck tablecloth and ghost phoenix chairs.",
                input_image="Set_2_GhostPhoenixChair",
                ref_image="Set_3.5_PERMUTATION_1_PurpleChargerPlates_Floating",
                output_name="Fused_Image_A",
                visible_to_user=True,
            ),
            
            # === SET 4: Cutlery ===
            Operation(
                op_type=OperationType.EDIT,
                name="cutlery_gold",
                prompt="Add gold cutlery beside each purple charger plate - fork on left, knife and spoon on right.",
                input_image="Fused_Image_A",
                output_name="Set_4_GoldCutlery",
                visible_to_user=False,
            ),
            Operation(
                op_type=OperationType.EDIT,
                name="cutlery_black",
                prompt="Change the cutlery to matte black cutlery matching the reference.",
                input_image="previous",
                ref_url="https://cb2.scene7.com/is/image/CB2/20PcPrllBrshdBlkFltwrSSHF22",
                output_name="Set_4_PERMUTATION_1_BlackCutlery",
                visible_to_user=False,
            ),
            Operation(
                op_type=OperationType.TABLE_REMOVAL,
                name="table_removal_set4",
                prompt="Remove the table. Leave the black cutlery floating.",
                input_image="previous",
                output_name="Set_4.5_PERMUTATION_1_BlackCutlery_Floating",
                visible_to_user=False,
            ),
            
            # === FUSION: Fused A + Set 4.5 → Fused Image B ===
            Operation(
                op_type=OperationType.FUSION,
                name="fusion_B",
                prompt="Place the floating black cutlery beside each purple charger plate on the table.",
                input_image="Fused_Image_A",
                ref_image="Set_4.5_PERMUTATION_1_BlackCutlery_Floating",
                output_name="Fused_Image_B",
                visible_to_user=True,
            ),
            
            # === SET 5: Dinner Plates ===
            Operation(
                op_type=OperationType.EDIT,
                name="dinner_plates_red",
                prompt="Add red dinner plates on top of each purple charger plate.",
                input_image="Fused_Image_B",
                output_name="Set_5_RedDinnerPlates",
                visible_to_user=False,
            ),
            Operation(
                op_type=OperationType.EDIT,
                name="dinner_plates_white",
                prompt="Change the dinner plates to simple white dinner plates matching the reference.",
                input_image="previous",
                ref_url="https://dijf55il5e0d1.cloudfront.net/images/na/9/1/3/91387_1000.jpg",
                output_name="Set_5_PERMUTATION_1_WhiteDinnerPlates",
                visible_to_user=False,
            ),
            Operation(
                op_type=OperationType.TABLE_REMOVAL,
                name="table_removal_set5",
                prompt="Remove the table. Leave the white dinner plates floating.",
                input_image="previous",
                output_name="Set_5.5_PERMUTATION_1_WhiteDinnerPlates_Floating",
                visible_to_user=False,
            ),
            
            # === FUSION: Fused B + Set 5.5 → Fused Image C ===
            Operation(
                op_type=OperationType.FUSION,
                name="fusion_C",
                prompt="Place the floating white dinner plates on top of the purple charger plates.",
                input_image="Fused_Image_B",
                ref_image="Set_5.5_PERMUTATION_1_WhiteDinnerPlates_Floating",
                output_name="Fused_Image_C",
                visible_to_user=True,
            ),
            
            # === SET 6: Napkins ===
            Operation(
                op_type=OperationType.EDIT,
                name="napkins_pink",
                prompt="Add pink napkins folded on each white dinner plate.",
                input_image="Fused_Image_C",
                output_name="Set_6_PinkNapkins",
                visible_to_user=False,
            ),
            Operation(
                op_type=OperationType.EDIT,
                name="napkins_champagne",
                prompt="Change the napkins to champagne/gold shantung napkins matching the reference.",
                input_image="previous",
                ref_url="https://bbjlatavola.com/wp-content/uploads/2020/06/Champagne-Shantung-Napkin-R.jpg",
                output_name="Set_6_PERMUTATION_1_ChampagneNapkins",
                visible_to_user=False,
            ),
            Operation(
                op_type=OperationType.TABLE_REMOVAL,
                name="table_removal_set6",
                prompt="Remove the table. Leave the champagne napkins floating.",
                input_image="previous",
                output_name="Set_6.5_PERMUTATION_1_ChampagneNapkins_Floating",
                visible_to_user=False,
            ),
            
            # === FUSION: Fused C + Set 6.5 → Fused Image D ===
            Operation(
                op_type=OperationType.FUSION,
                name="fusion_D",
                prompt="Place the floating champagne napkins on top of the white dinner plates.",
                input_image="Fused_Image_C",
                ref_image="Set_6.5_PERMUTATION_1_ChampagneNapkins_Floating",
                output_name="Fused_Image_D",
                visible_to_user=True,
            ),
            
            # === SET 7: Centerpiece ===
            Operation(
                op_type=OperationType.EDIT,
                name="centerpiece_pink_gold",
                prompt="Add a pink rose ball centerpiece on a gold stand to the center of the table.",
                input_image="Fused_Image_D",
                output_name="Set_7_PinkAndGoldFloralCenterpiece",
                visible_to_user=False,
            ),
            Operation(
                op_type=OperationType.EDIT,
                name="centerpiece_purple_black",
                prompt="Change the centerpiece to a purple and black ostrich feather centerpiece on a tall black stand matching the reference.",
                input_image="previous",
                ref_url="https://cdn1.bigcommerce.com/server600/eqcqd/products/3609/images/5264/ZFCpurpleblack_large__18514.1369340681.772.1026.jpg?c=2",
                output_name="Set_7_PERMUTATION_1_PurpleAndBlackCenterpiece",
                visible_to_user=False,
            ),
            Operation(
                op_type=OperationType.TABLE_REMOVAL,
                name="table_removal_set7",
                prompt="Remove the table. Leave the purple/black centerpiece floating.",
                input_image="previous",
                output_name="Set_7.5_PERMUTATION_1_PurpleAndBlackCenterpiece_Floating",
                visible_to_user=False,
            ),
            
            # === FUSION: Fused D + Set 7.5 → Fused Image E ===
            Operation(
                op_type=OperationType.FUSION,
                name="fusion_E",
                prompt="Place the floating purple/black feather centerpiece in the center of the table.",
                input_image="Fused_Image_D",
                ref_image="Set_7.5_PERMUTATION_1_PurpleAndBlackCenterpiece_Floating",
                output_name="Fused_Image_E",
                visible_to_user=True,
            ),
            
            # === SET 8: Glassware ===
            Operation(
                op_type=OperationType.EDIT,
                name="glassware_wine",
                prompt="Add wine glasses at each place setting above the knife.",
                input_image="Fused_Image_E",
                output_name="Set_8_WineGlasses",
                visible_to_user=False,
            ),
            Operation(
                op_type=OperationType.TABLE_REMOVAL,
                name="table_removal_set8",
                prompt="Remove the table. Leave the wine glasses floating.",
                input_image="previous",
                output_name="Set_8.5_WineGlasses_Floating",
                visible_to_user=False,
            ),
            
            # === FUSION: Fused E + Set 8.5 → Fused Image F (FINAL) ===
            Operation(
                op_type=OperationType.FUSION,
                name="fusion_F",
                prompt="Place the floating wine glasses at each place setting on the table.",
                input_image="Fused_Image_E",
                ref_image="Set_8.5_WineGlasses_Floating",
                output_name="Fused_Image_F_FINAL",
                visible_to_user=True,
            ),
        ]
    )


# =============================================================================
# TEST RUNNER
# =============================================================================

class TestRunner:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.image_cache: Dict[str, Image.Image] = {}
        self.results: List[Dict[str, Any]] = []
    
    def resolve_image_path(self, ref: str, config: TestConfig, output_dir: str) -> str:
        """Resolve image reference to actual path"""
        if ref == "base":
            return config.base_image
        elif ref == "previous":
            return None  # Will use cached previous
        elif ref.endswith(('.png', '.jpg', '.jpeg')):
            # Check if it's already a full path
            if os.path.exists(ref):
                return ref
            # Check test dir
            test_path = os.path.join(config.test_dir, ref)
            if os.path.exists(test_path):
                return test_path
            # Check output dir
            output_path = os.path.join(output_dir, ref)
            if os.path.exists(output_path):
                return output_path
        else:
            # Named reference from cache
            return None
        return ref
    
    def get_image(self, ref: str, config: TestConfig, output_dir: str) -> Image.Image:
        """Get image by reference"""
        if ref == "base":
            if "base" not in self.image_cache:
                self.image_cache["base"] = resize_to_fixed(
                    Image.open(config.base_image).convert("RGB")
                )
            return self.image_cache["base"]
        
        if ref in self.image_cache:
            return self.image_cache[ref]
        
        # Try to load from file
        path = self.resolve_image_path(ref, config, output_dir)
        if path and os.path.exists(path):
            img = resize_to_fixed(Image.open(path).convert("RGB"))
            self.image_cache[ref] = img
            return img
        
        raise ValueError(f"Cannot find image: {ref}")
    
    def run_test(self, config: TestConfig) -> Dict[str, Any]:
        """Run a complete test"""
        print_banner(f"RUNNING {config.name}", "🎨")
        
        # Setup directories
        output_dir = os.path.join(RESULTS_DIR, config.name.replace(" ", "_"))
        ref_cache_dir = os.path.join(config.test_dir, "ref_cache")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(ref_cache_dir, exist_ok=True)
        
        # Clear cache for new test
        self.image_cache.clear()
        
        # Load base image
        print(f"📷 Loading base image: {config.base_image}")
        base_img = resize_to_fixed(Image.open(config.base_image).convert("RGB"))
        self.image_cache["base"] = base_img
        base_img.save(os.path.join(output_dir, "step_0_original.png"))
        
        test_results = {
            "name": config.name,
            "output_dir": output_dir,
            "operations": [],
            "total_time": 0,
        }
        
        previous_result = None
        test_start = time.time()
        
        for i, op in enumerate(config.operations, 1):
            op_start = time.time()
            print(f"\n{'='*60}")
            print(f"Step {i}/{len(config.operations)}: {op.name}")
            print(f"Type: {op.op_type.value}")
            print(f"{'='*60}")
            
            try:
                # Get input image
                if op.input_image == "previous":
                    if previous_result is None:
                        raise ValueError("No previous result available")
                    input_img = previous_result
                else:
                    input_img = self.get_image(op.input_image, config, output_dir)
                
                # Get reference image if needed
                ref_img = None
                if op.ref_image:
                    ref_img = self.get_image(op.ref_image, config, output_dir)
                elif op.ref_url:
                    ref_img = load_or_download_image(url=op.ref_url, cache_dir=ref_cache_dir)
                    if ref_img:
                        ref_img = ensure_square(ref_img, REF_SIZE)
                
                # Run operation
                seed = BASE_SEED + i
                
                if op.op_type == OperationType.EDIT:
                    if ref_img is None:
                        ref_img = Image.new('RGB', (REF_SIZE, REF_SIZE), 'white')
                    result = run_edit(
                        self.pipeline, 
                        input_img, 
                        resize_reference(ref_img),
                        op.prompt,
                        op.steps,
                        seed
                    )
                
                elif op.op_type == OperationType.TABLE_REMOVAL:
                    result = run_table_removal(
                        self.pipeline,
                        input_img,
                        op.steps,
                        seed
                    )
                
                elif op.op_type == OperationType.FUSION:
                    if ref_img is None:
                        raise ValueError("Fusion requires reference image")
                    result = run_fusion(
                        self.pipeline,
                        input_img,
                        ref_img,
                        op.prompt,
                        op.steps,
                        seed
                    )
                
                # Save result
                output_name = op.output_name or f"step_{i}_{op.name}"
                output_path = os.path.join(output_dir, f"{output_name}.png")
                result.save(output_path)
                print(f"💾 Saved: {output_path}")
                
                # Cache result
                self.image_cache[op.output_name or op.name] = result
                previous_result = result
                
                op_time = time.time() - op_start
                print(f"⏱️  Completed in {format_time(op_time)}")
                
                test_results["operations"].append({
                    "name": op.name,
                    "type": op.op_type.value,
                    "output": output_path,
                    "time": op_time,
                    "visible_to_user": op.visible_to_user,
                    "success": True,
                })
                
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()
                test_results["operations"].append({
                    "name": op.name,
                    "type": op.op_type.value,
                    "error": str(e),
                    "success": False,
                })
        
        test_results["total_time"] = time.time() - test_start
        
        # Save test report
        report_path = os.path.join(output_dir, "report.txt")
        with open(report_path, "w") as f:
            f.write(f"{config.name} Test Report\n")
            f.write(f"{'='*50}\n")
            f.write(f"Date: {datetime.now()}\n")
            f.write(f"Total operations: {len(config.operations)}\n")
            f.write(f"Total time: {format_time(test_results['total_time'])}\n\n")
            
            for op_result in test_results["operations"]:
                status = "✅" if op_result.get("success") else "❌"
                f.write(f"{status} {op_result['name']} ({op_result['type']})")
                if op_result.get("time"):
                    f.write(f" - {op_result['time']:.2f}s")
                if op_result.get("error"):
                    f.write(f" - Error: {op_result['error']}")
                f.write("\n")
        
        print_banner(f"{config.name} COMPLETE", "✅")
        print(f"Total time: {format_time(test_results['total_time'])}")
        print(f"Results: {output_dir}")
        
        return test_results


# =============================================================================
# MAIN
# =============================================================================

def main():
    print_banner("WEDDING DECOR COMPREHENSIVE TEST RUNNER", "🎨")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load model
    pipeline = load_pipeline()
    warmup(pipeline)
    
    # Initialize runner
    runner = TestRunner(pipeline)
    
    # Get test configs
    tests = [
        get_test1_config(),
        get_test2_config(),
        get_test3_config(),
    ]
    
    # Run all tests
    all_results = []
    total_start = time.time()
    
    for test_config in tests:
        try:
            result = runner.run_test(test_config)
            all_results.append(result)
        except Exception as e:
            print(f"❌ Test {test_config.name} failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Final summary
    total_time = time.time() - total_start
    
    print_banner("ALL TESTS COMPLETE", "🏁")
    print(f"Total time: {format_time(total_time)}")
    print(f"\nResults saved to: {RESULTS_DIR}")
    
    for result in all_results:
        success_count = sum(1 for op in result["operations"] if op.get("success"))
        total_count = len(result["operations"])
        print(f"  {result['name']}: {success_count}/{total_count} operations ({format_time(result['total_time'])})")
    
    # Save overall summary
    summary_path = os.path.join(RESULTS_DIR, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("Wedding Decor Test Summary\n")
        f.write(f"{'='*50}\n")
        f.write(f"Date: {datetime.now()}\n")
        f.write(f"Total time: {format_time(total_time)}\n\n")
        
        for result in all_results:
            success_count = sum(1 for op in result["operations"] if op.get("success"))
            f.write(f"{result['name']}: {success_count}/{len(result['operations'])} ops, {format_time(result['total_time'])}\n")
    
    print(f"\n✨ Done at {datetime.now().strftime('%H:%M:%S')}")


if __name__ == "__main__":
    # Allow running specific tests
    import sys
    
    if len(sys.argv) > 1:
        test_num = int(sys.argv[1])
        print(f"Running Test {test_num} only")
        
        pipeline = load_pipeline()
        warmup(pipeline)
        runner = TestRunner(pipeline)
        
        if test_num == 1:
            runner.run_test(get_test1_config())
        elif test_num == 2:
            runner.run_test(get_test2_config())
        elif test_num == 3:
            runner.run_test(get_test3_config())
    else:
        main()
