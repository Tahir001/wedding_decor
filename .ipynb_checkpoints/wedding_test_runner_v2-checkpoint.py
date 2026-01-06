"""
===============================================================================
WEDDING DECOR COMPREHENSIVE TEST RUNNER v2
===============================================================================
Uses LOCAL reference images (no URL downloads needed).
Based on Qwen-Image-Edit-2509 research:
- Reference images: 384x384 (optimal for text encoder control)
- Base/output images: 1024x1024 (max ~1 megapixel supported)
- Lightning 8-step LoRA for fast inference
===============================================================================
"""

import os
import gc
import sys
import time
import math
import torch
from PIL import Image
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from diffusers import QwenImageEditPlusPipeline, FlowMatchEulerDiscreteScheduler

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths - Update for your environment
BASE_DIR = "/workspace/wedding_decor"
IMAGES_DIR = os.path.join(BASE_DIR, "images")
TESTS_DIR = os.path.join(IMAGES_DIR, "tests")
RESULTS_DIR = os.path.join(TESTS_DIR, "Results")

# Model settings
LORA_WEIGHTS = "Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-8steps-V1.0-fp32.safetensors"

# Image dimensions (based on Qwen-Image-Edit-2509 research)
# - Reference images: 384x384 (optimal for text encoder control per musubi-tuner)
# - Base images: 1024x1024 (max 1 megapixel supported)
FIXED_WIDTH = 1024
FIXED_HEIGHT = 1024
REF_SIZE = 384

# Generation settings
BASE_SEED = 42
DEFAULT_STEPS = 8


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def print_banner(text, char="="):
    line = char * 70
    print(f"\n{line}")
    print(f"  {text}")
    print(f"{line}\n")


def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.2f}s"
    return f"{int(seconds // 60)}m {seconds % 60:.1f}s"


def resize_to_fixed(img: Image.Image) -> Image.Image:
    """Resize image to fixed output dimensions"""
    return img.resize((FIXED_WIDTH, FIXED_HEIGHT), Image.LANCZOS)


def resize_reference(img: Image.Image) -> Image.Image:
    """Resize reference image to optimal control size (384x384)"""
    w, h = img.size
    if w != h:
        # Center crop to square
        min_dim = min(w, h)
        left = (w - min_dim) // 2
        top = (h - min_dim) // 2
        img = img.crop((left, top, left + min_dim, top + min_dim))
    return img.resize((REF_SIZE, REF_SIZE), Image.LANCZOS)


def ensure_square(img: Image.Image, target_size: int) -> Image.Image:
    """Center-crop and resize to square"""
    w, h = img.size
    if w != h:
        min_dim = min(w, h)
        left = (w - min_dim) // 2
        top = (h - min_dim) // 2
        img = img.crop((left, top, left + min_dim, top + min_dim))
    return img.resize((target_size, target_size), Image.LANCZOS)


def load_local_image(path: str, as_reference: bool = False) -> Optional[Image.Image]:
    """Load image from local path"""
    if not os.path.exists(path):
        print(f"   ❌ File not found: {path}")
        return None
    
    img = Image.open(path).convert("RGB")
    
    if as_reference:
        img = resize_reference(img)
        print(f"   📷 Loaded reference: {os.path.basename(path)} → {img.size}")
    else:
        img = resize_to_fixed(img)
        print(f"   📷 Loaded: {os.path.basename(path)} → {img.size}")
    
    return img


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_pipeline():
    """Load and configure the Qwen-Image-Edit-2509 pipeline"""
    print_banner("LOADING MODEL")
    
    gc.collect()
    torch.cuda.empty_cache()
    
    # Scheduler configuration for dynamic shifting
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
    
    # Load Lightning LoRA for faster inference
    print("⚡ Loading Lightning 8-step LoRA...")
    pipeline.load_lora_weights(
        "lightx2v/Qwen-Image-Lightning",
        weight_name=LORA_WEIGHTS
    )
    print("✅ LoRA loaded")
    
    pipeline.set_progress_bar_config(disable=True)
    
    return pipeline


def warmup(pipeline):
    """Run warmup inference to optimize CUDA kernels"""
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

def run_edit(pipeline, input_img: Image.Image, ref_img: Image.Image, 
             prompt: str, steps: int, seed: int) -> Image.Image:
    """Run edit operation with reference image"""
    with torch.inference_mode():
        output = pipeline(
            image=[input_img, ref_img],
            prompt=prompt,
            negative_prompt="blurry, distorted, low quality, deformed, artifacts, missing items, wrong count, wrong position",
            num_inference_steps=steps,
            true_cfg_scale=1.0,
            guidance_scale=1.0,
            generator=torch.Generator("cuda").manual_seed(seed),
        )
    
    result = output.images[0]
    if result.size != (FIXED_WIDTH, FIXED_HEIGHT):
        result = resize_to_fixed(result)
    return result


def run_table_removal(pipeline, input_img: Image.Image, 
                      steps: int, seed: int) -> Image.Image:
    """Remove table, keep products floating"""
    # Use blank reference for removal
    ref_img = Image.new('RGB', (REF_SIZE, REF_SIZE), (250, 250, 250))
    
    prompt = (
        "Remove the table from this image. Leave all products floating in the air "
        "exactly where they are. Do not change size, position, or orientation of any items. "
        "Keep a clean white/light gray background."
    )
    
    with torch.inference_mode():
        output = pipeline(
            image=[input_img, ref_img],
            prompt=prompt,
            negative_prompt="table, tablecloth, furniture, surface, fabric draping",
            num_inference_steps=steps,
            true_cfg_scale=1.0,
            guidance_scale=1.0,
            generator=torch.Generator("cuda").manual_seed(seed),
        )
    
    result = output.images[0]
    if result.size != (FIXED_WIDTH, FIXED_HEIGHT):
        result = resize_to_fixed(result)
    return result


def run_fusion(pipeline, base_img: Image.Image, overlay_img: Image.Image,
               prompt: str, steps: int, seed: int) -> Image.Image:
    """Fuse floating elements onto base table scene"""
    # Overlay goes to reference position
    overlay_resized = ensure_square(overlay_img, REF_SIZE)
    
    with torch.inference_mode():
        output = pipeline(
            image=[base_img, overlay_resized],
            prompt=prompt,
            negative_prompt="misaligned, wrong position, duplicated items, missing items, blurry, distorted",
            num_inference_steps=steps,
            true_cfg_scale=1.0,
            guidance_scale=1.0,
            generator=torch.Generator("cuda").manual_seed(seed),
        )
    
    result = output.images[0]
    if result.size != (FIXED_WIDTH, FIXED_HEIGHT):
        result = resize_to_fixed(result)
    return result


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class OperationType(Enum):
    EDIT = "edit"
    TABLE_REMOVAL = "table_removal"
    FUSION = "fusion"


@dataclass
class Operation:
    name: str
    op_type: OperationType
    prompt: str
    output_name: str
    input_image: str = "previous"  # "base", "previous", or specific output name
    ref_image: Optional[str] = None  # Local path to reference image
    steps: int = DEFAULT_STEPS
    visible_to_user: bool = True


@dataclass
class TestConfig:
    name: str
    test_dir: str
    base_image: str
    operations: List[Operation] = field(default_factory=list)


# =============================================================================
# TEST CONFIGURATIONS - Using Local Reference Images
# =============================================================================

def get_test1_config() -> TestConfig:
    """
    Test 1: Blue pintuck tablecloth + Ghost chiavari chairs
    → Green/gold chargers + Rose gold cutlery
    → Blue/white dinner plates + Rose gold napkins
    → Champagne flutes + Green centerpiece
    """
    test_dir = os.path.join(TESTS_DIR, "Test 1")
    ref_dir = os.path.join(test_dir, "reference_images_test1")
    
    return TestConfig(
        name="Test 1",
        test_dir=test_dir,
        base_image=os.path.join(test_dir, "Set_1_RoundTable_WhiteTablecloth_GoldChiavariChairsWhiteCushion.png"),
        operations=[
            # === SET 1: Tablecloth + Chairs ===
            Operation(
                name="tablecloth_blue_pintuck",
                op_type=OperationType.EDIT,
                input_image="base",
                ref_image=os.path.join(ref_dir, "Royal-Blue-Pintuck-Tablecloth.jpg"),
                prompt="Change the tablecloth on the table to a royal blue pintuck tablecloth matching the reference. Keep the gold chiavari chairs with white cushions. 8 chairs around the round table.",
                output_name="Set_1_BluePintuckTablecloth",
                visible_to_user=False,
            ),
            Operation(
                name="chairs_ghost",
                op_type=OperationType.EDIT,
                input_image="Set_1_BluePintuckTablecloth",
                ref_image=os.path.join(ref_dir, "Acrylic-Ghost-Clear-Wedding-Chiavari-Chairs.png"),
                prompt="Replace all 8 chairs with clear acrylic ghost chiavari chairs with white cushions matching the reference. Keep the blue pintuck tablecloth on the round table.",
                output_name="Set_1_PERMUTATION_1_BluePintuck_GhostChairs",
                visible_to_user=True,
            ),
            
            # === SET 2: Charger Plates + Cutlery ===
            Operation(
                name="charger_plates_green_gold",
                op_type=OperationType.EDIT,
                input_image="Set_1_PERMUTATION_1_BluePintuck_GhostChairs",
                ref_image=os.path.join(ref_dir, "green_and_gold_charger_plates.jpg"),
                prompt="Add 8 green and gold baroque charger plates at each place setting on the blue pintuck tablecloth, matching the reference. One at each seat position. Ghost chiavari chairs around the table.",
                output_name="Set_2_ChargerPlates_NoTable",
                visible_to_user=False,
            ),
            Operation(
                name="table_removal_set2",
                op_type=OperationType.TABLE_REMOVAL,
                input_image="Set_2_ChargerPlates_NoTable",
                prompt="",
                output_name="Set_2.5_ChargerPlates_Floating",
                visible_to_user=False,
            ),
            Operation(
                name="cutlery_rose_gold",
                op_type=OperationType.EDIT,
                input_image="Set_2.5_ChargerPlates_Floating",
                ref_image=os.path.join(ref_dir, "stainless-steel-rose-gold-cutlery-set.png"),
                prompt="Add rose gold cutlery next to each floating charger plate. Fork on left, knife and spoon on right of each plate. Matching the reference style.",
                output_name="Set_2_ChargerPlates_RoseGoldCutlery",
                visible_to_user=False,
            ),
            Operation(
                name="table_removal_set2_cutlery",
                op_type=OperationType.TABLE_REMOVAL,
                input_image="Set_2_ChargerPlates_RoseGoldCutlery",
                prompt="",
                output_name="Set_2.5_ChargerPlates_Cutlery_Floating",
                visible_to_user=False,
            ),
            Operation(
                name="fusion_A",
                op_type=OperationType.FUSION,
                input_image="Set_1_PERMUTATION_1_BluePintuck_GhostChairs",
                ref_image="Set_2.5_ChargerPlates_Cutlery_Floating",
                prompt="Place the floating green/gold charger plates with rose gold cutlery onto the blue pintuck tablecloth table with ghost chiavari chairs. Align 8 place settings evenly around the round table.",
                output_name="Image_A",
                visible_to_user=True,
            ),
            
            # === SET 3: Dinner Plates + Napkins ===
            Operation(
                name="dinner_plates_blue_white",
                op_type=OperationType.EDIT,
                input_image="Image_A",
                ref_image=os.path.join(ref_dir, "vintage_floral_charger_plate_blue.png"),
                prompt="Add 8 blue and white vintage floral dinner plates on top of each charger plate, matching the reference. Keep the cutlery on the sides.",
                output_name="Set_3_DinnerPlates",
                visible_to_user=False,
            ),
            Operation(
                name="napkins_rose_gold",
                op_type=OperationType.EDIT,
                input_image="Set_3_DinnerPlates",
                ref_image=os.path.join(ref_dir, "rectangular_rose_gold_napkins.png"),
                prompt="Add rose gold satin napkins folded elegantly on top of each dinner plate, matching the reference. Keep all other elements in place.",
                output_name="Set_3_DinnerPlates_Napkins",
                visible_to_user=False,
            ),
            Operation(
                name="table_removal_set3",
                op_type=OperationType.TABLE_REMOVAL,
                input_image="Set_3_DinnerPlates_Napkins",
                prompt="",
                output_name="Set_3.5_DinnerPlates_Napkins_Floating",
                visible_to_user=False,
            ),
            Operation(
                name="fusion_B",
                op_type=OperationType.FUSION,
                input_image="Image_A",
                ref_image="Set_3.5_DinnerPlates_Napkins_Floating",
                prompt="Place the floating dinner plates with napkins onto the table setup. Stack dinner plates on charger plates, napkins on top. Keep all existing elements.",
                output_name="Image_B",
                visible_to_user=True,
            ),
            
            # === SET 4: Centerpiece + Glassware ===
            Operation(
                name="centerpiece_green",
                op_type=OperationType.EDIT,
                input_image="Image_B",
                ref_image=os.path.join(ref_dir, "green_crystal_centerpiece.png"),
                prompt="Add a green crystal vase centerpiece in the center of the round table, matching the reference. Keep all place settings around it.",
                output_name="Set_4_Centerpiece",
                visible_to_user=False,
            ),
            Operation(
                name="glassware_champagne_flutes",
                op_type=OperationType.EDIT,
                input_image="Set_4_Centerpiece",
                ref_image=None,  # No specific reference
                prompt="Add champagne flutes at each place setting, positioned at the top right of each plate. 8 champagne glasses total.",
                output_name="Set_4_Centerpiece_Glassware",
                visible_to_user=False,
            ),
            Operation(
                name="table_removal_set4",
                op_type=OperationType.TABLE_REMOVAL,
                input_image="Set_4_Centerpiece_Glassware",
                prompt="",
                output_name="Set_4.5_Centerpiece_Glassware_Floating",
                visible_to_user=False,
            ),
            Operation(
                name="fusion_C",
                op_type=OperationType.FUSION,
                input_image="Image_B",
                ref_image="Set_4.5_Centerpiece_Glassware_Floating",
                prompt="Place the floating centerpiece and champagne glasses onto the table. Centerpiece in middle, glasses at each place setting. Final complete table setup.",
                output_name="Image_C_FINAL",
                visible_to_user=True,
            ),
        ]
    )


def get_test2_config() -> TestConfig:
    """
    Test 2: Gold chiavari chairs + Gold chargers/cutlery
    → Red plates + Pink napkins
    → Pink/gold centerpiece + Wine glasses
    """
    test_dir = os.path.join(TESTS_DIR, "Test 2")
    ref_dir = os.path.join(test_dir, "reference_images_test2")
    
    return TestConfig(
        name="Test 2",
        test_dir=test_dir,
        base_image=os.path.join(test_dir, "Set_1_RoundTable_WhiteTablecloth_GoldChiavariChairsWhiteCushion.png"),
        operations=[
            # === SET 1: Chairs ===
            Operation(
                name="chairs_gold_chiavari",
                op_type=OperationType.EDIT,
                input_image="base",
                ref_image=None,  # Keep existing gold chiavari
                prompt="Keep the gold chiavari chairs with white cushions around the round table with white tablecloth. 8 chairs total.",
                output_name="Set_1_GoldChiavari",
                visible_to_user=True,
            ),
            
            # === SET 2: Chargers + Cutlery ===
            Operation(
                name="charger_plates_gold",
                op_type=OperationType.EDIT,
                input_image="Set_1_GoldChiavari",
                ref_image=None,
                prompt="Add 8 gold charger plates at each place setting on the white tablecloth. One at each seat position around the round table.",
                output_name="Set_2_GoldChargerPlates",
                visible_to_user=False,
            ),
            Operation(
                name="cutlery_gold",
                op_type=OperationType.EDIT,
                input_image="Set_2_GoldChargerPlates",
                ref_image=None,
                prompt="Add gold cutlery at each place setting. Fork on left, knife and spoon on right of each charger plate.",
                output_name="Set_2_GoldChargerPlates_GoldCutlery",
                visible_to_user=False,
            ),
            Operation(
                name="table_removal_set2",
                op_type=OperationType.TABLE_REMOVAL,
                input_image="Set_2_GoldChargerPlates_GoldCutlery",
                prompt="",
                output_name="Set_2.5_Chargers_Cutlery_Floating",
                visible_to_user=False,
            ),
            Operation(
                name="fusion_A",
                op_type=OperationType.FUSION,
                input_image="Set_1_GoldChiavari",
                ref_image="Set_2.5_Chargers_Cutlery_Floating",
                prompt="Place the floating gold charger plates and cutlery onto the white tablecloth table with gold chiavari chairs. 8 place settings evenly around the table.",
                output_name="Image_A",
                visible_to_user=True,
            ),
            
            # === SET 3: Plates + Napkins ===
            Operation(
                name="plates_red",
                op_type=OperationType.EDIT,
                input_image="Image_A",
                ref_image=None,
                prompt="Add 8 red dinner plates on top of each gold charger plate. Keep the cutlery on the sides.",
                output_name="Set_3_RedPlates",
                visible_to_user=False,
            ),
            Operation(
                name="napkins_pink",
                op_type=OperationType.EDIT,
                input_image="Set_3_RedPlates",
                ref_image=None,
                prompt="Add pink satin napkins folded elegantly on top of each red dinner plate. Keep all other elements.",
                output_name="Set_3_RedPlates_PinkNapkins",
                visible_to_user=False,
            ),
            Operation(
                name="table_removal_set3",
                op_type=OperationType.TABLE_REMOVAL,
                input_image="Set_3_RedPlates_PinkNapkins",
                prompt="",
                output_name="Set_3.5_Plates_Napkins_Floating",
                visible_to_user=False,
            ),
            Operation(
                name="fusion_B",
                op_type=OperationType.FUSION,
                input_image="Image_A",
                ref_image="Set_3.5_Plates_Napkins_Floating",
                prompt="Place the floating red plates with pink napkins onto the table. Stack on charger plates, napkins on top.",
                output_name="Image_B",
                visible_to_user=True,
            ),
            
            # === SET 4: Centerpiece + Glassware ===
            Operation(
                name="centerpiece_pink_gold",
                op_type=OperationType.EDIT,
                input_image="Image_B",
                ref_image=None,
                prompt="Add a pink and gold floral centerpiece in the center of the round table. Elegant roses and gold accents.",
                output_name="Set_4_Centerpiece",
                visible_to_user=False,
            ),
            Operation(
                name="glassware_wine",
                op_type=OperationType.EDIT,
                input_image="Set_4_Centerpiece",
                ref_image=None,
                prompt="Add wine glasses at each place setting, positioned at the top right of each plate. 8 wine glasses total.",
                output_name="Set_4_Centerpiece_WineGlasses",
                visible_to_user=False,
            ),
            Operation(
                name="table_removal_set4",
                op_type=OperationType.TABLE_REMOVAL,
                input_image="Set_4_Centerpiece_WineGlasses",
                prompt="",
                output_name="Set_4.5_Centerpiece_Glassware_Floating",
                visible_to_user=False,
            ),
            Operation(
                name="fusion_C",
                op_type=OperationType.FUSION,
                input_image="Image_B",
                ref_image="Set_4.5_Centerpiece_Glassware_Floating",
                prompt="Place the floating centerpiece and wine glasses onto the table. Centerpiece in middle, glasses at each setting. Final complete table.",
                output_name="Image_C_FINAL",
                visible_to_user=True,
            ),
        ]
    )


def get_test3_config() -> TestConfig:
    """
    Test 3: Most comprehensive - 8 sets
    Champagne pintuck + Ghost phoenix chairs
    → Purple chargers → Black cutlery → White plates
    → Champagne napkins → Purple/black centerpiece → Wine glasses
    """
    test_dir = os.path.join(TESTS_DIR, "Test 3")
    ref_dir = os.path.join(test_dir, "reference_images_test3")
    
    return TestConfig(
        name="Test 3",
        test_dir=test_dir,
        base_image=os.path.join(test_dir, "Set_1_RoundTable_WhiteTablecloth_GoldChiavariChairsWhiteCushion.png"),
        operations=[
            # === SET 1: Tablecloth ===
            Operation(
                name="tablecloth_champagne_pintuck",
                op_type=OperationType.EDIT,
                input_image="base",
                ref_image=os.path.join(ref_dir, "champagne-pintuck-tablecloth.png"),
                prompt="Change the tablecloth to a champagne/beige pintuck tablecloth matching the reference. Keep the gold chiavari chairs with white cushions.",
                output_name="Set_1_ChampagnePintuck",
                visible_to_user=False,
            ),
            
            # === SET 2: Chairs ===
            Operation(
                name="chairs_ghost_phoenix",
                op_type=OperationType.EDIT,
                input_image="Set_1_ChampagnePintuck",
                ref_image=os.path.join(ref_dir, "clear_oval_chairs.jpg"),
                prompt="Replace all 8 chairs with clear acrylic ghost phoenix chairs matching the reference. Oval back design, transparent. Keep the champagne pintuck tablecloth.",
                output_name="Set_2_GhostPhoenixChairs",
                visible_to_user=True,
            ),
            
            # === SET 3: Charger Plates ===
            Operation(
                name="charger_plates_purple",
                op_type=OperationType.EDIT,
                input_image="Set_2_GhostPhoenixChairs",
                ref_image=os.path.join(ref_dir, "Reef-Acrylic-Plastic-Charger-Plate-Purple.jpg"),
                prompt="Add 8 purple reef acrylic charger plates at each place setting, matching the reference. One at each seat position.",
                output_name="Set_3_PurpleChargers",
                visible_to_user=False,
            ),
            Operation(
                name="table_removal_set3",
                op_type=OperationType.TABLE_REMOVAL,
                input_image="Set_3_PurpleChargers",
                prompt="",
                output_name="Set_3.5_PurpleChargers_Floating",
                visible_to_user=False,
            ),
            Operation(
                name="fusion_chargers",
                op_type=OperationType.FUSION,
                input_image="Set_2_GhostPhoenixChairs",
                ref_image="Set_3.5_PurpleChargers_Floating",
                prompt="Place the floating purple charger plates onto the champagne pintuck tablecloth table. 8 chargers evenly around the table.",
                output_name="Image_A_Chargers",
                visible_to_user=True,
            ),
            
            # === SET 4: Cutlery ===
            Operation(
                name="cutlery_black",
                op_type=OperationType.EDIT,
                input_image="Image_A_Chargers",
                ref_image=os.path.join(ref_dir, "black_cutlery_set.jpg"),
                prompt="Add black matte cutlery at each place setting matching the reference. Fork on left, knife and spoon on right of each charger plate.",
                output_name="Set_4_BlackCutlery",
                visible_to_user=False,
            ),
            Operation(
                name="table_removal_set4",
                op_type=OperationType.TABLE_REMOVAL,
                input_image="Set_4_BlackCutlery",
                prompt="",
                output_name="Set_4.5_Cutlery_Floating",
                visible_to_user=False,
            ),
            Operation(
                name="fusion_cutlery",
                op_type=OperationType.FUSION,
                input_image="Image_A_Chargers",
                ref_image="Set_4.5_Cutlery_Floating",
                prompt="Place the floating black cutlery onto the table beside each purple charger plate.",
                output_name="Image_B_Cutlery",
                visible_to_user=True,
            ),
            
            # === SET 5: Plates ===
            Operation(
                name="plates_white",
                op_type=OperationType.EDIT,
                input_image="Image_B_Cutlery",
                ref_image=os.path.join(ref_dir, "pure_white_dinner_plate.jpg"),
                prompt="Add 8 pure white dinner plates on top of each purple charger plate, matching the reference. Keep cutlery on the sides.",
                output_name="Set_5_WhitePlates",
                visible_to_user=False,
            ),
            Operation(
                name="table_removal_set5",
                op_type=OperationType.TABLE_REMOVAL,
                input_image="Set_5_WhitePlates",
                prompt="",
                output_name="Set_5.5_Plates_Floating",
                visible_to_user=False,
            ),
            Operation(
                name="fusion_plates",
                op_type=OperationType.FUSION,
                input_image="Image_B_Cutlery",
                ref_image="Set_5.5_Plates_Floating",
                prompt="Place the floating white dinner plates on top of each purple charger plate.",
                output_name="Image_C_Plates",
                visible_to_user=True,
            ),
            
            # === SET 6: Napkins ===
            Operation(
                name="napkins_champagne",
                op_type=OperationType.EDIT,
                input_image="Image_C_Plates",
                ref_image=os.path.join(ref_dir, "Champagne-Shantung-Napkin-R.jpg"),
                prompt="Add champagne shantung napkins folded elegantly on each white dinner plate, matching the reference.",
                output_name="Set_6_ChampagneNapkins",
                visible_to_user=False,
            ),
            Operation(
                name="table_removal_set6",
                op_type=OperationType.TABLE_REMOVAL,
                input_image="Set_6_ChampagneNapkins",
                prompt="",
                output_name="Set_6.5_Napkins_Floating",
                visible_to_user=False,
            ),
            Operation(
                name="fusion_napkins",
                op_type=OperationType.FUSION,
                input_image="Image_C_Plates",
                ref_image="Set_6.5_Napkins_Floating",
                prompt="Place the floating champagne napkins onto each white dinner plate.",
                output_name="Image_D_Napkins",
                visible_to_user=True,
            ),
            
            # === SET 7: Centerpiece ===
            Operation(
                name="centerpiece_purple_black",
                op_type=OperationType.EDIT,
                input_image="Image_D_Napkins",
                ref_image=os.path.join(ref_dir, "centerpiece_purpleblack_tree.jpg"),
                prompt="Add a purple and black tree branch centerpiece in the center of the table, matching the reference. Elegant and dramatic.",
                output_name="Set_7_Centerpiece",
                visible_to_user=False,
            ),
            Operation(
                name="table_removal_set7",
                op_type=OperationType.TABLE_REMOVAL,
                input_image="Set_7_Centerpiece",
                prompt="",
                output_name="Set_7.5_Centerpiece_Floating",
                visible_to_user=False,
            ),
            Operation(
                name="fusion_centerpiece",
                op_type=OperationType.FUSION,
                input_image="Image_D_Napkins",
                ref_image="Set_7.5_Centerpiece_Floating",
                prompt="Place the floating purple/black tree centerpiece in the center of the table.",
                output_name="Image_E_Centerpiece",
                visible_to_user=True,
            ),
            
            # === SET 8: Glassware ===
            Operation(
                name="glassware_wine",
                op_type=OperationType.EDIT,
                input_image="Image_E_Centerpiece",
                ref_image=os.path.join(ref_dir, "Set_8.5_WineGlasses.jpeg"),
                prompt="Add wine glasses at each place setting, positioned at the top right of each plate. 8 wine glasses matching the reference style.",
                output_name="Set_8_WineGlasses",
                visible_to_user=False,
            ),
            Operation(
                name="table_removal_set8",
                op_type=OperationType.TABLE_REMOVAL,
                input_image="Set_8_WineGlasses",
                prompt="",
                output_name="Set_8.5_Glassware_Floating",
                visible_to_user=False,
            ),
            Operation(
                name="fusion_final",
                op_type=OperationType.FUSION,
                input_image="Image_E_Centerpiece",
                ref_image="Set_8.5_Glassware_Floating",
                prompt="Place the floating wine glasses at each place setting. Final complete elegant table setup.",
                output_name="Image_F_FINAL",
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
    
    def get_image(self, ref: str, config: TestConfig, output_dir: str) -> Image.Image:
        """Get image by reference ID or path"""
        # Check cache first
        if ref in self.image_cache:
            return self.image_cache[ref]
        
        # Check if it's a file path
        if os.path.exists(ref):
            img = Image.open(ref).convert("RGB")
            img = resize_to_fixed(img)
            self.image_cache[ref] = img
            return img
        
        # Check output directory
        for ext in ['', '.png', '.jpg', '.jpeg']:
            path = os.path.join(output_dir, f"{ref}{ext}")
            if os.path.exists(path):
                img = Image.open(path).convert("RGB")
                img = resize_to_fixed(img)
                self.image_cache[ref] = img
                return img
        
        # Check test directory
        for ext in ['', '.png', '.jpg', '.jpeg']:
            path = os.path.join(config.test_dir, f"{ref}{ext}")
            if os.path.exists(path):
                img = Image.open(path).convert("RGB")
                img = resize_to_fixed(img)
                self.image_cache[ref] = img
                return img
        
        raise ValueError(f"Cannot find image: {ref}")
    
    def run_test(self, config: TestConfig) -> dict:
        """Run a complete test"""
        print_banner(f"RUNNING {config.name}", "🎨")
        
        # Setup directories
        output_dir = os.path.join(RESULTS_DIR, config.name.replace(" ", "_"))
        os.makedirs(output_dir, exist_ok=True)
        
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
                    # Check if it's a cached result or a file path
                    if op.ref_image in self.image_cache:
                        ref_img = self.image_cache[op.ref_image]
                        print(f"   📷 Using cached: {op.ref_image}")
                    elif os.path.exists(op.ref_image):
                        ref_img = load_local_image(op.ref_image, as_reference=True)
                    else:
                        # Try to find in output dir
                        ref_path = os.path.join(output_dir, f"{op.ref_image}.png")
                        if os.path.exists(ref_path):
                            ref_img = load_local_image(ref_path, as_reference=True)
                        else:
                            print(f"   ⚠️  Reference not found: {op.ref_image}")
                
                # Run operation
                seed = BASE_SEED + i
                
                if op.op_type == OperationType.EDIT:
                    if ref_img is None:
                        ref_img = Image.new('RGB', (REF_SIZE, REF_SIZE), 'white')
                        print(f"   ℹ️  Using blank reference (no ref image specified)")
                    else:
                        ref_img = resize_reference(ref_img)
                    result = run_edit(
                        self.pipeline, 
                        input_img, 
                        ref_img,
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
    print_banner("WEDDING DECOR COMPREHENSIVE TEST RUNNER v2", "🎨")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"\nImage sizes:")
    print(f"  - Base/Output: {FIXED_WIDTH}x{FIXED_HEIGHT}")
    print(f"  - Reference: {REF_SIZE}x{REF_SIZE}")
    
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
