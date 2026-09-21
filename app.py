# Old Photo AI
# Created and developed by: Armin Hamzeh
# AI Photo Restoration & Enhancement Project

import gc
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import gradio as gr
from PIL import Image
from huggingface_hub import PyTorchModelHubMixin
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    Sam2Model,
    Sam2Processor,
)


# =========================================================
# Paths / device
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DDCOLOR_DIR = BASE_DIR / "DDColor"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if DEVICE.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


# =========================================================
# DDColor
# =========================================================

def ensure_ddcolor():
    if DDCOLOR_DIR.exists():
        return

    print("DDColor not found. Downloading...")
    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "https://github.com/piddnad/DDColor.git",
            str(DDCOLOR_DIR),
        ],
        check=True,
    )


ensure_ddcolor()

import sys

sys.path.insert(0, str(DDCOLOR_DIR))

from ddcolor import DDColor, ColorizationPipeline


class DDColorHF(DDColor, PyTorchModelHubMixin):
    def __init__(self, config=None, **kwargs):
        if isinstance(config, dict):
            kwargs = {**config, **kwargs}
        super().__init__(**kwargs)


print("Loading DDColor Tiny...")

ddcolor_model = DDColorHF.from_pretrained(
    "piddnad/ddcolor_paper_tiny"
)

ddcolor_model = ddcolor_model.to(DEVICE)
ddcolor_model.eval()

colorizer = ColorizationPipeline(
    ddcolor_model,
    input_size=512,
    device=DEVICE,
)

print("DDColor ready.")


def colorize_image(image: Image.Image) -> Image.Image:
    if image is None:
        raise ValueError("تصویر وارد نشده است.")

    image = image.convert("RGB")

    image_np = np.asarray(image)

    image_bgr = cv2.cvtColor(
        image_np,
        cv2.COLOR_RGB2BGR
    )

    with torch.inference_mode():
        result_bgr = colorizer.process(image_bgr)

    result_rgb = cv2.cvtColor(
        result_bgr,
        cv2.COLOR_BGR2RGB
    )

    return Image.fromarray(result_rgb)


# =========================================================
# Grounding DINO
# =========================================================

DINO_MODEL = "IDEA-Research/grounding-dino-tiny"

print("Loading Grounding DINO...")

dino_processor = AutoProcessor.from_pretrained(
    DINO_MODEL
)

dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
    DINO_MODEL
).to(DEVICE)

dino_model.eval()

print("Grounding DINO ready.")


def detect_object(
    image: Image.Image,
    text: str,
    threshold: float = 0.30,
):
    if image is None:
        raise ValueError("تصویر وارد نشده است.")

    if not text or not text.strip():
        raise ValueError("نام شیء خالی است.")

    text = text.strip()

    if not text.endswith("."):
        text += "."

    inputs = dino_processor(
        images=image,
        text=[text],
        return_tensors="pt",
    )

    for key in inputs:
        if hasattr(inputs[key], "to"):
            inputs[key] = inputs[key].to(DEVICE)

    with torch.inference_mode():
        outputs = dino_model(**inputs)

    results = dino_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=threshold,
        text_threshold=0.25,
        target_sizes=[
            (image.height, image.width)
        ],
    )

    result = results[0]

    if len(result["boxes"]) == 0:
        return None

    best_index = int(
        torch.argmax(result["scores"]).item()
    )

    box = (
        result["boxes"][best_index]
        .detach()
        .cpu()
        .tolist()
    )

    score = float(
        result["scores"][best_index]
        .detach()
        .cpu()
    )

    return {
        "box": box,
        "score": score,
    }


# =========================================================
# SAM 2
# =========================================================

SAM_MODEL = "facebook/sam2.1-hiera-tiny"

print("Loading SAM 2...")

sam_processor = Sam2Processor.from_pretrained(
    SAM_MODEL
)

sam_model = Sam2Model.from_pretrained(
    SAM_MODEL
).to(DEVICE)

sam_model.eval()

print("SAM 2 ready.")


def create_object_mask(
    image: Image.Image,
    box
):
    x1, y1, x2, y2 = [
        float(v) for v in box
    ]

    inputs = sam_processor(
        images=image,
        input_boxes=[
            [[x1, y1, x2, y2]]
        ],
        return_tensors="pt",
    )

    for key in inputs:
        if hasattr(inputs[key], "to"):
            inputs[key] = inputs[key].to(DEVICE)

    with torch.inference_mode():
        outputs = sam_model(
            **inputs,
            multimask_output=False,
        )

    masks = sam_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"],
    )

    mask = masks[0][0]

    if mask.ndim == 3:
        mask = mask[0]

    mask = mask.numpy()

    mask = (
        (mask > 0).astype(np.uint8)
    ) * 255

    kernel = np.ones(
        (5, 5),
        dtype=np.uint8
    )

    mask = cv2.dilate(
        mask,
        kernel,
        iterations=1
    )

    return Image.fromarray(
        mask,
        mode="L"
    )


# =========================================================
# LaMa
# =========================================================

from simple_lama_inpainting import SimpleLama

print("Loading LaMa...")

lama = SimpleLama(
    device=DEVICE
)

print("LaMa ready.")


# =========================================================
# Persian commands
# =========================================================

PERSIAN_OBJECTS = {
    "ماشین": "a car.",
    "خودرو": "a car.",
    "آدم": "a person.",
    "انسان": "a person.",
    "مرد": "a man.",
    "زن": "a woman.",
    "بچه": "a child.",
    "کودک": "a child.",
    "سگ": "a dog.",
    "گربه": "a cat.",
    "درخت": "a tree.",
    "دوچرخه": "a bicycle.",
    "موتور": "a motorcycle.",
    "صندلی": "a chair.",
    "میز": "a table.",
    "کیف": "a bag.",
    "کلاه": "a hat.",
}


def extract_object_name(command: str):

    if not command:
        raise ValueError(
            "دستور حذف شیء را وارد کن."
        )

    text = command.strip().lower()

    patterns = [
        r"لطفا",
        r"خواهشا",
        r"خواهش میکنم",
        r"خواهش می‌کنم",
        r"رو حذف کن",
        r"را حذف کن",
        r"حذف کن",
        r"حذفش کن",
        r"از عکس حذف کن",
        r"از تصویر حذف کن",
        r"از عکس بردار",
        r"از تصویر بردار",
    ]

    for pattern in patterns:
        text = re.sub(
            pattern,
            "",
            text
        )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    if text in PERSIAN_OBJECTS:
        return PERSIAN_OBJECTS[text]

    return text


# =========================================================
# Processing pipeline
# =========================================================

def remove_object(
    image: Image.Image,
    object_command: str
):
    image = image.convert("RGB")

    object_query = extract_object_name(
        object_command
    )

    print(
        "Searching for:",
        object_query
    )

    detection = detect_object(
        image,
        object_query
    )

    if detection is None:
        raise ValueError(
            f"نتوانستم «{object_command}» را در تصویر پیدا کنم."
        )

    box = detection["box"]
    score = detection["score"]

    print(
        f"Confidence: {score:.3f}"
    )

    mask = create_object_mask(
        image,
        box
    )

    result = lama(
        image,
        mask
    )

    return result, mask


def process_image(
    image,
    remove_command="",
    do_colorize=True
):
    if image is None:
        raise ValueError(
            "لطفاً یک تصویر انتخاب کن."
        )

    current_image = image.convert(
        "RGB"
    )

    if (
        remove_command
        and remove_command.strip()
    ):
        current_image, _ = remove_object(
            current_image,
            remove_command
        )

    if do_colorize:
        current_image = colorize_image(
            current_image
        )

    return current_image


def cleanup_memory():

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def safe_process(
    image,
    remove_command="",
    do_colorize=True
):
    try:

        result = process_image(
            image,
            remove_command,
            do_colorize,
        )

        cleanup_memory()

        return result

    except Exception:

        cleanup_memory()

        raise


def save_output(
    image,
    prefix="result"
):
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_path = (
        OUTPUT_DIR
        / f"{prefix}_{timestamp}.png"
    )

    image.save(
        output_path,
        format="PNG"
    )

    return str(output_path)


# =========================================================
# Gradio UI
# =========================================================

def ui_colorize(image):

    if image is None:
        raise gr.Error(
            "لطفاً یک عکس انتخاب کن."
        )

    try:

        result = colorize_image(
            image
        )

        output_path = save_output(
            result,
            "colorized"
        )

        cleanup_memory()

        return result, output_path

    except Exception as e:

        cleanup_memory()

        raise gr.Error(
            str(e)
        )


def ui_remove(
    image,
    command
):

    if image is None:
        raise gr.Error(
            "لطفاً یک عکس انتخاب کن."
        )

    if (
        not command
        or not command.strip()
    ):
        raise gr.Error(
            "بنویس چه چیزی حذف شود."
        )

    try:

        result, mask = remove_object(
            image,
            command
        )

        output_path = save_output(
            result,
            "removed"
        )

        cleanup_memory()

        return (
            result,
            mask,
            output_path
        )

    except Exception as e:

        cleanup_memory()

        raise gr.Error(
            str(e)
        )


def ui_full(
    image,
    command
):

    if image is None:
        raise gr.Error(
            "لطفاً یک عکس انتخاب کن."
        )

    try:

        result = safe_process(
            image,
            command,
            True
        )

        output_path = save_output(
            result,
            "final"
        )

        cleanup_memory()

        return result, output_path

    except Exception as e:

        cleanup_memory()

        raise gr.Error(
            str(e)
        )


with gr.Blocks(
    title="Old Photo AI"
) as demo:

    gr.Markdown(
        """
        # 🖼️ Old Photo AI
        ### بازیابی، رنگی‌کردن و حذف هوشمند اشیا از عکس
        """
    )

    with gr.Tab("🎨 رنگی کردن عکس"):

        with gr.Row():

            with gr.Column():

                color_input = gr.Image(
                    type="pil",
                    label="عکس قدیمی",
                    format="png",
                )

                color_button = gr.Button(
                    "🎨 رنگی کردن",
                    variant="primary",
                )

            with gr.Column():

                color_output = gr.Image(
                    type="pil",
                    label="نتیجه",
                )

                color_download = gr.File(
                    label="دانلود نتیجه"
                )

        color_button.click(
            fn=ui_colorize,
            inputs=color_input,
            outputs=[
                color_output,
                color_download
            ],
        )

    with gr.Tab("🧹 حذف شیء"):

        with gr.Row():

            with gr.Column():

                remove_input = gr.Image(
                    type="pil",
                    label="عکس",
                    format="png",
                )

                remove_command = gr.Textbox(
                    label="چه چیزی حذف شود؟",
                    placeholder="مثلاً: ماشین را حذف کن",
                    lines=2,
                )

                remove_button = gr.Button(
                    "🧹 حذف شیء",
                    variant="primary",
                )

            with gr.Column():

                remove_output = gr.Image(
                    type="pil",
                    label="نتیجه",
                )

                remove_mask = gr.Image(
                    type="pil",
                    label="ماسک تشخیص",
                )

                remove_download = gr.File(
                    label="دانلود نتیجه"
                )

        remove_button.click(
            fn=ui_remove,
            inputs=[
                remove_input,
                remove_command
            ],
            outputs=[
                remove_output,
                remove_mask,
                remove_download
            ],
        )

    with gr.Tab("✨ پردازش کامل"):

        with gr.Row():

            with gr.Column():

                full_input = gr.Image(
                    type="pil",
                    label="عکس قدیمی",
                    format="png",
                )

                full_command = gr.Textbox(
                    label="دستور حذف",
                    placeholder="مثلاً: ماشین را حذف کن",
                    lines=2,
                )

                full_button = gr.Button(
                    "✨ پردازش کامل",
                    variant="primary",
                )

            with gr.Column():

                full_output = gr.Image(
                    type="pil",
                    label="نتیجه نهایی",
                )

                full_download = gr.File(
                    label="دانلود نتیجه"
                )

        full_button.click(
            fn=ui_full,
            inputs=[
                full_input,
                full_command
            ],
            outputs=[
                full_output,
                full_download
            ],
        )

    gr.Markdown(
        """
        ---
        **سازنده: آرمین حمزه — پروژه هوش مصنوعی**

        موتورهای پروژه:
        DDColor • Grounding DINO • SAM 2 • LaMa
        """
    )


if __name__ == "__main__":

    print("=" * 50)
    print("Old Photo AI")
    print("Python device:", DEVICE)
    print("=" * 50)

    demo.launch(
        debug=True,
        share=True
    )
