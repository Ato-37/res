"""Smallest possible pipeline check: load a Qwen2.5-VL model in eager attention
mode, run one image + one text prompt through it, and report whether
output_attentions works plus how much GPU memory it used.

This is a wiring/feasibility check, not a real experiment. Run it before
touching video or the 7B model on the RTX5090 box.

Usage:
    python scripts/spike_test_attention.py --image path/to/image.jpg --prompt "What is in this image?"
"""

import argparse

import torch
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to a local test image")
    parser.add_argument("--prompt", default="Describe this image in one sentence.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    # 28*28 is one vision-token's worth of pixels for this model; capping both
    # min and max at the same value forces a fixed, small token budget so the
    # vision tower doesn't blow up memory on a high-resolution test image.
    parser.add_argument("--max-pixels", type=int, default=256 * 28 * 28)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        quantization_config=quant_config,
        device_map=device,
    )
    processor = AutoProcessor.from_pretrained(
        args.model,
        min_pixels=256 * 28 * 28,
        max_pixels=args.max_pixels,
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": args.image},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # 1. Sanity check: does generation still produce sensible text under eager attention?
    generated = model.generate(**inputs, max_new_tokens=30)
    output_text = processor.batch_decode(
        generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    print("generated text:", output_text)

    # 2. Attention check: can we pull per-head attention weights out of a forward pass?
    with torch.no_grad():
        out = model(**inputs, output_attentions=True)

    n_layers = len(out.attentions)
    layer0 = out.attentions[0]
    print(f"num_layers_with_attention={n_layers}")
    print(f"layer0_attention_shape={tuple(layer0.shape)}")  # expect [batch, heads, seq, seq]

    if device == "cuda":
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"peak_gpu_memory_gb={peak_gb:.2f}")


if __name__ == "__main__":
    main()
