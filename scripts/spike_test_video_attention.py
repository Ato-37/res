"""Same pipeline check as spike_test_attention.py, but with a short video
instead of a single image: sample a handful of frames ourselves (so we
control frame count and resolution directly, no decord dependency), feed
them to Qwen2.5-VL as a pre-sampled frame sequence, and check that eager
attention + output_attentions still works and see how much extra memory
the added frames cost versus the single-image test.

Usage:
    python scripts/spike_test_video_attention.py --video path/to/clip.mp4 --num-frames 4
"""

import argparse
import shutil
import tempfile

import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


def extract_frames(video_path, num_frames, max_dim):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise RuntimeError(f"Could not read frame count from {video_path}")

    indices = [round(i * (total - 1) / max(num_frames - 1, 1)) for i in range(num_frames)]
    tmp_dir = tempfile.mkdtemp(prefix="spike_video_frames_")
    frame_paths = []

    for pos, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Could not read frame {idx} from {video_path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        img.thumbnail((max_dim, max_dim))
        path = f"{tmp_dir}/frame_{pos:03d}.jpg"
        img.save(path)
        frame_paths.append(path)

    cap.release()
    print(f"video total_frames={total}, sampled indices={indices}")
    return frame_paths, tmp_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to a local test video")
    parser.add_argument("--num-frames", type=int, default=4)
    parser.add_argument("--frame-size", type=int, default=224, help="Max width/height per sampled frame")
    parser.add_argument("--prompt", default="Describe what happens in this video in one sentence.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--max-pixels", type=int, default=256 * 28 * 28, help="Vision-token budget per frame")
    parser.add_argument("--quant", choices=["4bit", "8bit", "none"], default="4bit")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}, quant={args.quant}")

    frame_paths, tmp_dir = extract_frames(args.video, args.num_frames, args.frame_size)

    try:
        if args.quant == "4bit":
            quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        elif args.quant == "8bit":
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            quant_config = None

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
                    {"type": "video", "video": frame_paths},
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

        print(f"total_input_tokens={inputs['input_ids'].shape[1]}")

        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        generated = model.generate(**inputs, max_new_tokens=30)
        output_text = processor.batch_decode(
            generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        print("generated text:", output_text)

        with torch.no_grad():
            out = model(**inputs, output_attentions=True)

        n_layers = len(out.attentions)
        layer0 = out.attentions[0]
        print(f"num_layers_with_attention={n_layers}")
        print(f"layer0_attention_shape={tuple(layer0.shape)}")

        if device == "cuda":
            peak_gb = torch.cuda.max_memory_allocated() / 1e9
            print(f"peak_gpu_memory_gb={peak_gb:.2f}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
