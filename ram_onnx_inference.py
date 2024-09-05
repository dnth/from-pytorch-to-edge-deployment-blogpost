import time

import numpy as np
import onnxruntime as ort
from PIL import Image


def transform_numpy(image):
    # Convert to RGB
    image = image.convert("RGB")

    # Resize
    image = image.resize((384, 384), Image.BILINEAR)

    # Convert to numpy array and normalize
    img_np = np.array(image).astype(np.float32) / 255.0

    # Transpose from (H, W, C) to (C, H, W)
    img_np = img_np.transpose(2, 0, 1)

    # Normalize
    mean = np.array([0.485, 0.456, 0.406]).reshape(-1, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(-1, 1, 1)
    img_np = (img_np - mean) / std

    img_np = img_np.astype(np.float32)

    return img_np


def postprocess(output, tag_list, tag_list_chinese, delete_tag_index=None):
    tags = output[0]

    if delete_tag_index is not None:
        tags[:, delete_tag_index] = 0

    tag_output = []
    tag_output_chinese = []

    for b in range(tags.shape[0]):
        index = np.argwhere(tags[b] == 1)
        token = np.array(tag_list)[index].squeeze(axis=1)
        tag_output.append(" | ".join(token))
        token_chinese = np.array(tag_list_chinese)[index].squeeze(axis=1)
        tag_output_chinese.append(" | ".join(token_chinese))

    return tag_output, tag_output_chinese


def load_tag_lists():
    with open("ram_tag_list.txt", "r") as f:
        tag_list = [line.strip() for line in f.readlines()]
    with open("ram_tag_list_chinese.txt", "r", encoding="utf-8") as f:
        tag_list_chinese = [line.strip() for line in f.readlines()]
    return tag_list, tag_list_chinese


def create_onnx_session(model_path, provider):
    providers = [provider] if provider != "CPUExecutionProvider" else [provider]
    session = ort.InferenceSession(model_path, providers=providers)
    return session


def run_inference(image_path, session, tag_list, tag_list_chinese):
    # Load and preprocess the image
    image = Image.open(image_path)
    transformed_image = transform_numpy(image)
    transformed_image = np.expand_dims(transformed_image, axis=0)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Run inference
    output = session.run([output_name], {input_name: transformed_image})

    # Postprocess the output
    processed_output, processed_output_chinese = postprocess(
        output, tag_list, tag_list_chinese
    )

    return processed_output, processed_output_chinese


def perform_warmup(session):
    print("Performing warmup inference...")
    warmup_image = np.random.rand(1, 3, 384, 384).astype(np.float32)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    for _ in range(10):
        _ = session.run([output_name], {input_name: warmup_image})
    print("Warmup inference completed (10 iterations).")
    print()


def run_and_print_results(image_path, session, tag_list, tag_list_chinese, model_name):
    start_time = time.time()
    english_tags, chinese_tags = run_inference(
        image_path, session, tag_list, tag_list_chinese
    )
    runtime = (time.time() - start_time) * 1000

    print(f"\n{model_name} results:")
    print(f"English tags ({model_name}):", english_tags)
    print(f"Chinese tags ({model_name}):", chinese_tags)
    print(f"Runtime per image ({model_name}): {runtime:.2f} ms")

    return english_tags, chinese_tags, runtime


if __name__ == "__main__":
    image_path = "image_test_ram.jpg"
    model_path = "ram.onnx"

    tag_list, tag_list_chinese = load_tag_lists()

    providers = ["CPUExecutionProvider"]

    for provider in providers:
        try:
            session = create_onnx_session(model_path, provider)
            print(f"\nUsing {provider}:")
            perform_warmup(session)
            english_tags, chinese_tags, runtime = run_and_print_results(
                image_path, session, tag_list, tag_list_chinese, provider
            )
        except Exception as e:
            print(f"Error with {provider}: {str(e)}")
