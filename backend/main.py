import os
import time

import gradio as gr
import numpy as np
from PIL import Image
import pydicom
from scipy.ndimage import label
import openvino as ov


def get_path() -> str:
    return os.path.dirname(os.path.abspath(__file__))


PATH_SEPARATOR = "\\" if os.name == "nt" else "/"

# Initialize OpenVINO
core = ov.Core()

# Detect available devices and prioritize GPU over CPU
available_devices = core.available_devices
device = "CPU"
if "GPU" in available_devices:
    device = "GPU"
    print(f"Using GPU for inference")
else:
    print(f"GPU not available, using CPU for inference")

# Load OpenVINO INT8 models
spine_model_path = f"{get_path()}/spine_model/model_int8.xml"
s1_model_path = f"{get_path()}/s1_model/model_int8.xml"

# Compile models for the selected device
spine_model = core.compile_model(spine_model_path, device)
s1_model = core.compile_model(s1_model_path, device)

# Get output layers for inference
spine_output_layer = spine_model.output(0)
s1_output_layer = s1_model.output(0)

css = """
.tab-wrapper, footer, #prev_button, #next_button {
  display: none !important;
}
"""

js = """
function() {
    // Sort gradio.File component
    const sortFileTable = function() {
        observer.disconnect();
        const fileTables = document.querySelectorAll('table.file-preview');

        fileTables.forEach(table => {
            const tbody = table.querySelector('tbody');
            if (!tbody || tbody.children.length <= 1) return;

            const rows = Array.from(tbody.querySelectorAll('tr.file'));
            const sortedRows = rows.sort((a, b) => {
                const filenameA = (
                    (a.querySelector('.stem')?.textContent || '') + 
                    (a.querySelector('.ext')?.textContent || '')
                ).toLowerCase();
                const filenameB = (
                    (b.querySelector('.stem')?.textContent || '') + 
                    (b.querySelector('.ext')?.textContent || '')
                ).toLowerCase();

                return filenameA.localeCompare(filenameB);
            });

            rows.forEach(row => tbody.removeChild(row));
            sortedRows.forEach(row => tbody.appendChild(row));
        });

        observer.observe(document.body, {
            childList: true, 
            subtree: true
        });
    };
    const observer = new MutationObserver((mutations) => {
        for (const mutation of mutations) {
            if (mutation.type === 'childList') {
                setTimeout(sortFileTable, 50);
                break;
            }
        }
    });
    observer.observe(document.body, {
        childList: true, 
        subtree: true
    });

    // Create AnnotatedImage scrolling
    document.getElementById('annotated_image').addEventListener('wheel', function(event) {
        event.preventDefault();
        if (event.deltaY > 0) {
            document.getElementById('next_button').click();
        } else {
            document.getElementById('prev_button').click();
        }
    });
}
"""

def cleanup_segmentation(prediction, probabilities):
    """
    Cleans up a segmentation mask by merging small, incorrectly classified regions
    based on class proximity and confidence.

    Args:
        prediction (np.ndarray): The initial segmentation mask (2D array of class indices).
        probabilities (np.ndarray): The softmax probabilities for each class (3D array, C x H x W).

    Returns:
        np.ndarray: The cleaned-up segmentation mask.
    """
    cleaned_prediction = np.copy(prediction)

    # Use scipy.ndimage.label for connected components
    labeled_mask, num_features = label(prediction > 0)

    for i in range(1, num_features + 1):
        region_mask = (labeled_mask == i)
        classes_in_region = np.unique(prediction[region_mask])

        if len(classes_in_region) > 1:
            # Check if classes are close enough to merge (and not background)
            class_diff = np.max(classes_in_region) - np.min(classes_in_region)

            if class_diff > 0 and class_diff <= 2:
                # Calculate average confidence for each class in the region
                avg_confidences = {}
                for c in classes_in_region:
                    if c == 0: continue # Skip background

                    # Get the confidence of the specific class for the region
                    class_probs = probabilities[c, :, :]
                    avg_confidences[c] = np.mean(class_probs[region_mask])

                if not avg_confidences: continue

                # Find the class with the highest average confidence
                dominant_class = max(avg_confidences, key=avg_confidences.get)

                # Merge the region to the dominant class
                cleaned_prediction[region_mask] = dominant_class

    return cleaned_prediction


def segment(dicom_paths: list[str]):
    start_time = time.time()
    if not dicom_paths:
        return [], None

    if [path for path in dicom_paths if not path.endswith(('.dcm', '.dicom'))]:
        raise gr.Error("Please select a folder containing DICOM files only.")

    # Combined model class order after merging S1 and spine models
    # Class 0 is background
    # Classes 1-9 are from spine_model: L5, L4, L3, L2, L1, T12, T11, T10, T9
    # Class 10 is S1 (from s1_model)
    # Classes 11-19 are from spine_model: spinal_canal, L5/S1, L4/L5, L3/L4, L2/L3, L1/L2, T12/L1, T11/T12, T10/T11, T9/T10
    model_class_order = [
        "L5", "L4", "L3", "L2", "L1", "T12", "T11", "T10", "T9", "S1",
        "spinal_canal", "L5/S1", "L4/L5", "L3/L4", "L2/L3", "L1/L2", "T12/L1", "T11/T12", "T10/T11", "T9/T10"
    ]

    # Desired display order
    display_order = [
        "S1", "L5/S1", "L5", "L4/L5", "L4", "L3/L4", "L3", "L2/L3", "L2", "L1/L2", "L1",
        "T12/L1", "T12", "T11/T12", "T11", "T10/T11", "T10", "T9/T10", "T9", "spinal_canal"
    ]

    segmentation_results = []

    for filename in sorted(dicom_paths, key=lambda x: x.split(PATH_SEPARATOR)[-1]):
        dicom_data = pydicom.dcmread(filename)
        pixel_array = dicom_data.pixel_array

        # Normalize pixel array to 0-1 range for model input
        if pixel_array.dtype != np.float32:
            pixel_array = pixel_array.astype(np.float32)
        pixel_array = (pixel_array - np.min(pixel_array)) / (np.max(pixel_array) - np.min(pixel_array) + 1e-6)

        # Create PIL image for display (8-bit)
        display_array = (pixel_array * 255).astype(np.uint8)
        image = Image.fromarray(display_array).convert("L")
        original_shape = image.size

        # Resize image for model input
        resized_image = Image.fromarray(pixel_array).resize((512, 512), Image.Resampling.LANCZOS)

        # Convert to numpy array and add batch and channel dimensions
        input_array = np.array(resized_image).astype(np.float32)
        input_array = np.expand_dims(np.expand_dims(input_array, axis=0), axis=0)  # Shape: (1, 1, 512, 512)

        # Run inference on both models
        # Spine model prediction
        spine_output = spine_model(input_array)[spine_output_layer]
        # Apply softmax manually
        spine_exp = np.exp(spine_output - np.max(spine_output, axis=1, keepdims=True))
        spine_probabilities = (spine_exp / np.sum(spine_exp, axis=1, keepdims=True)).squeeze(0)
        spine_predictions = np.argmax(spine_output, axis=1).squeeze(0)

        # S1 model prediction (outputs class 10 for S1)
        s1_output = s1_model(input_array)[s1_output_layer]
        # Apply softmax manually
        s1_exp = np.exp(s1_output - np.max(s1_output, axis=1, keepdims=True))
        s1_probabilities = (s1_exp / np.sum(s1_exp, axis=1, keepdims=True)).squeeze(0)
        s1_predictions = np.argmax(s1_output, axis=1).squeeze(0)

        # Merge predictions: use S1 where predicted, otherwise use spine model
        # S1 model outputs: 0 (background) or 1 (S1), we map 1 -> 10
        predictions = spine_predictions.copy()
        s1_mask = s1_predictions == 1
        predictions[s1_mask] = 10

        # Shift spine model's classes 10-19 to make room for S1 at position 10
        high_class_mask = (spine_predictions >= 10) & (~s1_mask)
        predictions[high_class_mask] = spine_predictions[high_class_mask] + 1

        # Merge probabilities (21 classes total: background + 20 classes)
        merged_probabilities = np.zeros((21, predictions.shape[0], predictions.shape[1]), dtype=np.float32)
        # Classes 0-9 from spine model
        merged_probabilities[0:10] = spine_probabilities[0:10]
        # Class 10 from s1 model
        merged_probabilities[10] = s1_probabilities[1]
        # Classes 11-20 from spine model (shifted)
        merged_probabilities[11:21] = spine_probabilities[10:20]

        # Clean up small misclassified regions
        predictions = cleanup_segmentation(predictions, merged_probabilities)

        # Resize predictions back to original shape
        predictions_resized = np.array(Image.fromarray(predictions.astype(np.uint8)).resize(
            (original_shape[0], original_shape[1]),
            Image.Resampling.NEAREST
        ))

        # Create masks for each class (skip class 0 which is background)
        masks_dict = {}
        for class_idx, class_name in enumerate(model_class_order, start=1):
            # Create binary mask for this class
            class_mask = (predictions_resized == class_idx).astype(np.float32) * 0.5

            if np.any(class_mask > 0):
                masks_dict[class_name] = class_mask
                print(f"Found {class_name} in image")

        # Reorder masks according to display_order
        masks_and_classes = []
        for class_name in display_order:
            if class_name in masks_dict:
                masks_and_classes.append((masks_dict[class_name], class_name))

        segmentation_results.append((image, masks_and_classes))

    print(f"Processing time: {time.time() - start_time:.2f} seconds")
    print("Segmentation completed.")
    return segmentation_results, segmentation_results[0] if segmentation_results else (None, [])

def show_dicom(dicom_paths: list[str] | None) -> str:
    if dicom_paths is None:
        dicom_paths = []

    for html_path in [hp for hp in sorted(os.listdir(get_path())) if hp.startswith("viewer-iframe-")]:
        os.remove(f"{get_path()}/{html_path}")

    with open(f"{get_path()}/viewer.html") as file:
        html = file.read()

    html = html.replace("[]; // Replace this", str(sorted(dicom_paths, key=lambda x: x.split(PATH_SEPARATOR)[-1])))
    filename = f"viewer-iframe-{int(time.time() * 1000)}.html"

    with open(f"{get_path()}/{filename}", "w") as file:
        file.write(html)

    return f"""<iframe id="dicom-iframe" style="width: 100%; height: calc(100vh - 52px); border: none;" src="gradio_api/file={get_path()}/{filename}"></iframe>"""

def change_image(segmented_image_array: list, current_image_index: int, direction: int):
    return (new_index := min(max(current_image_index + direction, 0), len(segmented_image_array) - 1)), segmented_image_array[new_index]

with gr.Blocks(css=css, js=js, title="QuickRAD") as demo:
    segmented_image_array = gr.State([])
    current_image_index = gr.State(0)

    with gr.Tabs() as tabs:
        with gr.TabItem("Input", id=0):
            with gr.Row():
                with gr.Column():
                    file_explorer = gr.File(file_count="directory", label="DICOM mappa kiválasztása")
                    switch_button = gr.Button("Futtatás")
                with gr.Column():
                    dicom_viewer = gr.HTML(value=f"""<iframe id="dicom-iframe" style="width: 100%; height: calc(100vh - 52px); border: none;" src="gradio_api/file={get_path()}/viewer.html"></iframe>""", padding=False)

        with gr.TabItem("Output", id=1):
            annotated_output = gr.AnnotatedImage(
                height="calc(100vh - 108px)", # 26px top, 26px bottom, 56px button + gap
                elem_id="annotated_image",
                show_label=False,
                show_fullscreen_button=False
            )
            back_button = gr.Button("Vissza", elem_id="back_button")
            prev_button = gr.Button("", elem_id="prev_button")
            next_button = gr.Button("", elem_id="next_button")

    file_explorer.change(
        fn=segment,
        inputs=[file_explorer],
        outputs=[segmented_image_array, annotated_output],
    )

    file_explorer.change(
        fn=show_dicom,
        inputs=[file_explorer],
        outputs=[dicom_viewer],
        show_progress="hidden"
    )

    switch_button.click(
        fn=lambda: gr.Tabs(selected=1),
        inputs=[],
        outputs=[tabs]
    )

    back_button.click(
        fn=lambda: gr.Tabs(selected=0),
        inputs=[],
        outputs=[tabs]
    )

    prev_button.click(
        fn=change_image,
        inputs=[segmented_image_array, current_image_index, gr.State(-1)],
        outputs=[current_image_index, annotated_output],
        show_progress="hidden"
    )

    next_button.click(
        fn=change_image,
        inputs=[segmented_image_array, current_image_index, gr.State(1)],
        outputs=[current_image_index, annotated_output],
        show_progress="hidden"
    )

if __name__ == "__main__":
    demo.launch(allowed_paths=[get_path()], server_port=35565)
