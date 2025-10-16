import os
import time

import gradio as gr
import numpy as np
from PIL import Image
import pydicom
from ultralytics import YOLO

def get_path() -> str:
    return os.path.dirname(os.path.abspath(__file__))

PATH_SEPARATOR = "\\" if os.name == "nt" else "/"

# model = YOLO(f"{get_path()}/yolo11n-seg-minirad.pt")
s1_model = YOLO(f"{get_path()}/yolo11n-s1.pt")
vds_model = YOLO(f"{get_path()}/yolo11m-vds.pt")

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

def segment(dicom_paths: list[str], filter_processes=True, enable_deduplication=True):
    start_time = time.time()
    if not dicom_paths:
        return [], None

    if [path for path in dicom_paths if not path.endswith(('.dcm', '.dicom'))]:
        raise gr.Error("Please select a folder containing DICOM files only.")

    class_order = [
        "S1", "L5/S1", "L5", "L4/L5", "L4", "L3/L4", "L3", "L2/L3", "L2", 
        "L1/L2", "L1", "T12/L1", "T12", "T11/T12", "T11", "T10/T11", 
        "T10", "T9/T10", "T9", "spinal_canal"
    ]

    vertebrae = [cls for cls in class_order if '/' not in cls and cls != 'spinal_canal']
    discs = [cls for cls in class_order if '/' in cls]

    segmentation_results = []

    for filename in sorted(dicom_paths, key=lambda x: x.split(PATH_SEPARATOR)[-1]):
        dicom_data = pydicom.dcmread(filename)
        pixel_array = dicom_data.pixel_array

        if pixel_array.dtype != np.uint8:
            pixel_array = pixel_array.astype(np.float32)
            pixel_array = (pixel_array - np.min(pixel_array)) / (np.max(pixel_array) - np.min(pixel_array) + 1e-6)
            pixel_array = (pixel_array * 255).astype(np.uint8)

        image = Image.fromarray(pixel_array).convert("L")
        original_shape = image.size
        resized_image = image.resize((640, 640), Image.Resampling.LANCZOS)

        s1_results = s1_model.predict(resized_image, verbose=False)
        s1_center = None
        s1_mask_obj = None  

        for result in s1_results:
            if result.masks and len(result.masks.data) > 0:
                confidence_scores = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)

                best_s1_idx = None
                best_s1_conf = 0

                for i, (conf, class_id) in enumerate(zip(confidence_scores, class_ids)):
                    class_name = s1_model.names[class_id]
                    if conf > 0.1 and class_name.lower() == 's1':
                        if conf > best_s1_conf:
                            best_s1_conf = conf
                            best_s1_idx = i

                if best_s1_idx is not None:
                    s1_mask = result.masks.data[best_s1_idx].cpu().numpy()

                    y_coords, x_coords = np.where(s1_mask > 0.5)
                    if len(y_coords) > 0:
                        y_min, y_max = np.min(y_coords), np.max(y_coords)
                        x_min, x_max = np.min(x_coords), np.max(x_coords)

                        s1_center = ((y_min + y_max) / 2, (x_min + x_max) / 2)
                        print(f"S1 center found at: {s1_center}")

                        binary_mask = np.zeros((512, 512), dtype=np.float32)
                        binary_mask[s1_mask > 0.5] = 1.0

                        resized_s1_mask = np.array(Image.fromarray(binary_mask).resize(
                            (original_shape[0], original_shape[1]), 
                            Image.Resampling.NEAREST
                        ))

                        s1_mask_obj = {
                            'mask': resized_s1_mask,
                            'confidence': best_s1_conf,
                            'class_type': 's1',
                            'assigned_class': 'S1'
                        }
                break

        if s1_center is None:
            print("Warning: S1 vertebra not detected in this image")

            segmentation_results.append((image, []))
            continue

        vertebra_disc_results = vds_model.predict(resized_image, verbose=False)

        detected_objects = []
        spinal_canal_objects = []
        disc_masks = []  

        for result in vertebra_disc_results:
            masks = result.masks
            confidence_scores = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)

            if masks and len(masks.data) > 0:
                for i, (mask_tensor, conf, class_id) in enumerate(zip(masks.data, confidence_scores, class_ids)):
                    if conf > 0.1:
                        class_name = vds_model.names[class_id].lower()
                        if class_name == 'disc':
                            mask_array = mask_tensor.cpu().numpy()
                            binary_mask = np.zeros((640, 640), dtype=np.float32)
                            binary_mask[mask_array > 0.5] = 1.0

                            resized_mask = np.array(Image.fromarray(binary_mask).resize(
                                (original_shape[0], original_shape[1]), 
                                Image.Resampling.NEAREST
                            ))
                            disc_masks.append(resized_mask)

        def is_main_vertebra(mask, disc_masks, proximity_threshold=2):
            """Filter vertebrae by checking if they touch or are close to disc masks"""
            if not disc_masks:

                return True

            vertebra_coords = np.where(mask > 0.5)
            if len(vertebra_coords[0]) == 0:
                return False

            vertebra_y, vertebra_x = vertebra_coords
            vert_y_min, vert_y_max = np.min(vertebra_y), np.max(vertebra_y)
            vert_x_min, vert_x_max = np.min(vertebra_x), np.max(vertebra_x)

            for disc_mask in disc_masks:
                disc_coords = np.where(disc_mask > 0.5)
                if len(disc_coords[0]) == 0:
                    continue

                disc_y, disc_x = disc_coords
                disc_y_min, disc_y_max = np.min(disc_y), np.max(disc_y)
                disc_x_min, disc_x_max = np.min(disc_x), np.max(disc_x)

                y_distance = max(0, max(vert_y_min - disc_y_max, disc_y_min - vert_y_max))
                x_distance = max(0, max(vert_x_min - disc_x_max, disc_x_min - vert_x_max))

                if max(y_distance, x_distance) <= proximity_threshold:
                    return True

            return False

        for result in vertebra_disc_results:
            masks = result.masks
            confidence_scores = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)

            if masks and len(masks.data) > 0:
                for i, (mask_tensor, conf, class_id) in enumerate(zip(masks.data, confidence_scores, class_ids)):
                    if conf > 0.1:
                        class_name = vds_model.names[class_id].lower()
                        mask_array = mask_tensor.cpu().numpy()

                        binary_mask = np.zeros((640, 640), dtype=np.float32)
                        binary_mask[mask_array > 0.5] = 1.0

                        resized_mask = np.array(Image.fromarray(binary_mask).resize(
                            (original_shape[0], original_shape[1]), 
                            Image.Resampling.NEAREST
                        ))

                        if class_name == 'spinal_canal':
                            spinal_canal_objects.append({
                                'mask': resized_mask,
                                'confidence': conf,
                                'class_type': class_name,
                                'assigned_class': 'spinal_canal'
                            })
                            continue

                        if filter_processes and class_name == 'vertebra':
                            if not is_main_vertebra(resized_mask, disc_masks):
                                print(f"Filtered out vertebra detection")
                                continue

                        y_coords, x_coords = np.where(resized_mask > 0.5)
                        if len(y_coords) > 0:
                            y_min, y_max = np.min(y_coords), np.max(y_coords)
                            x_min, x_max = np.min(x_coords), np.max(x_coords)

                            obj_center = ((y_min + y_max) / 2, (x_min + x_max) / 2)

                            distance = np.sqrt((obj_center[0] - s1_center[0])**2 + 
                                             (obj_center[1] - s1_center[1])**2)

                            detected_objects.append({
                                'mask': resized_mask,
                                'confidence': conf,
                                'class_type': class_name,  
                                'center': obj_center,
                                'distance_from_s1': distance,
                                'random_id': np.random.randint(1000000)  
                            })

        filtered_objects = []

        if enable_deduplication:

            detected_objects.sort(key=lambda x: x['confidence'], reverse=True)

            for obj in detected_objects:
                should_keep = True

                for kept_obj in filtered_objects:

                    intersection = np.logical_and(obj['mask'] > 0, kept_obj['mask'] > 0)
                    intersection_area = np.sum(intersection)

                    union = np.logical_or(obj['mask'] > 0, kept_obj['mask'] > 0)
                    union_area = np.sum(union)

                    if union_area > 0:
                        iou = intersection_area / union_area
                        if iou > 0.7:
                            should_keep = False
                            break

                if should_keep:
                    filtered_objects.append(obj)
        else:

            filtered_objects = detected_objects

        vertebra_objects = [obj for obj in filtered_objects if obj['class_type'].lower() == 'vertebra']
        disc_objects = [obj for obj in filtered_objects if obj['class_type'].lower() == 'disc']

        vertebra_objects.sort(key=lambda x: x['distance_from_s1'])
        disc_objects.sort(key=lambda x: x['distance_from_s1'])
        print([(obj.get("random_id"), obj.get("distance_from_s1"), obj.get("center")) for obj in vertebra_objects], flush=True)
        print([(obj.get("random_id"), obj.get("distance_from_s1"), obj.get("center")) for obj in disc_objects], flush=True)

        vertebra_idx = vertebrae.index('L5')  
        for i, obj in enumerate(vertebra_objects):
            if vertebra_idx + i < len(vertebrae):
                obj['assigned_class'] = vertebrae[vertebra_idx + i]
            else:

                obj['assigned_class'] = None

        disc_idx = discs.index('L5/S1')  
        for i, obj in enumerate(disc_objects):
            if disc_idx + i < len(discs):
                obj['assigned_class'] = discs[disc_idx + i]
            else:

                obj['assigned_class'] = None

        class_to_masks = {class_name: np.zeros((original_shape[1], original_shape[0]), dtype=np.float32)
                         for class_name in class_order}

        all_objects = [s1_mask_obj]
        all_objects += filtered_objects + spinal_canal_objects

        for obj in all_objects:
            if obj.get('assigned_class'):
                mask = obj['mask'] * 0.5  
                class_to_masks[obj['assigned_class']] = np.maximum(
                    class_to_masks[obj['assigned_class']], mask)
                print(obj.get('assigned_class'), obj.get("random_id"), flush=True)

        masks_and_classes = []
        for class_name in class_order:
            mask = class_to_masks[class_name]
            if np.any(mask > 0):
                masks_and_classes.append((mask, class_name))

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
                    switch_button = gr.Button("Futtatás", variant="primary")
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
