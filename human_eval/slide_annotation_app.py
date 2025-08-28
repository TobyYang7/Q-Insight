#!/usr/bin/env python3
"""
Slide Quality Annotation Interface using Gradio
Sequentially displays slides from training data for quality scoring and updates the data.
Allows for navigation within a paper's slides by changing the filename index.
"""

import gradio as gr
import json
import random
import os
import re
from datetime import datetime
from pathlib import Path

# Configuration
ROOT_FOLDER = "./paper-slide-crawler"  
HUMAN_EVAL_FOLDER = "human_eval"
TRAIN_DATA_PATH = "dataset/slide_quality_train.json"
ANNOTATION_OUTPUT_PATH = os.path.join(HUMAN_EVAL_FOLDER, "slide_annotations.jsonl")
SKIPPED_OUTPUT_PATH = os.path.join(HUMAN_EVAL_FOLDER, "skipped_slides.jsonl")


class SlideAnnotator:
    def __init__(self):
        self.train_data = self.load_train_data()
        self.current_item = None
        self.current_index = 0

    def load_train_data(self):
        """Load training data from JSON file"""
        if not os.path.exists(TRAIN_DATA_PATH):
            print(f"Error: Training data file not found at {TRAIN_DATA_PATH}")
            return [{"image_path": "not_found.jpg", "paper_id": "N/A", "conference": "N/A", "slide_id": "N/A"}]
        with open(TRAIN_DATA_PATH, 'r') as f:
            data = json.load(f)
            try:
                data.sort(key=lambda x: (x.get('paper_id', ''), int(x.get('slide_id', 0))))
            except (ValueError, TypeError):
                print("Warning: Could not sort slides by slide_id. Using default file order.")
            return data

    def get_slide_at_index(self, index):
        """Get a slide at a specific index from the training data"""
        if not (0 <= index < len(self.train_data)):
            return None, "Index out of bounds", None

        self.current_index = index
        self.current_item = self.train_data[index]
        image_path = os.path.join(ROOT_FOLDER, self.current_item["image_path"])
        relative_path = self.current_item["image_path"]

        if not os.path.exists(image_path):
            print(f"Warning: Image not found at {image_path}.")
            return "https://placehold.co/600x400/FFF/000?text=Image+Not+Found", self.format_slide_info(not_found=True), relative_path

        return image_path, self.format_slide_info(), relative_path

    def navigate_intra_paper(self, direction):
        """
        Navigates to the next/previous slide within the same paper by +/- the filename.
        This modifies the current_item in memory.
        """
        if not self.current_item:
            return "https://placehold.co/600x400/FFF/000?text=No+Slide+Loaded", "No slide selected", ""

        current_relative_path = self.current_item.get("image_path", "")
        p = Path(current_relative_path)

        match = re.search(r'(\d+)', p.stem)
        if not match:
            full_path = os.path.join(ROOT_FOLDER, current_relative_path)
            return full_path, self.format_slide_info(), current_relative_path

        slide_num_str = match.group(1)
        new_slide_num = int(slide_num_str) + direction
        if new_slide_num < 0:
            new_slide_num = 0

        new_stem = p.stem.replace(slide_num_str, str(new_slide_num), 1)
        new_relative_path = str(p.with_stem(new_stem))

        self.current_item["image_path"] = new_relative_path
        self.current_item["slide_id"] = str(new_slide_num)
        self.train_data[self.current_index] = self.current_item

        full_display_path = os.path.join(ROOT_FOLDER, new_relative_path)

        if not os.path.exists(full_display_path):
            print(f"Warning: Image not found at {full_display_path}.")
            return "https://placehold.co/600x400/FFF/000?text=Image+Not+Found", self.format_slide_info(not_found=True), new_relative_path

        return full_display_path, self.format_slide_info(), new_relative_path

    def format_slide_info(self, not_found=False):
        """Format slide information for display"""
        if not self.current_item:
            return "No slide selected"

        info = f"""
**Paper ID:** {self.current_item.get('paper_id', 'N/A')}
**Conference:** {self.current_item.get('conference', 'N/A')}
**Slide ID:** {self.current_item.get('slide_id', 'N/A')}
---
**Dataset Index:** {self.current_index + 1} / {len(self.train_data)}
"""
        if not_found:
            info += "\n\n**<span style='color:red;'>ERROR: Image file not found!</span>**"
        return info

    def save_annotation(self, score, index):
        """Save annotation to JSONL file and update the score in the loaded data"""
        if not (0 <= index < len(self.train_data)):
            return "❌ Error: Invalid index."

        self.current_item = self.train_data[index]

        self.train_data[index]['new_score'] = score
        self.train_data[index]['original_score'] = score

        annotation = {
            "timestamp": datetime.now().isoformat(),
            "paper_id": self.current_item.get('paper_id'),
            "slide_id": self.current_item.get('slide_id'),
            "image_path": self.current_item.get('image_path'),
            "new_score": score,
            "annotator": "gradio_interface_sequential"
        }

        with open(ANNOTATION_OUTPUT_PATH, 'a', encoding='utf-8') as f:
            f.write(json.dumps(annotation, ensure_ascii=False) + '\n')

        return f"✅ Score {score} saved for slide at index {index + 1}."

    def save_changes_to_file(self):
        """Saves the modified train_data list back to the original JSON file."""
        try:
            backup_path = TRAIN_DATA_PATH + f".bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.rename(TRAIN_DATA_PATH, backup_path)
            with open(TRAIN_DATA_PATH, 'w', encoding='utf-8') as f:
                json.dump(self.train_data, f, indent=4, ensure_ascii=False)
            return f"✅ Successfully saved changes to {TRAIN_DATA_PATH}. Backup created at {backup_path}"
        except Exception as e:
            return f"❌ Error saving file: {e}"


# Initialize annotator
annotator = SlideAnnotator()

# --- Gradio UI and Logic Functions ---


def load_slide_at_index(index):
    """Loads a slide and updates the UI for a given index."""
    image_path, slide_info, relative_path = annotator.get_slide_at_index(index)
    current_score = annotator.train_data[index].get('new_score', 5.0)
    return image_path, slide_info, current_score, relative_path


def change_slide_in_paper(direction):
    """Changes the slide number in the filename and updates the UI."""
    return annotator.navigate_intra_paper(direction)


def go_to_next_slide(current_index):
    """Navigate to the next slide in the dataset."""
    next_index = min(len(annotator.train_data) - 1, current_index + 1)
    return next_index, *load_slide_at_index(next_index)


def go_to_previous_slide(current_index):
    """Navigate to the previous slide in the dataset."""
    prev_index = max(0, current_index - 1)
    return prev_index, *load_slide_at_index(prev_index)


def submit_and_go_next(score, current_index):
    """Save the score and automatically move to the next slide."""
    status_message = annotator.save_annotation(score, current_index)
    next_index, image_path, slide_info, new_score, relative_path = go_to_next_slide(current_index)
    return next_index, image_path, slide_info, new_score, relative_path, status_message


# Create Gradio interface
with gr.Blocks(title="Slide Quality Annotation Tool") as app:
    gr.Markdown("# 🎯 Slide Quality Annotation Tool (Sequential)")
    current_index_state = gr.State(value=0)

    with gr.Row():
        with gr.Column(scale=2):
            slide_image = gr.Image(label="Slide Image", height=500, interactive=False)

        with gr.Column(scale=1):
            slide_info = gr.Markdown(label="Slide Information")
            image_path_display = gr.Textbox(label="Image Path", interactive=False)

            score_slider = gr.Slider(
                minimum=1, maximum=10, step=0.01, value=5.0,
                label="Quality Score (1-10)", info="1 = Very Poor, 10 = Excellent"
            )

            gr.Markdown("--- \n ### Navigate by Filename")
            with gr.Row():
                intra_prev_btn = gr.Button("<")
                intra_next_btn = gr.Button(">")

            gr.Markdown("--- \n ### Navigate by Dataset Index")
            # Removed the "Previous ◀️" button as requested
            # prev_btn = gr.Button("◀️ Previous")
            # Removed the "Next ▶️" button as requested
            # next_btn = gr.Button("Next ▶️")

            submit_btn = gr.Button("✅ Submit and Next", variant="primary")
            save_btn = gr.Button("💾 Save All Changes to File", variant="stop")
            status_text = gr.Textbox(label="Status", value="Ready to start.", interactive=False)

    # Event Handlers
    # Removed the prev_btn event handler as requested
    # prev_btn.click(
    #     fn=go_to_previous_slide,
    #     inputs=[current_index_state],
    #     outputs=[current_index_state, slide_image, slide_info, score_slider, image_path_display, prev_btn]
    # )

    # The event handler for the "Next ▶️" button has been removed
    # next_btn.click(...)

    submit_btn.click(
        fn=submit_and_go_next,
        inputs=[score_slider, current_index_state],
        # The outputs list has been updated to remove the 'prev_btn' reference
        outputs=[current_index_state, slide_image, slide_info, score_slider, image_path_display, status_text]
    )

    intra_prev_btn.click(
        fn=lambda: change_slide_in_paper(-1),
        inputs=[],
        outputs=[slide_image, slide_info, image_path_display]
    )

    intra_next_btn.click(
        fn=lambda: change_slide_in_paper(1),
        inputs=[],
        outputs=[slide_image, slide_info, image_path_display]
    )

    save_btn.click(
        fn=annotator.save_changes_to_file,
        inputs=[],
        outputs=[status_text]
    )

    app.load(
        fn=load_slide_at_index,
        inputs=[current_index_state],
        # The outputs list has been updated to remove the 'prev_btn' reference
        outputs=[slide_image, slide_info, score_slider, image_path_display]
    )

if __name__ == "__main__":
    os.makedirs(HUMAN_EVAL_FOLDER, exist_ok=True)
    print(f"Training data loaded: {len(annotator.train_data)} slides")
    print(f"Annotations will be logged to: {ANNOTATION_OUTPUT_PATH}")

    app.launch(
        server_name="0.0.0.0",
        server_port=7862,
        share=True,
        show_error=True
    )
