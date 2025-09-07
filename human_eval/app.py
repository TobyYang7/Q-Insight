#!/usr/bin/env python3
"""
Slide Quality Annotation Interface using Gradio
Quality scoring for slides from the compare dataset
"""

import gradio as gr
import json
import random
import os
import re
import glob
from datetime import datetime
from pathlib import Path

# Configuration
COMPARE_DATASET_PATH = "/data1/toby/Q-Insight/dataset/compare"
HUMAN_EVAL_FOLDER = "/data1/toby/Q-Insight/human_eval"
QUALITY_ANNOTATION_OUTPUT_PATH = os.path.join(HUMAN_EVAL_FOLDER, "quality_annotations.jsonl")

# Deficiency categories from the original app (simplified version)
DEFICIENCY_CATEGORIES = {
    "Composition & Layout": [
        "Poor Visual Hierarchy",
        "Content Alignment Issues",
        "Content Overflow/Cut-off",
        "Unbalanced Space Distribution"
    ],
    "Typography": [
        "Illegible Typeface Selection or Usage",
        "Improper Font Sizing",
        "Excessive Text Volume",
        "Improper Line/Character Spacing"
    ],
    "Imagery & Visualizations": [
        "Irrelevant Visual Content",
        "Improper Image Sizing",
        "Inconsistent Visual Style Usage",
        "Inappropriate or Mismatched Color Combinations"
    ]
}

# Chinese translations for UI elements
CHINESE_TRANSLATIONS = {
    "Composition & Layout": "构图与布局",
    "Typography": "字体排版",
    "Imagery & Visualizations": "图像与可视化",
    "Poor Visual Hierarchy": "视觉层次混乱",
    "Content Alignment Issues": "内容对齐问题",
    "Content Overflow/Cut-off": "内容溢出/被截断",
    "Unbalanced Space Distribution": "空间分布不均衡",
    "Illegible Typeface Selection or Usage": "字体选择或使用不当",
    "Improper Font Sizing": "字体大小不当",
    "Excessive Text Volume": "文本量过多",
    "Improper Line/Character Spacing": "行距/字符间距不当",
    "Irrelevant Visual Content": "视觉内容不相关",
    "Improper Image Sizing": "图像尺寸不当",
    "Inconsistent Visual Style Usage": "视觉风格使用不一致",
    "Inappropriate or Mismatched Color Combinations": "颜色搭配不当或不匹配"
}

def get_bilingual_text(english_text):
    """Get bilingual text (English / Chinese) for display"""
    chinese_text = CHINESE_TRANSLATIONS.get(english_text, english_text)
    return f"{english_text} / {chinese_text}"

def convert_bilingual_to_english(bilingual_choices):
    """Convert bilingual choices back to English for saving"""
    if not bilingual_choices:
        return []

    english_choices = []
    for choice in bilingual_choices:
        # Extract English part before " / "
        if " / " in choice:
            english_part = choice.split(" / ")[0]
            english_choices.append(english_part)
        else:
            # If no " / " found, use the original choice
            english_choices.append(choice)

    return english_choices

class SlideQualityAnnotator:
    def __init__(self):
        self.slide_data = self.load_compare_dataset()
        self.current_user = None
        self.user_progress = {}
        self.user_slide_orders = {}  # Store individual slide order for each user
        self.current_quality_index = 0

    def load_compare_dataset(self):
        """Load slide data from the compare dataset"""
        data = []
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        
        # Categories in the compare dataset
        categories = ['base', 'good', 'weak', 'gt_60']
        
        for category in categories:
            category_path = os.path.join(COMPARE_DATASET_PATH, category)
            if not os.path.exists(category_path):
                print(f"Warning: Category folder not found: {category_path}")
                continue
                
            # Find all id folders in this category
            id_folders = glob.glob(os.path.join(category_path, "id*"))
            
            for id_folder in id_folders:
                folder_name = os.path.basename(id_folder)
                
                # Find all slide images in this id folder
                slide_files = []
                for ext in image_extensions:
                    slide_files.extend(glob.glob(os.path.join(id_folder, f"*{ext}")))
                
                for slide_path in slide_files:
                    filename = os.path.basename(slide_path)
                    slide_id = os.path.splitext(filename)[0]
                    
                    # Try to read meta.txt if it exists (for gt_60 category)
                    meta_info = {}
                    meta_file = os.path.join(id_folder, "meta.txt")
                    if os.path.exists(meta_file):
                        try:
                            with open(meta_file, 'r', encoding='utf-8') as f:
                                for line in f:
                                    if ':' in line:
                                        key, value = line.strip().split(':', 1)
                                        meta_info[key.strip()] = value.strip()
                        except Exception as e:
                            print(f"Error reading meta file {meta_file}: {e}")
                    
                    data.append({
                        "image_path": slide_path,
                        "category": category,
                        "id_folder": folder_name,
                        "slide_id": slide_id,
                        "filename": filename,
                        "meta_info": meta_info
                    })
        
        # Don't shuffle here - each user will get their own random order
        print(f"Loaded {len(data)} slides from compare dataset")
        return data


    def get_slide_at_index(self, index):
        """Get a slide at a specific index for quality scoring"""
        if not self.current_user:
            return None, "No user logged in", None, {}, {}, {}, {}
        
        # Get user's slide order
        user_slide_order = self.user_slide_orders.get(self.current_user, [])
        if not user_slide_order:
            return None, "No slides available for user", None, {}, {}, {}, {}
        
        if not (0 <= index < len(user_slide_order)):
            return None, "Index out of bounds", None, {}, {}, {}, {}
        
        self.current_quality_index = index
        # Get the actual slide index from user's order
        actual_slide_index = user_slide_order[index]
        slide = self.slide_data[actual_slide_index]
        
        # Get previous scores if they exist
        composition_score = slide.get('composition_score', 3)
        typography_score = slide.get('typography_score', 3)
        imagery_score = slide.get('imagery_score', 3)
        
        if not os.path.exists(slide['image_path']):
            print(f"Warning: Image not found at {slide['image_path']}.")
            return ("https://placehold.co/600x400/FFF/000?text=Image+Not+Found",
                    self.format_slide_info(slide, not_found=True), slide['image_path'],
                    composition_score, typography_score, imagery_score)
        
        return (slide['image_path'], self.format_slide_info(slide), slide['image_path'],
                composition_score, typography_score, imagery_score)

    def format_slide_info(self, slide, not_found=False):
        """Format slide information for display"""
        if not slide:
            return "No slide selected"
        
        # Get user's total slides count
        user_slide_order = self.user_slide_orders.get(self.current_user, [])
        total_user_slides = len(user_slide_order)
        
        info = f"""
**Category:** {slide.get('category', 'N/A')}
**ID Folder:** {slide.get('id_folder', 'N/A')}
**Slide ID:** {slide.get('slide_id', 'N/A')}
**Filename:** {slide.get('filename', 'N/A')}
---
**User Progress:** {self.current_quality_index + 1} / {total_user_slides}
**Total Dataset:** {len(self.slide_data)} slides
"""
        
        # Add meta information if available
        meta_info = slide.get('meta_info', {})
        if meta_info:
            info += "\n**Meta Information:**\n"
            for key, value in meta_info.items():
                info += f"- {key}: {value}\n"
        
        if not_found:
            info += "\n\n**<span style='color:red;'>ERROR: Image file not found!</span>**"
        
        return info

    def save_quality_annotation(self, composition_score, typography_score, imagery_score,
                               composition_deficiencies, typography_deficiencies, imagery_deficiencies, index):
        """Save quality annotation to JSONL file"""
        if not self.current_user:
            return "❌ Error: No user logged in."
        
        user_slide_order = self.user_slide_orders.get(self.current_user, [])
        if not (0 <= index < len(user_slide_order)):
            return "❌ Error: Invalid index."
        
        # Get the actual slide index from user's order
        actual_slide_index = user_slide_order[index]
        slide = self.slide_data[actual_slide_index]
        
        # Calculate overall score as average
        overall_score = (composition_score + typography_score + imagery_score) / 3
        
        # Update slide data
        self.slide_data[actual_slide_index]['composition_score'] = composition_score
        self.slide_data[actual_slide_index]['typography_score'] = typography_score
        self.slide_data[actual_slide_index]['imagery_score'] = imagery_score
        self.slide_data[actual_slide_index]['overall_score'] = overall_score
        
        # Convert bilingual choices back to English for saving
        comp_def_english = convert_bilingual_to_english(composition_deficiencies)
        typo_def_english = convert_bilingual_to_english(typography_deficiencies)
        img_def_english = convert_bilingual_to_english(imagery_deficiencies)
        
        annotation = {
            "timestamp": datetime.now().isoformat(),
            "annotation_type": "quality",
            "slide_info": {
                "category": slide.get('category'),
                "id_folder": slide.get('id_folder'),
                "slide_id": slide.get('slide_id'),
                "filename": slide.get('filename'),
                "image_path": slide.get('image_path')
            },
            "scores": {
                "composition": composition_score,
                "typography": typography_score,
                "imagery": imagery_score,
                "overall": overall_score
            },
            "deficiencies": {
                "composition": comp_def_english,
                "typography": typo_def_english,
                "imagery": img_def_english
            },
            "annotator": self.current_user or "unknown"
        }
        
        # Save to user-specific file
        user_output_path = os.path.join(HUMAN_EVAL_FOLDER, f"quality_annotations_{self.current_user}.jsonl")
        with open(user_output_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(annotation, ensure_ascii=False) + '\n')
        
        # Update user progress
        if self.current_user:
            self.user_progress[self.current_user] = index + 1
            self.save_user_progress()
        
        return f"✅ Quality annotation saved for slide at index {index + 1}. Overall score: {overall_score:.2f} / 质量标注已保存，幻灯片索引 {index + 1}，总分：{overall_score:.2f}"


    def set_current_user(self, username):
        """Set current user and initialize their slide order"""
        if not username or not username.strip():
            return "❌ Please enter a valid username / 请输入有效的用户名"
        
        self.current_user = username.strip()
        
        # Initialize user's slide order if not exists
        if self.current_user not in self.user_slide_orders:
            # Create a random order for this user
            user_slide_indices = list(range(len(self.slide_data)))
            random.shuffle(user_slide_indices)
            self.user_slide_orders[self.current_user] = user_slide_indices
            print(f"Created random slide order for user: {self.current_user}")
        
        # Load user progress
        self.load_user_progress()
        
        # Set current index to user's last position or 0
        start_index = self.user_progress.get(self.current_user, 0)
        user_slide_order = self.user_slide_orders.get(self.current_user, [])
        if start_index >= len(user_slide_order):
            start_index = 0
        
        self.current_quality_index = start_index
        
        return f"✅ Welcome {self.current_user}! Starting from slide {start_index + 1} / 欢迎 {self.current_user}！从幻灯片 {start_index + 1} 开始"

    def count_annotations(self):
        """Count annotations for the current user"""
        quality_count = 0
        
        if self.current_user:
            # Count quality annotations
            quality_file = os.path.join(HUMAN_EVAL_FOLDER, f"quality_annotations_{self.current_user}.jsonl")
            if os.path.exists(quality_file):
                try:
                    with open(quality_file, 'r', encoding='utf-8') as f:
                        quality_count = len(f.readlines())
                except Exception as e:
                    print(f"Error counting quality annotations: {e}")
        
        return quality_count

    def load_user_progress(self):
        """Load user progress from file"""
        progress_file = os.path.join(HUMAN_EVAL_FOLDER, "user_progress.json")
        if os.path.exists(progress_file):
            try:
                with open(progress_file, 'r', encoding='utf-8') as f:
                    self.user_progress = json.load(f)
            except Exception as e:
                print(f"Error loading user progress: {e}")
                self.user_progress = {}
        else:
            self.user_progress = {}

    def save_user_progress(self):
        """Save user progress to file"""
        progress_file = os.path.join(HUMAN_EVAL_FOLDER, "user_progress.json")
        try:
            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump(self.user_progress, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving user progress: {e}")

# Initialize annotator
annotator = SlideQualityAnnotator()

# --- Helper functions for UI updates ---

def update_deficiency_visibility(comp_score, typo_score, img_score):
    """Update visibility of deficiency selection based on scores"""
    return (
        gr.update(visible=comp_score <= 2),
        gr.update(visible=typo_score <= 2),
        gr.update(visible=img_score <= 2)
    )

def set_user_and_start(username):
    """Set username and start annotation session"""
    status = annotator.set_current_user(username)
    if "✅" in status:
        quality_count = annotator.count_annotations()
        # Load first slide
        slide_data = load_quality_slide_at_index(annotator.current_quality_index)
        return (status, gr.update(visible=False), gr.update(visible=True),
                str(quality_count), *slide_data)
    else:
        return status, gr.update(visible=True), gr.update(visible=False), "0", *([None] * 10)

def load_quality_slide_at_index(index):
    """Load a slide for quality scoring"""
    result = annotator.get_slide_at_index(index)
    if len(result) == 6:
        image_path, slide_info, relative_path, comp_score, typo_score, img_score = result
        
        # Ensure scores are numeric
        comp_score = int(comp_score) if isinstance(comp_score, (int, float)) else 3
        typo_score = int(typo_score) if isinstance(typo_score, (int, float)) else 3
        img_score = int(img_score) if isinstance(img_score, (int, float)) else 3
        
        return (image_path,
                comp_score, typo_score, img_score,
                gr.update(visible=(comp_score <= 2)), gr.update(visible=(typo_score <= 2)),
                gr.update(visible=(img_score <= 2)), [], [], [])
    else:
        # Handle error case
        return (None, 3, 3, 3,
                gr.update(visible=False), gr.update(visible=False),
                gr.update(visible=False), [], [], [])

def submit_quality_annotation(comp_score, typo_score, img_score,
                             comp_def, typo_def, img_def, current_index):
    """Save quality annotation and go to next slide"""
    status_message = annotator.save_quality_annotation(
        comp_score, typo_score, img_score,
        comp_def, typo_def, img_def, current_index
    )
    
    # Go to next slide in user's order
    user_slide_order = annotator.user_slide_orders.get(annotator.current_user, [])
    next_index = (current_index + 1) % len(user_slide_order)
    slide_data = load_quality_slide_at_index(next_index)
    
    quality_count = annotator.count_annotations()
    
    return (next_index, *slide_data, status_message, str(quality_count))

# Create Gradio interface
with gr.Blocks(title="Slide Quality Annotation Tool / 幻灯片质量标注工具") as app:
    gr.Markdown("# 🎯 Slide Quality Annotation Tool / 幻灯片质量标注工具")
    
    # State variables
    current_quality_index = gr.State(value=0)
    
    # Username input section
    with gr.Column(visible=True) as login_section:
        gr.Markdown("## Please enter your username to start annotation / 请输入用户名开始标注")
        username_input = gr.Textbox(label="Username / 用户名", placeholder="Enter your username... / 请输入您的用户名...")
        start_btn = gr.Button("Start Annotation / 开始标注", variant="primary")
        login_status = gr.Textbox(label="Status / 状态", value="Please enter your username to continue. / 请输入用户名继续。", interactive=False)
    
    # Main annotation interface
    with gr.Column(visible=False) as main_section:
        # Progress display
        with gr.Row():
            quality_annotations_count = gr.Textbox(label="质量标注数量 / Quality Annotations", interactive=False, scale=1)
        
        # Quality Scoring Interface
        with gr.Row():
            with gr.Column(scale=1):
                quality_slide_image = gr.Image(label="Slide Image / 幻灯片图像", height=400, interactive=False)
                quality_submit_btn = gr.Button("✅ Submit and Next / 提交并下一个", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                gr.Markdown("### Quality Scores (1-5 scale) / 质量评分（1-5分制）")
                gr.Markdown("*1 = Very Poor / 很差, 2 = Poor / 差, 3 = Fair / 一般, 4 = Good / 好, 5 = Excellent / 优秀*")
                
                quality_composition_score = gr.Slider(
                    minimum=1, maximum=5, step=1, value=3,
                    label=get_bilingual_text("Composition & Layout"),
                    info="Visual hierarchy, layout, space distribution / 视觉层次、布局、空间分布"
                )
                
                quality_typography_score = gr.Slider(
                    minimum=1, maximum=5, step=1, value=3,
                    label=get_bilingual_text("Typography"),
                    info="Font selection, sizing, text styling / 字体选择、大小、文本样式"
                )
                
                quality_imagery_score = gr.Slider(
                    minimum=1, maximum=5, step=1, value=3,
                    label=get_bilingual_text("Imagery & Visualizations"),
                    info="Image quality, relevance, sizing / 图像质量、相关性、尺寸"
                )
                
                # Deficiency selection sections (initially hidden)
                with gr.Column(visible=False) as quality_composition_deficiencies_section:
                    gr.Markdown(f"**⚠️ {get_bilingual_text('Composition & Layout')} Deficiencies / 缺陷 (Score ≤ 2)**")
                    quality_composition_deficiencies = gr.CheckboxGroup(
                        choices=[get_bilingual_text(item) for item in DEFICIENCY_CATEGORIES["Composition & Layout"]],
                        label="Select deficiencies: / 选择缺陷:",
                    )
                
                with gr.Column(visible=False) as quality_typography_deficiencies_section:
                    gr.Markdown(f"**⚠️ {get_bilingual_text('Typography')} Deficiencies / 缺陷 (Score ≤ 2)**")
                    quality_typography_deficiencies = gr.CheckboxGroup(
                        choices=[get_bilingual_text(item) for item in DEFICIENCY_CATEGORIES["Typography"]],
                        label="Select deficiencies: / 选择缺陷:",
                    )
                
                with gr.Column(visible=False) as quality_imagery_deficiencies_section:
                    gr.Markdown(f"**⚠️ {get_bilingual_text('Imagery & Visualizations')} Deficiencies / 缺陷 (Score ≤ 2)**")
                    quality_imagery_deficiencies = gr.CheckboxGroup(
                        choices=[get_bilingual_text(item) for item in DEFICIENCY_CATEGORIES["Imagery & Visualizations"]],
                        label="Select deficiencies: / 选择缺陷:",
                    )
                
                gr.Markdown("---")
                quality_status_text = gr.Textbox(label="Status / 状态", value="Ready to start. / 准备开始。", interactive=False, lines=2)
    
    # Event Handlers
    
    # Username and login
    start_btn.click(
        fn=set_user_and_start,
        inputs=[username_input],
        outputs=[login_status, login_section, main_section, quality_annotations_count,
                 quality_slide_image, quality_composition_score, quality_typography_score, quality_imagery_score,
                 quality_composition_deficiencies_section, quality_typography_deficiencies_section,
                 quality_imagery_deficiencies_section, quality_composition_deficiencies,
                 quality_typography_deficiencies, quality_imagery_deficiencies]
    )
    
    # Quality scoring events
    quality_composition_score.change(
        fn=lambda score: gr.update(visible=score <= 2),
        inputs=[quality_composition_score],
        outputs=[quality_composition_deficiencies_section]
    )
    
    quality_typography_score.change(
        fn=lambda score: gr.update(visible=score <= 2),
        inputs=[quality_typography_score],
        outputs=[quality_typography_deficiencies_section]
    )
    
    quality_imagery_score.change(
        fn=lambda score: gr.update(visible=score <= 2),
        inputs=[quality_imagery_score],
        outputs=[quality_imagery_deficiencies_section]
    )
    
    quality_submit_btn.click(
        fn=submit_quality_annotation,
        inputs=[quality_composition_score, quality_typography_score, quality_imagery_score,
                quality_composition_deficiencies, quality_typography_deficiencies, quality_imagery_deficiencies,
                current_quality_index],
        outputs=[current_quality_index, quality_slide_image,
                 quality_composition_score, quality_typography_score, quality_imagery_score,
                 quality_composition_deficiencies_section, quality_typography_deficiencies_section,
                 quality_imagery_deficiencies_section, quality_composition_deficiencies,
                 quality_typography_deficiencies, quality_imagery_deficiencies,
                 quality_status_text, quality_annotations_count]
    )

if __name__ == "__main__":
    os.makedirs(HUMAN_EVAL_FOLDER, exist_ok=True)
    print(f"Slide data loaded: {len(annotator.slide_data)} slides")
    print(f"Annotations will be logged to user-specific files in: {HUMAN_EVAL_FOLDER}")
    
    app.launch(
        server_name="0.0.0.0",
        server_port=7864,
        share=True,
        show_error=True
    )
