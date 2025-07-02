# Virtual Staging

This project performs virtual staging on empty room images using a sophisticated pipeline of generative AI models. The core goal is to generate high-quality, realistic staged photos while strictly preserving the structural integrity and 3D geometry of the original empty room.

## Approach

The architecture is built on a synergistic pipeline of state-of-the-art models, each chosen for its specific strengths. The training and inference processes are designed to be robust and produce geometrically consistent results.

### 1. Data Preparation: The "Grounding" Pipeline

To create high-quality training data, we avoid brittle segmentation techniques. Instead, we use a "grounding" pipeline that leverages the contextual understanding of a large multimodal model (Florence-2) to guide a powerful segmentation model (SAMv2).

1. **Captioning:** Florence-2 generates a rich, descriptive caption for a staged image.
2. **Phrase Grounding:** The image and its own caption are fed back into Florence-2. The model identifies and provides precise bounding boxes for the objects it just described (e.g., a box for "the grey sofa").
3. **Guided Segmentation:** These high-confidence bounding boxes are passed to SAMv2, which generates perfect, pixel-level masks for only those objects.

This process is fast, accurate, and semantically aware, producing superior training data.

### 2. Training Strategy: Two-Phase Learning

We use a two-phase training strategy to teach the ControlNet model two distinct skills in sequence:

1. **Unpaired Pre-training (Art School):** The model is first trained on a large dataset of *staged* images. Its task is to re-create the furniture within masked-out regions of these images. This teaches the model a rich visual vocabulary of what high-quality furniture, lighting, and textures look like, independent of any specific room.
2. **Paired Fine-tuning (The Client Project):** The pre-trained model is then fine-tuned on our `paired` dataset (empty room vs. staged room). This teaches the model the specific skill of applying its stylistic knowledge while strictly preserving the walls, windows, and lighting of the original empty room.

### 3. Inference: The "Structural Integrity" Pipeline

To ensure the final output respects the room's geometry, the inference process uses a multi-ControlNet guidance system.

1. **Structural Analysis:** The input empty room is analyzed to produce a **Canny edge map** (for sharp lines) and a **Depth map** (for 3D geometry).
2. **Triple-ControlNet Guidance:** When generating a "pseudo-staged" image for layout, we use three ControlNets simultaneously: our fine-tuned **Inpaint** model (for style), the **Canny** model (for structure), and the **Depth** model (for 3D perspective). This creates a strong structural scaffold that prevents hallucinations like trees growing from walls.
3. **Final Inpainting:** The high-quality pseudo-image is then used with the "Grounding" pipeline to generate a precise mask, which guides the final, high-resolution inpainting pass.

## Setup & Workflow

Ensure you have the necessary libraries installed from `requirements.txt`.

The entire workflow, from data preparation to training and inference, is orchestrated by the `test.sh` script. Please refer to this file for the exact commands and execution order. It is the single source of truth for running the project.

## Outlook & Future Work

This project serves as a robust proof-of-concept. Areas for future improvement include:

* [ ] **Scaling to High Resolution:** The current models (SD 1.5 at 512x512) are excellent for experimentation. The next step is to adapt the pipeline for modern, highser-resolution models like SDXL to produce production-quality images.
* [ ] **User Interface (UI):** Developing a Gradio or web-based UI to make the tool accessible to non-technical users (e.g., real estate agents) for easy image upload and prompt entry.
* [ ] **Advanced Style Control:** Finer control over object placement, style blending, and negative object constraints (e.g., "add a sofa but no lamps").
* [ ] **Handling Partially Furnished Rooms:** Extending the logic to intelligently add or replace furniture in rooms that are not completely empty.
