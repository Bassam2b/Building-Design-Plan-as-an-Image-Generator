# Building-Design-Plan-as-an-Image-Generator
This capstone project delivers a fully automated training pipeline for Stable Diffusion models using a custom dataset of architectural floor plans. It integrates every stage of the workflow from dataset preparation and verification to LoRA fine-tuning and high-quality image generation making it a production-ready system for real-world applications.

Key Features

Custom Dataset Support: Trained on 2,450 floor plan images with structured captions including dimensions, building types, and special requirements.

Dataset Verification: Automated checks for metadata validity, file existence, schema correctness, and image dimension compliance.

Preprocessing Pipeline: Intelligent cropping, resizing (512×512), RGB normalization, and quality optimization.

Training System: LoRA fine-tuning (rank=16, dropout=0.1) on Stable Diffusion 2.0 with mixed precision (fp16), AdamW optimizer, and gradient checkpointing.

Image Generation: Prompt-based pipeline for producing accurate architectural layouts with features like parking, gardens, and pools.

Production-Ready Design: Modular architecture with reproducible environment setup, checkpoint saving, and deployment-ready generation scripts.

Results

Verified and processed 2,450+ floor plan samples without corruption.

Achieved stable convergence by epoch 15 during 18-hour fine-tuning.

Generated high-fidelity architectural plans with coherent layouts and realistic proportions.

Showcased business value in architectural visualization, design iteration, and client presentations.
