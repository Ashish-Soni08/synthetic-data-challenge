#!/usr/bin/env python3
"""
Medium Model Training Script for MOSTLY_AI

Usage: modal run medium.py
"""

# Import from your main training script
from train_model import app, train_model

@app.local_entrypoint()
def main():
    """Medium model training with batch size 512."""
    
    # Medium model configuration
    train_model.remote(
        max_training_time=360,          
        max_epochs=200,                 
        batch_size=128,             
        model_name="MOSTLY_AI/Medium",
        max_sample_size=100000,         
        synthetic_dataset_size=100000,         
        gradient_accumulation_steps=1,
        max_sequence_window=100,
        enable_flexible_generation=False,
        value_protection=True,
        rare_category_replacement_method="SAMPLE",
        enable_model_report=True         
    )

if __name__ == "__main__":
    main()
