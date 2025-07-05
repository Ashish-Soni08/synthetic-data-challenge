#!/usr/bin/env python3
"""
Test Model Training Script for MOSTLY_AI

Usage: modal run test_training.py
"""

# Import from your main training script
from train_model import app, train_model

@app.local_entrypoint()
def main():
    """Small model training with batch size 32 for Testing."""
    
    # Small model configuration
    train_model.remote(
        max_training_time=5,          
        max_epochs=2,                 
        batch_size=64,             
        model_name="MOSTLY_AI/Small",
        max_sample_size=1000,
        synthetic_dataset_size=100,
        gradient_accumulation_steps=1,
        max_sequence_window=100,
        enable_flexible_generation=False,
        value_protection=True,
        rare_category_replacement_method="SAMPLE",
        enable_model_report=True        
    )

if __name__ == "__main__":
    main()
