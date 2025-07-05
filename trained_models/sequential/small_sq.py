from train_sequential_model import app, train_sequential_model

@app.local_entrypoint()
def main():
    """Small model training with batch size 32 for Testing."""
    
    # Small model configuration
    train_sequential_model.remote(
        max_training_time=60,          
        max_epochs=200,                 
        batch_size=128,             
        model_name="MOSTLY_AI/Small",
        max_sample_size=None,
        synthetic_dataset_size=20000,
        gradient_accumulation_steps=1,
        max_sequence_window=10,
        enable_flexible_generation=False,
        value_protection=True,
        rare_category_replacement_method="SAMPLE",
        enable_model_report=True        
    )

if __name__ == "__main__":
    main()
