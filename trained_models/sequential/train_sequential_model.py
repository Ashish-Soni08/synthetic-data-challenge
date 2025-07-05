#!/usr/bin/env python3
"""
Train MOSTLY_AI Models for Sequential Data Challenge

Usage: modal run (script name).py
"""

import modal
import time

# Modal Image Configuration
mostlyai_gpu_image = (
    modal.Image.debian_slim(python_version="3.12.10")
    .pip_install("uv", gpu="A10G")
    .run_commands("uv pip install --system --compile-bytecode scipy==1.13.1 'mostlyai[local-gpu]'")
    .add_local_dir("./00_original_datasets", remote_path="/root/data")
)

# Modal App and Volume Setup
app = modal.App("MostlyAI-Sequential-Data-Challenge")
volume = modal.Volume.from_name("mostlyai-challenge-volume", create_if_missing=True)

@app.function(image=mostlyai_gpu_image, gpu="A10G", volumes={"/vol": volume}, timeout=43200)
def train_sequential_model(
    data_path: str = "/root/data/sequential-training.csv",
    max_training_time: int = 360,
    max_epochs: int = 200,
    batch_size: int = 256,
    model_name: str = "MOSTLY_AI/Small",
    max_sample_size: int = None,
    synthetic_dataset_size: int = 10,
    gradient_accumulation_steps: int = 1,
    max_sequence_window: int = 10,
    enable_flexible_generation: bool = False,
    value_protection: bool = True,
    rare_category_replacement_method: str = "SAMPLE",
    enable_model_report: bool = True
):
    import pandas as pd
    from pathlib import Path
    from mostlyai.sdk import MostlyAI

    mostly = MostlyAI(local=True)

    models_dir = Path("/vol/models")
    models_dir.mkdir(exist_ok=True)
    generated_datasets_dir = Path("/vol/generated_datasets")
    generated_datasets_dir.mkdir(exist_ok=True)

    model_size = model_name.replace("MOSTLY_AI/", "")

    # Load data
    print(f"📊 Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"✅ Shape: {df.shape}")

    # Model config
    model_config = {
        'model': model_name,
        'max_sample_size': max_sample_size,
        'batch_size': batch_size,
        'gradient_accumulation_steps': gradient_accumulation_steps,
        'max_training_time': max_training_time,
        'max_epochs': max_epochs,
        'max_sequence_window': max_sequence_window,
        'enable_flexible_generation': enable_flexible_generation,
        'value_protection': value_protection,
        'enable_model_report': enable_model_report,
        'rare_category_replacement_method': rare_category_replacement_method,
    }

    
    # Generator config (single-table sequential)
    generator_config = {
    'name': f'Sequential Data Challenge - {model_size}-{batch_size}',
    'tables': [
        {
            'name': 'groups',
            'data': df[['group_id']].drop_duplicates(),
            'primary_key': 'group_id',
        },
        {
            'name': 'events',
            'data': df,
            'foreign_keys': [
                {
                    'column': 'group_id',
                    'referenced_table': 'groups',
                    'is_context': True,
                }
            ],
            'columns': [
                {'name': 'group_id', 'model_encoding_type': 'TABULAR_CATEGORICAL'},
                {'name': 'alice', 'model_encoding_type': 'TABULAR_CATEGORICAL'},
                {'name': 'david', 'model_encoding_type': 'TABULAR_NUMERIC_AUTO'},
                {'name': 'emily', 'model_encoding_type': 'TABULAR_CATEGORICAL'},
                {'name': 'jacob', 'model_encoding_type': 'TABULAR_NUMERIC_AUTO'},
                {'name': 'james', 'model_encoding_type': 'TABULAR_NUMERIC_AUTO'},
                {'name': 'john', 'model_encoding_type': 'TABULAR_CATEGORICAL'},
                {'name': 'mike', 'model_encoding_type': 'TABULAR_NUMERIC_DISCRETE'},
                {'name': 'lucas', 'model_encoding_type': 'TABULAR_NUMERIC_AUTO'},
                {'name': 'mary', 'model_encoding_type': 'TABULAR_NUMERIC_AUTO'},
                {'name': 'sarah', 'model_encoding_type': 'TABULAR_NUMERIC_AUTO'},
            ],
            'tabular_model_configuration': model_config
        }
    ]
    }

    print(f"⚙️ Training Configuration:")
    print(f"   Model: {model_name}")
    print(f"   Batch size: {batch_size}")
    print(f"   Max training time: {max_training_time} minutes")
    print(f"   Max epochs: {max_epochs}")
    print(f"   Sample size: {max_sample_size}")
    print(f"   Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        print(f"🔄 Starting training...")
        g = mostly.train(config=generator_config, start=True, wait=True, progress_bar=True)

        if g.training_status == "DONE":
            print(f"✅ Training completed successfully!")
            print(f"🆔 Generator ID: {g.id}")

            # Model export (if supported in your environment)
            g.export_to_file(f"/vol/models/{model_size}_{batch_size}_sequential.zip")
            print(f"✅ Model saved: {model_size}_{batch_size}_sequential.zip")

            print(f"🔄 Starting data generation...")
            try:
                sd = mostly.generate(
                    generator=g,
                    size=synthetic_dataset_size,
                    start=True,
                    wait=True,
                    progress_bar=True,
                )
                synthetic_dataset = sd.data()
                synthetic_dataset['events'].to_csv(f"/vol/generated_datasets/{model_size}_{batch_size}_sequential.csv", index=False)
                print(f"✅ Synthetic datasets saved")
                generation_success = True
            except Exception as e:
                print(f"❌ Error during data generation: {e}")
                generation_success = False

            volume.commit()
            print(f"✅ Results committed to volume!")

            return {
                'success': True,
                'model_filename': f"{model_size}_{batch_size}_sequential.zip",  # Uncomment if model export is supported
                'dataset_filename': f"{model_size}_{batch_size}_sequential.csv",
                'generator_id': str(g.id),
                'completed_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'metrics': metrics_data,
                'generation_success': generation_success
            }
        else:
            print(f"❌ Training failed with status: {g.training_status}")
            return {'success': False, 'error': f'Training status: {g.training_status}'}
    except Exception as e:
        print(f"❌ Error during training: {e}")
        return {'success': False, 'error': str(e)}