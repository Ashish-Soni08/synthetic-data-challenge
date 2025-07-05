#!/usr/bin/env python3
"""
Train MOSTLY_AI Models

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
app = modal.App("MostlyAI-Flat-Data-Challenge")
volume = modal.Volume.from_name("mostlyai-challenge-volume", create_if_missing=True)

@app.function(image=mostlyai_gpu_image, gpu="A10G", volumes={"/vol": volume}, timeout=43200)
def train_model(data_path: str = "/root/data/flat-training.csv",  # Path on Modal
                max_training_time: int = 360,  # Training time in minutes
                max_epochs: int = 200,  # Number of epochs
                batch_size: int = 1024,  # Batch size (powers of 2: 2, 4, 8, ..., 32768, 65536)
                model_name: str = "MOSTLY_AI/Small",  # Model name (Small, Medium, Large)
                max_sample_size: int = None,  # Sample size (None for full dataset)
                synthetic_dataset_size: int = 10,  # Number of samples to generate
                gradient_accumulation_steps: int = 1,  # Gradient accumulation steps
                max_sequence_window: int = 100,  # Max sequence window for TABULAR models
                enable_flexible_generation: bool = True,  # Enable flexible generation
                value_protection: bool = True,  # Enable value protection
                rare_category_replacement_method: str = "SAMPLE",  # Rare category replacement
                enable_model_report: bool = True  # Enable model report
                ):

    """Train a MOSTLY_AI Model on Modal for the FLAT DATA Challenge."""

    import pandas as pd
    from pathlib import Path
    from mostlyai.sdk import MostlyAI
    
    # Initialize MostlyAI
    mostly = MostlyAI(local=True)
    
    # Fix 1: Proper directory creation
    models_dir = Path("/vol/models")
    models_dir.mkdir(exist_ok=True)

    generated_datasets_dir = Path("/vol/generated_datasets")
    generated_datasets_dir.mkdir(exist_ok=True)

    model_size = model_name.replace("MOSTLY_AI/", "")
    
    # Load data
    print(f"📊 Loading training data from path on Modal: {data_path}")
    df = pd.read_csv(data_path)
    print(f"✅ Number of rows and columns: {df.shape}")
    
    # Training configuration
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
    
    generator_config = {
        'name': f'Flat Data Challenge - {model_size}-{batch_size}',
        'tables': [{
            'name': 'flat-dataset',
            'data': df,
            'columns': [
                {'name': 'dog', 'model_encoding_type': 'TABULAR_NUMERIC_DISCRETE'},     # 81 discrete values
                {'name': 'cat', 'model_encoding_type': 'TABULAR_CHARACTER'},           # Hex codes like A5DB
                {'name': 'rabbit', 'model_encoding_type': 'TABULAR_NUMERIC_AUTO'},     # High missing (75.98%)
                {'name': 'deer', 'model_encoding_type': 'TABULAR_NUMERIC_AUTO'},       # 300 decimal values
                {'name': 'panda', 'model_encoding_type': 'TABULAR_NUMERIC_DISCRETE'},  # Small range: -6 to 0
                {'name': 'koala', 'model_encoding_type': 'TABULAR_CATEGORICAL'},       # T0-T3 codes
                {'name': 'otter', 'model_encoding_type': 'TABULAR_CHARACTER'},         # Hex codes
                {'name': 'hedgehog', 'model_encoding_type': 'TABULAR_NUMERIC_AUTO'},   # High cardinality (729)
                {'name': 'squirrel', 'model_encoding_type': 'TABULAR_CATEGORICAL'},    # Binary 0/1
                {'name': 'dolphin', 'model_encoding_type': 'TABULAR_CATEGORICAL'},     # Binary 0/1
                {'name': 'penguin', 'model_encoding_type': 'TABULAR_CATEGORICAL'},     # Binary 10/20
                {'name': 'turtle', 'model_encoding_type': 'TABULAR_NUMERIC_DISCRETE'}, # Zero-inflated (78.83% zeros)
                {'name': 'elephant', 'model_encoding_type': 'TABULAR_NUMERIC_DISCRETE'}, # Range 1-29
                {'name': 'giraffe', 'model_encoding_type': 'TABULAR_CATEGORICAL'},     # Binary y/n
                {'name': 'lamb', 'model_encoding_type': 'TABULAR_CATEGORICAL'},        # Binary 0/5
                {'name': 'goat', 'model_encoding_type': 'TABULAR_CHARACTER'},          # Hex codes
                {'name': 'cow', 'model_encoding_type': 'TABULAR_NUMERIC_AUTO'},        # High missing (49.99%)
                {'name': 'horse', 'model_encoding_type': 'TABULAR_NUMERIC_DISCRETE'},  # Range 1-28
                {'name': 'donkey', 'model_encoding_type': 'TABULAR_CATEGORICAL'},      # X0-X8 codes
                {'name': 'pony', 'model_encoding_type': 'TABULAR_CATEGORICAL'},        # Mixed 5/M
                {'name': 'llama', 'model_encoding_type': 'TABULAR_CHARACTER'},         # Hex codes
                {'name': 'mouse', 'model_encoding_type': 'TABULAR_NUMERIC_DISCRETE'},  # 216 values
                {'name': 'hamster', 'model_encoding_type': 'TABULAR_CATEGORICAL'},     # D0-D11 codes
                {'name': 'guinea', 'model_encoding_type': 'TABULAR_NUMERIC_AUTO'},     # High cardinality (316)
                {'name': 'duck', 'model_encoding_type': 'TABULAR_CHARACTER'},          # Hex codes
                {'name': 'chicken', 'model_encoding_type': 'TABULAR_NUMERIC_AUTO'},    # 85 decimal values
                {'name': 'sparrow', 'model_encoding_type': 'TABULAR_CATEGORICAL'},     # Binary 0/120
                {'name': 'parrot', 'model_encoding_type': 'TABULAR_NUMERIC_DISCRETE'}, # Range 0-151
                {'name': 'finch', 'model_encoding_type': 'TABULAR_CATEGORICAL'},       # Binary 0/1
                {'name': 'canary', 'model_encoding_type': 'TABULAR_NUMERIC_DISCRETE'}, # Zero-inflated (88.25%)
                {'name': 'bee', 'model_encoding_type': 'TABULAR_NUMERIC_AUTO'},        # Normal distribution
                {'name': 'butterfly', 'model_encoding_type': 'TABULAR_CHARACTER'},     # Hex codes
                {'name': 'ladybug', 'model_encoding_type': 'TABULAR_NUMERIC_DISCRETE'}, # Range 0-70
                {'name': 'snail', 'model_encoding_type': 'TABULAR_NUMERIC_AUTO'},      # High cardinality (716)
                {'name': 'frog', 'model_encoding_type': 'TABULAR_NUMERIC_AUTO'},       # High cardinality (384)
                {'name': 'cricket', 'model_encoding_type': 'TABULAR_NUMERIC_DISCRETE'}, # Small range 0-5
                {'name': 'tamarin', 'model_encoding_type': 'TABULAR_CATEGORICAL'},     # Y0-Y6 codes
                {'name': 'wallaby', 'model_encoding_type': 'TABULAR_CHARACTER'},       # Hex codes
                {'name': 'wombat', 'model_encoding_type': 'TABULAR_NUMERIC_DISCRETE'}, # Range 3-260
                {'name': 'zebra', 'model_encoding_type': 'TABULAR_CATEGORICAL'},       # Binary 0/1
                {'name': 'flamingo', 'model_encoding_type': 'TABULAR_NUMERIC_AUTO'},   # Right-skewed decimals
                {'name': 'peacock', 'model_encoding_type': 'TABULAR_CATEGORICAL'},     # Binary 0/1
                {'name': 'bat', 'model_encoding_type': 'TABULAR_CATEGORICAL'},         # Binary 0/1
                {'name': 'fox', 'model_encoding_type': 'TABULAR_NUMERIC_DISCRETE'},    # Concentrated on value 5
                {'name': 'beaver', 'model_encoding_type': 'TABULAR_NUMERIC_DISCRETE'}, # Small range -2 to 1
                {'name': 'monkey', 'model_encoding_type': 'TABULAR_CATEGORICAL'},      # Z0-Z3 codes
                {'name': 'seal', 'model_encoding_type': 'TABULAR_NUMERIC_AUTO'},       # Decimal values
                {'name': 'robin', 'model_encoding_type': 'TABULAR_CATEGORICAL'},       # Binary 0/1
                {'name': 'loon', 'model_encoding_type': 'TABULAR_NUMERIC_AUTO'},       # High missing (49.88%)
                {'name': 'swan', 'model_encoding_type': 'TABULAR_NUMERIC_DISCRETE'},   # Small range 2-4
                {'name': 'goldfish', 'model_encoding_type': 'TABULAR_NUMERIC_DISCRETE'}, # Range 6-213
                {'name': 'minnow', 'model_encoding_type': 'TABULAR_CATEGORICAL'},      # Binary 0/1
                {'name': 'mole', 'model_encoding_type': 'TABULAR_CATEGORICAL'},        # Binary 0/1
                {'name': 'shrew', 'model_encoding_type': 'TABULAR_NUMERIC_AUTO'},      # Highly skewed, high cardinality
                {'name': 'puffin', 'model_encoding_type': 'TABULAR_NUMERIC_DISCRETE'}, # Zero-inflated (90.27%)
                {'name': 'owl', 'model_encoding_type': 'TABULAR_CATEGORICAL'},         # Binary 0/1
                {'name': 'bunny', 'model_encoding_type': 'TABULAR_NUMERIC_DISCRETE'},  # Small range 2-4
                {'name': 'bear', 'model_encoding_type': 'TABULAR_NUMERIC_AUTO'},       # Decimal range 1.4-2.4
                {'name': 'chipmunk', 'model_encoding_type': 'TABULAR_CHARACTER'},      # Hex codes
                {'name': 'cub', 'model_encoding_type': 'TABULAR_NUMERIC_AUTO'},        # High cardinality (651)
                {'name': 'acorn', 'model_encoding_type': 'TABULAR_NUMERIC_DISCRETE'},  # Range 2-164
                {'name': 'leaf', 'model_encoding_type': 'TABULAR_CHARACTER'},          # Hex codes
                {'name': 'cloud', 'model_encoding_type': 'TABULAR_NUMERIC_AUTO'},      # High missing (79.40%)
                {'name': 'rainbow', 'model_encoding_type': 'TABULAR_NUMERIC_AUTO'},    # Decimal range 0.6-4.5
                {'name': 'puddle', 'model_encoding_type': 'TABULAR_NUMERIC_DISCRETE'}, # Small range 1-3
                {'name': 'berry', 'model_encoding_type': 'TABULAR_NUMERIC_AUTO'},      # High cardinality (363)
                {'name': 'apple', 'model_encoding_type': 'TABULAR_NUMERIC_DISCRETE'},  # Negative values
                {'name': 'honey', 'model_encoding_type': 'TABULAR_NUMERIC_DISCRETE'},  # Small range 3-6
                {'name': 'pumpkin', 'model_encoding_type': 'TABULAR_NUMERIC_DISCRETE'}, # Negative values
                {'name': 'teddy', 'model_encoding_type': 'TABULAR_CHARACTER'},         # Hex codes
                {'name': 'blanket', 'model_encoding_type': 'TABULAR_CATEGORICAL'},     # A0-A9 codes
                {'name': 'button', 'model_encoding_type': 'TABULAR_NUMERIC_DISCRETE'}, # Negative values
                {'name': 'whistle', 'model_encoding_type': 'TABULAR_NUMERIC_DISCRETE'}, # Range -2 to 14
                {'name': 'marble', 'model_encoding_type': 'TABULAR_CATEGORICAL'},      # Binary 0/1
                {'name': 'wagon', 'model_encoding_type': 'TABULAR_NUMERIC_AUTO'},      # Negative decimals
                {'name': 'storybook', 'model_encoding_type': 'TABULAR_NUMERIC_DISCRETE'}, # Negative values
                {'name': 'candle', 'model_encoding_type': 'TABULAR_CATEGORICAL'},      # B0-B4 codes
                {'name': 'clover', 'model_encoding_type': 'TABULAR_NUMERIC_AUTO'},     # High cardinality (547)
                {'name': 'bubble', 'model_encoding_type': 'TABULAR_CATEGORICAL'},      # Binary 0/1
                {'name': 'cookie', 'model_encoding_type': 'TABULAR_CATEGORICAL'}       # C0-C19 codes
            ],
            'tabular_model_configuration': model_config
        }]
    }
    
    print(f"⚙️ Training Configuration:")
    print(f"   Model: {model_name}")
    print(f"   Batch size: {batch_size}")
    print(f"   Max training time: {max_training_time} minutes")
    print(f"   Max epochs: {max_epochs}")
    print(f"   Sample size: {max_sample_size}")
    print(f"   Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Train the model
        print(f"🔄 Starting training...")
        g = mostly.train(config=generator_config, start=True, wait=True, progress_bar=True)
        
        if g.training_status == "DONE":
            print(f"✅ Training completed successfully!")
            print(f"🆔 Generator ID: {g.id}")
            
            # Fix 2: Save model with clean naming
            g.export_to_file(f"/vol/models/{model_size}_{batch_size}.zip")
            print(f"✅ Model saved: {model_size}_{batch_size}.zip")
            
            # Fix 2: Wait 4 minutes for metrics computation
            print(f"⏱️ Waiting 4 minutes for metrics computation...")
            time.sleep(240)
            
            # Fix 2: Get metrics after wait
            print(f"🔍 Fetching metrics after 4-minute wait...")
            try:
                metrics = g.tables[0].tabular_model_metrics
                
                if metrics:
                    print(f"✅ METRICS FOUND!")
                    
                    # Extract key metrics
                    accuracy = metrics.accuracy.overall if metrics.accuracy else None
                    dcr_share = metrics.distances.dcr_share if metrics.distances else None
                    nndr_training = metrics.distances.nndr_training if metrics.distances else None
                    
                    # Fix 2: Qualification check
                    qualified = False
                    if dcr_share is not None and nndr_training is not None:
                        dcr_check = dcr_share < 0.52
                        nndr_check = nndr_training > 0.5
                        qualified = dcr_check and nndr_check
                    
                    print(f"📊 Metrics Results:")
                    print(f"   Accuracy: {accuracy}")
                    print(f"   DCR Share: {dcr_share}")
                    print(f"   NNDR Training: {nndr_training}")
                    print(f"🎯 Qualification Check:")
                    print(f"   DCR Share < 0.52: {'✅' if dcr_share and dcr_share < 0.52 else '❌'}")
                    print(f"   NNDR Training > 0.5: {'✅' if nndr_training and nndr_training > 0.5 else '❌'}")
                    print(f"   🏆 QUALIFIED: {'✅ YES' if qualified else '❌ NO'}")
                    
                    metrics_data = {
                        'accuracy': accuracy,
                        'dcr_share': dcr_share,
                        'nndr_training': nndr_training,
                        'qualified': qualified
                    }
                else:
                    print(f"❌ No metrics available after wait")
                    metrics_data = {'error': 'No metrics available'}
                    
            except Exception as metrics_error:
                print(f"❌ Error fetching metrics: {metrics_error}")
                metrics_data = {'error': str(metrics_error)}
            
            # # Fix 4: Data generation inside main try-except
            # print(f"🔄 Starting data generation...")
            # try:
            #     sd = mostly.generate(
            #         generator=g,
            #         size=synthetic_dataset_size,
            #         start=True,
            #         wait=True,
            #         progress_bar=True,
            #         config={
            #             'tables': [{
            #                 'name': "flat-dataset",
            #                 "configuration": {
            #                     "imputation": {
            #                         "columns": ["cloud", "cow", "loon", "rabbit"]
            #                     }
            #                 }
            #             }]
            #         }
            #     )
            #     synthetic_dataset = sd.data()
                
            #     # Save synthetic dataset
            #     synthetic_dataset.to_csv(f"/vol/generated_datasets/{model_size}_{batch_size}_synthetic_dataset.csv", index=False)
            #     print(f"✅ Synthetic dataset saved")
                
            #     generation_success = True
                
            # except Exception as e:
            #     print(f"❌ Error during data generation: {e}")
            #     generation_success = False
            
            # Commit to volume
            volume.commit()
            print(f"✅ Results committed to volume!")
            
            return {
                'success': True,
                'model_filename': f"{model_size}_{batch_size}.zip",
                #'dataset_filename': f"{model_size}_{batch_size}_synthetic_dataset.csv",
                'generator_id': str(g.id),
                'completed_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'metrics': metrics_data,
                #'generation_success': generation_success
            }
            
        else:
            print(f"❌ Training failed with status: {g.training_status}")
            return {'success': False, 'error': f'Training status: {g.training_status}'}
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        return {'success': False, 'error': str(e)}
