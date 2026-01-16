import os
import torch
import numpy as np
from datetime import datetime

from src.configs import Settings, STAMConfig
from src.dataset import get_data
from src.model import StamImu, StamImuStaticAdapter
from src.train import train_and_evaluate
from src.utils import (
    calculate_metrics,
    print_metrics,
    visualize_predictions_separate,
    get_test_data_for_visualization,
    set_seed
)

def main():
    print("\n" + "="*80)
    print(" STAM-IMU ".center(80))
    print("="*80)

    config = Settings()
    model_config = STAMConfig()

    set_seed(config.seed)

    print(f"\nConfiguration:")
    print(f"  Device: {config.device}")
    print(f"  Data directory: {config.root_dir}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Learning rate: {config.learning_rate}")

    print("\n" + "-"*60)
    print("Loading and preparing data...")
    print("-"*60)

    try:
        train_dataset, test_dataset, scaler_features, scaler_labels = get_data(config)
        print(f"  Data loaded successfully")
        print(f"  Training samples: {len(train_dataset)}")
        print(f"  Test samples: {len(test_dataset)}")
    except Exception as e:
        print(f"  Error loading data: {e}")
        print("\nPlease ensure your data files are in the correct format:")
        return

    print("\n" + "-"*60)
    print("Creating STAM-IMU model...")
    print("-"*60)

    model = StamImu(model_config)
    static_model = StamImuStaticAdapter(model_config)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    adapter_params = sum(p.numel() for p in model.adapter.parameters())
    static_adapter_params = sum(p.numel() for p in static_model.adapter.parameters())

    print(f"  Model created successfully")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Adapter parameters: {adapter_params:,}")
    print(f"  Static Adapter parameters: {static_adapter_params:,}")

    print("\n" + "="*60)
    print("TRAINING PHASE")
    print("="*60)

    print("STAM_IMU Training")
    start_time = datetime.now()
    model, history_adapt, y_pred_adapt, test_loader = train_and_evaluate(
        config, model, train_dataset, test_dataset
    )
    training_time = (datetime.now() - start_time).total_seconds()
    print(f"\nTraining completed in {training_time:.2f} seconds")
    print(f"  Final training loss: {history_adapt['train_loss'][-1]:.6f}")
    print(f"  Final validation loss: {history_adapt['val_loss'][-1]:.6f}")

    print("STAM_IMU(static adapter) Training")
    start_time = datetime.now()
    static_model, history_no_adapt, y_pred_no_adapt, test_loader = train_and_evaluate(
        config, static_model, train_dataset, test_dataset
    )
    training_time = (datetime.now() - start_time).total_seconds()
    print(f"\nTraining completed in {training_time:.2f} seconds")
    print(f"  Final training loss: {history_no_adapt['train_loss'][-1]:.6f}")
    print(f"  Final validation loss: {history_no_adapt['val_loss'][-1]:.6f}")

    X_test, y_test = get_test_data_for_visualization(test_loader)

    print("\n" + "="*60)
    print("EVALUATION WITHOUT ADAPTATION")
    print("="*60)

    metrics_no_adapt = calculate_metrics(y_test, y_pred_no_adapt, scaler_labels)
    print_metrics(metrics_no_adapt, "Performance WITHOUT Self-Adaptation")

    print("\n" + "="*60)
    print("EVALUATION WITH ADAPTATION")
    print("="*60)

    metrics_adapt = calculate_metrics(y_test, y_pred_adapt, scaler_labels)
    print_metrics(metrics_adapt, "Performance WITH Self-Adaptation")

    print("\n" + "="*60)
    print("PERFORMANCE IMPROVEMENT")
    print("="*60)

    mse_improvement = (metrics_no_adapt['overall']['mse'] - metrics_adapt['overall']['mse']) / metrics_no_adapt['overall']['mse'] * 100
    mae_improvement = (metrics_no_adapt['overall']['mae'] - metrics_adapt['overall']['mae']) / metrics_no_adapt['overall']['mae'] * 100
    r2_improvement = (metrics_adapt['overall']['r2'] - metrics_no_adapt['overall']['r2']) / (1 - metrics_no_adapt['overall']['r2']) * 100

    print(f"\nOverall Improvements with Adaptation:")
    print(f"  MSE reduction: {mse_improvement:.2f}%")
    print(f"  MAE reduction: {mae_improvement:.2f}%")
    print(f"  R² improvement: {r2_improvement:.2f}%")

    print(f"\nPer-Dimension Improvements:")
    print(f"{'Dimension':<12} {'MSE Reduction':<15} {'MAE Reduction':<15} {'R² Improvement':<15}")
    print('-'*57)

    for dim_name in metrics_no_adapt['per_dimension'].keys():
        mse_imp = (metrics_no_adapt['per_dimension'][dim_name]['mse'] -
                   metrics_adapt['per_dimension'][dim_name]['mse']) / metrics_no_adapt['per_dimension'][dim_name]['mse'] * 100
        mae_imp = (metrics_no_adapt['per_dimension'][dim_name]['mae'] -
                   metrics_adapt['per_dimension'][dim_name]['mae']) / metrics_no_adapt['per_dimension'][dim_name]['mae'] * 100
        r2_imp = metrics_adapt['per_dimension'][dim_name]['r2'] - metrics_no_adapt['per_dimension'][dim_name]['r2']

        print(f"{dim_name:<12} {mse_imp:<15.2f}% {mae_imp:<15.2f}% {r2_imp:<15.4f}")

    print("\n" + "="*60)
    print("VISUALIZATION")
    print("="*60)

    num_points = 300
    if getattr(config, "viz_start_idx", None) is not None:
        start_idx = max(0, int(config.viz_start_idx))
        num_points = min(num_points, len(y_test) - start_idx)
    else:
        start_idx = max(0, len(y_test) - num_points)

    print(f"\nGenerating separate visualizations for the last {num_points} samples "
          f"(from {start_idx} to {start_idx + num_points})")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = f'result/stam_imu_{timestamp}'
    os.makedirs(result_dir, exist_ok=True)

    save_paths = visualize_predictions_separate(
        X_test, y_test, y_pred_no_adapt, y_pred_adapt,
        scaler_features, scaler_labels,
        start_idx=start_idx,
        num_samples=num_points,
        save_dir=result_dir
    )

    print(f"\n  All visualization figures saved to: {result_dir}")
    print(f"  - Before adaptation: {os.path.basename(save_paths[0])}")
    print(f"  - After adaptation: {os.path.basename(save_paths[1])}")
    print(f"  - Error comparison: {os.path.basename(save_paths[2])}")

    print("\n" + "="*60)
    print(f"METRICS FOR VISUALIZATION SEGMENT (Samples {start_idx}-{start_idx + getattr(config, 'viz_samples', num_points)})")
    print("="*60)

    viz_end = min(start_idx + getattr(config, "viz_samples", num_points), len(y_test))
    y_test_viz = y_test[start_idx:viz_end]
    y_pred_no_adapt_viz = y_pred_no_adapt[start_idx:viz_end]
    y_pred_adapt_viz = y_pred_adapt[start_idx:viz_end]

    metrics_viz_no_adapt = calculate_metrics(y_test_viz, y_pred_no_adapt_viz, scaler_labels)
    metrics_viz_adapt = calculate_metrics(y_test_viz, y_pred_adapt_viz, scaler_labels)

    print("\n>>> Without Adaptation:")
    print(f"  MSE: {metrics_viz_no_adapt['overall']['mse']:.6f}")
    print(f"  MAE: {metrics_viz_no_adapt['overall']['mae']:.6f}")
    print(f"  R²:  {metrics_viz_no_adapt['overall']['r2']:.6f}")

    print("\n>>> With Adaptation:")
    print(f"  MSE: {metrics_viz_adapt['overall']['mse']:.6f}")
    print(f"  MAE: {metrics_viz_adapt['overall']['mae']:.6f}")
    print(f"  R²:  {metrics_viz_adapt['overall']['r2']:.6f}")

    metrics_file = os.path.join(result_dir, 'evaluation_metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write("STAM-IMU Evaluation Results\n")
        f.write("="*50 + "\n\n")
        f.write(f"Visualization Segment: Samples {start_idx}-{viz_end}\n")
        f.write(f"Total Test Samples: {len(y_test)}\n\n")

        f.write("OVERALL METRICS\n")
        f.write("-"*30 + "\n")
        f.write("Without Adaptation:\n")
        f.write(f"  MSE: {metrics_no_adapt['overall']['mse']:.6f}\n")
        f.write(f"  MAE: {metrics_no_adapt['overall']['mae']:.6f}\n")
        f.write(f"  R²:  {metrics_no_adapt['overall']['r2']:.6f}\n\n")

        f.write("With Adaptation:\n")
        f.write(f"  MSE: {metrics_adapt['overall']['mse']:.6f}\n")
        f.write(f"  MAE: {metrics_adapt['overall']['mae']:.6f}\n")
        f.write(f"  R²:  {metrics_adapt['overall']['r2']:.6f}\n\n")

        f.write("IMPROVEMENTS\n")
        f.write("-"*30 + "\n")
        f.write(f"MSE reduction: {mse_improvement:.2f}%\n")
        f.write(f"MAE reduction: {mae_improvement:.2f}%\n")
        f.write(f"R² improvement: {r2_improvement:.2f}%\n\n")

        f.write("PER-DIMENSION METRICS\n")
        f.write("-"*30 + "\n")
        for dim_name in metrics_no_adapt['per_dimension'].keys():
            f.write(f"\n{dim_name}:\n")
            f.write(f"  Before - MSE: {metrics_no_adapt['per_dimension'][dim_name]['mse']:.6f}, ")
            f.write(f"MAE: {metrics_no_adapt['per_dimension'][dim_name]['mae']:.6f}, ")
            f.write(f"R²: {metrics_no_adapt['per_dimension'][dim_name]['r2']:.6f}\n")
            f.write(f"  After  - MSE: {metrics_adapt['per_dimension'][dim_name]['mse']:.6f}, ")
            f.write(f"MAE: {metrics_adapt['per_dimension'][dim_name]['mae']:.6f}, ")
            f.write(f"R²: {metrics_adapt['per_dimension'][dim_name]['r2']:.6f}\n")

    print(f"\n  Detailed metrics saved to: {metrics_file}")

    if getattr(config, "save_model", False):
        model_path = os.path.join(result_dir, 'stam_imu_model.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'model_config': model_config,
            'history_adapt': history_adapt,
            'history_no_adapt': history_no_adapt,
            'metrics_no_adapt': metrics_no_adapt,
            'metrics_adapt': metrics_adapt,
            'scaler_features': scaler_features,
            'scaler_labels': scaler_labels
        }, model_path)
        print(f"\n  Model saved to {model_path}")

    print("\n" + "="*80)
    print(" STAM-IMU Training and Evaluation Complete ".center(80))
    print("="*80)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    except Exception as e:
        print(f"\n\nError occurred: {e}")
        import traceback
        traceback.print_exc()
