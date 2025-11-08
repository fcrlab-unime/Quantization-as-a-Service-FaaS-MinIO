import torch
import torch.nn as nn
from torch.quantization import quantize_fx, QConfig, MinMaxObserver, PerChannelMinMaxObserver, quantize_dynamic
from torch.ao.quantization.qconfig_mapping import QConfigMapping

# Metrics Calculation
def evaluate_metrics(test_dataloader, model):
    """Evaluate all metrics in one pass."""

    is_classification = True if determine_problem_type_from_dataloader(test_dataloader) == "classification" else False

    model.eval()
    total, correct, total_loss, predictions, true_labels = 0, 0, 0.0, [], []
    loss_fn = nn.CrossEntropyLoss() if is_classification else nn.MSELoss()

    with torch.no_grad():
        for data, labels in test_dataloader:
            outputs = model(data)
            if is_classification:
                total_loss += loss_fn(outputs, labels).item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
            else:
                total_loss += loss_fn(outputs.squeeze(), labels).item()
                predictions.append(outputs.squeeze())
                true_labels.append(labels)

            total += labels.size(0)

    results = {"loss": total_loss / total}
    if is_classification:
        results["accuracy"] = correct / total
    else:
        predictions = torch.cat(predictions)
        true_labels = torch.cat(true_labels)
        ss_residual = torch.sum((true_labels - predictions) ** 2).item()
        ss_total = torch.sum((true_labels - torch.mean(true_labels)) ** 2).item()
        results.update({
            "mse": total_loss / total,
            "r2": 1 - (ss_residual / ss_total),
            "mae": torch.mean(torch.abs(true_labels - predictions)).item()
        })
    return results




# Quantization
def quantize_model_fx(model, calibration_loader, num_batches=1, type=torch.qint8):
    """Quantize a model using FX Graph Mode."""
    try:

        model_to_quantize = model
        model_to_quantize.eval()
        # Set the quantization configuration
        qconfig = QConfig(
            activation=MinMaxObserver.with_args(dtype=torch.quint8 if type == torch.qint8 else torch.float16),
            weight=PerChannelMinMaxObserver.with_args(dtype=type)
        )
        model_to_quantize.qconfig = qconfig
        
        # Get example input from the calibration dataloader
        example_inputs = next(iter(calibration_loader))[0]
        print("Example input shape:", example_inputs.shape)  # Debugging

        # Prepare the model for quantization
        prepared_model = quantize_fx.prepare_fx(model_to_quantize, QConfigMapping().set_global(torch.quantization.default_qconfig), example_inputs)

        # Forward pass for a few batches
        for i, (batch_data, _) in enumerate(calibration_loader):
            print(f"Processing batch {i + 1}...")  # Debugging
            prepared_model(batch_data)
            if i >= num_batches - 1:
                break

        # Convert the model after quantization
        quantized_model = quantize_fx.convert_fx(prepared_model)
        print("Quantization complete.")  # Debugging
        
        return quantized_model
    except Exception as e:
        print("Error during quantization:", e)
        raise e  # Re-raise the exception for further handling

def quantize_model_dynamic(model, calibration_loader, num_batches=1, type=torch.qint8):
    """Quantize a model using Dynamic Quantization without modifying the model in place."""

    # Set the quantization engine
    #torch.backends.quantized.engine = 'qnnpack'

    # Set the model to evaluation mode before quantization
  

    # Create a new model by copying the original model
    quantized_model = model
    quantized_model.eval()
    # Apply dynamic quantization to the specified layers (only Linear layers in this case)
    quantized_model = quantize_dynamic(
        quantized_model,  # The model to quantize
        {torch.nn.Linear},  # Layers to quantize dynamically
        dtype=type  # Type of quantization (e.g., torch.qint8)
    )

    # Perform a pass through the model for calibration (no need for explicit calibration in dynamic quantization)
    for i, (batch_data, _) in enumerate(calibration_loader):
        # Since we are only interested in updating the internal statistics, we don't need the model's output
        quantized_model(batch_data)
        if i >= num_batches - 1:
            break

    return quantized_model

def determine_problem_type_from_dataloader(dataloader):
    """
    Determines if the dataset in a DataLoader is for a regression or classification problem
    based on the label values.

    Args:
        dataloader (DataLoader): The PyTorch DataLoader containing the dataset.

    Returns:
        str: 'regression' or 'classification'
    """
    labels = []

    # Collect all labels from the DataLoader
    for _, label in dataloader:
        labels.extend(label.cpu().numpy())

    labels = torch.tensor(labels)
    unique_labels = labels.unique()

    if labels.dtype in [torch.float32, torch.float64]:
        # Threshold for determining regression vs classification (arbitrary: 20 unique values)
        if len(unique_labels) > 20:
            return "regression"
        else:
            return "classification"
    else:
        return "classification"
