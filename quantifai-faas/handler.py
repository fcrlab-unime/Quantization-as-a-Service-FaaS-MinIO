import os
import json
import torch
import logging
from torch.utils.data import DataLoader
from minio import Minio
from minio.error import S3Error
from urllib.parse import urljoin
# Assumendo che questi moduli siano disponibili nello stesso package/ambiente
from .genericDataset import GenericDataset 
from .quantUtils import evaluate_metrics, quantize_model_fx, quantize_model_dynamic 
from .utils import load_class_from_file, parse_dynamic_args 

# ============================================================
# CONFIGURATION
# ============================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "172.16.6.62:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_SECURE = os.getenv("MINIO_SECURE", "0") == "1"

client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=MINIO_SECURE
)

# Fixed MinIO buckets
BUCKET_MODELS = "models"
BUCKET_WEIGHTS = "weights"
BUCKET_DATASETS = "datasets"
BUCKET_OUTPUT = "quantized" # Il bucket dove vengono salvati i modelli e il record delle metriche

# ============================================================
# ENTRY POINT
# ============================================================
def handle(event, context):
    """OpenFaaS entry point for quantization and evaluation using MinIO storage."""
    try:
        body = event.get("body", "") if isinstance(event, dict) else getattr(event, "body", "")
        if not body:
            raise ValueError("Empty request body")

        req = json.loads(body)
        result = execute(req)
        return {"statusCode": 200, "body": json.dumps(result)}

    except json.JSONDecodeError as e:
        logging.error("Invalid JSON input", exc_info=True)
        return {"statusCode": 400, "body": json.dumps({"error": "Invalid JSON", "details": str(e)})}
    except Exception as e:
        logging.error("Unhandled exception", exc_info=True)
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}


# ============================================================
# CORE EXECUTION LOGIC
# ============================================================
def execute(req: dict) -> dict:
    """
    Perform model quantization and metric evaluation using files stored in MinIO,
    and publishes a metrics summary record to MinIO.
    
    Il set di calibrazione è implicitamente il set di test fornito.
    """
    # --- Extract parameters ---
    python_path = req["python_path"]
    weights_path = req["weights_path"]
    
    # ⚠️ MODIFICA CHIAVE: Il set di calibrazione (calibration_set) 
    # è impostato uguale al set di test (test_set).
    test_set = req["test_set"]
    calibration_set = test_set 
    
    model_class = req["model_class"]
    dynamic_args = parse_dynamic_args(req.get("args", ""))
    batch_size = int(req.get("batch_size", 32))
    num_batches = int(req.get("num_batches", 1))
    static_quantization = req.get("static_quantization", "")
    dynamic_quantization = req.get("dynamic_quantization", "")
    evaluate_metrics_flag = req.get("evaluate_metrics", "False").lower() == "true"

    os.makedirs("/tmp", exist_ok=True)

    # --- Local temp paths ---
    py_local = os.path.join("/tmp", os.path.basename(python_path))
    weights_local = os.path.join("/tmp", os.path.basename(weights_path))
    # calibration_local e test_local punteranno allo stesso file scaricato
    calibration_local = os.path.join("/tmp", os.path.basename(calibration_set))
    test_local = os.path.join("/tmp", os.path.basename(test_set))

    # 1. Download all needed files from MinIO
    client.fget_object(BUCKET_MODELS, python_path, py_local)
    client.fget_object(BUCKET_WEIGHTS, weights_path, weights_local)
    
    # Scarica il dataset che funge sia da test che da calibrazione
    client.fget_object(BUCKET_DATASETS, test_set, test_local) 
    
    logging.info(f"All required files downloaded from MinIO. Dataset used for Test and Calibration: {test_set}")

    # 2. Load datasets
    # Entrambi i loader usano lo stesso file scaricato localmente
    calibration_data = torch.load(calibration_local)
    test_data = torch.load(test_local)
    shape = calibration_data["data"][0].shape

    calibration_ds = GenericDataset(calibration_local, shape)
    test_ds = GenericDataset(test_local, shape)
    calibration_loader = DataLoader(calibration_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # 3. Load model
    model = load_class_from_file(model_class, os.path.dirname(py_local), py_local, **dynamic_args)
    model.load_state_dict(torch.load(weights_local, map_location=DEVICE))
    model = model.to(DEVICE).eval()
    logging.info("Model loaded and ready for quantization.")

    # 4. Evaluate unquantized model (optional)
    raw_metrics = None
    if evaluate_metrics_flag:
        raw_metrics = evaluate_metrics(test_loader, model)
        raw_metrics["model_size_mb"] = os.path.getsize(weights_local) / (1024 * 1024)

    # 5. Quantization phase
    quantized_models = {"static_quantization": {}, "dynamic_quantization": {}}

    static_qtypes = [x.strip() for x in static_quantization.split(",") if x.strip()]
    dynamic_qtypes = [x.strip() for x in dynamic_quantization.split(",") if x.strip()]

    if dynamic_qtypes:
        for dtype in dynamic_qtypes:
            dtype_torch = torch.qint8 if dtype == "int8" else torch.float16
            quant_model = quantize_model_dynamic(model, calibration_loader, num_batches, type=dtype_torch)
            quantized_models["dynamic_quantization"][dtype] = quant_model
            logging.info(f"Dynamic quantization done ({dtype}).")

    if static_qtypes:
        for dtype in static_qtypes:
            dtype_torch = torch.qint8 if dtype == "int8" else torch.float16
            quant_model = quantize_model_fx(model, calibration_loader, num_batches, type=dtype_torch)
            quantized_models["static_quantization"][dtype] = quant_model
            logging.info(f"Static quantization done ({dtype}).")

    # 6. Save quantized models
    download_urls = {}
    scheme = "https" if MINIO_SECURE else "http" # Definisci 'scheme' prima dell'uso
    for qtype, models_dict in quantized_models.items():
        for dtype, qmodel in models_dict.items():
            # Percorso del modello quantizzato
            qpath = f"/tmp/{model_class}_{qtype}_{dtype}.pth"
            torch.save(qmodel.state_dict(), qpath)
            
            # Chiave di output MinIO
            output_key = f"{model_class}_{qtype}_{dtype}.pth"
            if not client.bucket_exists(BUCKET_OUTPUT):
                client.make_bucket(BUCKET_OUTPUT)
            
            # Caricamento del modello
            client.fput_object(BUCKET_OUTPUT, output_key, qpath)
            logging.info(f"Quantized model uploaded: {BUCKET_OUTPUT}/{output_key}")
            
            # Costruzione dell'URL di download
            download_urls[f"{qtype}_{dtype}"] = urljoin(
                f"{scheme}://{MINIO_ENDPOINT}/", 
                f"{BUCKET_OUTPUT}/{output_key}"
            )

    # 7. Evaluate quantized models (optional)
    quantized_metrics = {}
    if evaluate_metrics_flag:
        for qtype, models_dict in quantized_models.items():
            quantized_metrics[qtype] = {}
            for dtype, qmodel in models_dict.items():
                metrics = evaluate_metrics(test_loader, qmodel)
                qpath = f"/tmp/{model_class}_{qtype}_{dtype}.pth"
                # Ricalcola la dimensione del file salvato
                metrics["model_size_mb"] = os.path.getsize(qpath) / (1024 * 1024)
                quantized_metrics[qtype][dtype] = metrics
                logging.info(f"Evaluation done for {qtype} ({dtype}).")

    # 8. Publish Metrics Record
    # Consolida tutte le metriche e gli URL in un unico summary
    output_summary = {
        "model_class": model_class,
        "raw_metrics": raw_metrics,
        "quantized_metrics": quantized_metrics,
        "download_urls": download_urls
    }

    # Salva il summary su un file JSON temporaneo
    summary_filename = f"{model_class}_quant_summary.json"
    summary_local_path = os.path.join("/tmp", summary_filename)
    
    with open(summary_local_path, 'w') as f:
        json.dump(output_summary, f, indent=4)
    
    # Carica il file JSON nel bucket di output
    summary_output_key = summary_filename
    if not client.bucket_exists(BUCKET_OUTPUT):
        client.make_bucket(BUCKET_OUTPUT)
    
    client.fput_object(BUCKET_OUTPUT, summary_output_key, summary_local_path)
    logging.info(f"Quantization summary uploaded: {BUCKET_OUTPUT}/{summary_output_key}")
    
    # Costruisci l'URL del summary per la risposta
    summary_url = urljoin(f"{scheme}://{MINIO_ENDPOINT}/", f"{BUCKET_OUTPUT}/{summary_output_key}")

    # 9. Construct final response
    return {
        "status": "success",
        "raw_metrics": raw_metrics,
        "quantized_metrics": quantized_metrics,
        "download_urls": download_urls,
        "summary_url": summary_url
    }