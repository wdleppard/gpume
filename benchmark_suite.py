# benchmark_suite.py
import time
import torch
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Any
import psutil
import GPUtil
from transformers import pipeline
import cv2
import pytesseract
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import json
import logging

@dataclass
class BenchmarkConfig:
    workload_type: str  # 'ocr', 'nlp', 'hybrid', 'forms'
    num_documents: int = 1000
    pages_per_doc: int = 5
    batch_size: int = 8
    num_workers: int = 4
    use_fp16: bool = True
    use_multiprocessing: bool = True

@dataclass
class BenchmarkResults:
    total_documents: int
    total_pages: int
    total_time: float
    docs_per_second: float
    pages_per_second: float
    avg_gpu_utilization: float
    avg_gpu_memory: float
    avg_cpu_utilization: float
    avg_system_memory: float
    errors: List[str]
    
class DocumentExtractionBenchmark:
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.metrics = []
        
    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def _monitor_resources(self):
        """Collect system resource metrics"""
        gpus = GPUtil.getGPUs()
        
        metrics = {
            'timestamp': time.time(),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'gpu_utilization': [gpu.load * 100 for gpu in gpus],
            'gpu_memory': [gpu.memoryUtil * 100 for gpu in gpus],
            'gpu_temperature': [gpu.temperature for gpu in gpus]
        }
        
        return metrics
    
    def run_ocr_benchmark(self):
        """Benchmark OCR-based extraction"""
        self.logger.info("Starting OCR benchmark...")
        
        # Simulate document images
        dummy_images = [
            np.random.randint(0, 255, (2480, 3508, 3), dtype=np.uint8)
            for _ in range(self.config.batch_size)
        ]
        
        start_time = time.time()
        processed_pages = 0
        
        for doc_idx in range(self.config.num_documents):
            for page_idx in range(self.config.pages_per_doc):
                # Preprocess image
                img = dummy_images[page_idx % self.config.batch_size]
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # OCR extraction (simulate with sleep for now)
                # In production, use: pytesseract.image_to_string(gray)
                time.sleep(0.01)  # Simulate OCR processing
                
                processed_pages += 1
                
                # Monitor resources periodically
                if processed_pages % 100 == 0:
                    self.metrics.append(self._monitor_resources())
        
        total_time = time.time() - start_time
        
        return BenchmarkResults(
            total_documents=self.config.num_documents,
            total_pages=processed_pages,
            total_time=total_time,
            docs_per_second=self.config.num_documents / total_time,
            pages_per_second=processed_pages / total_time,
            avg_gpu_utilization=np.mean([m['gpu_utilization'][0] for m in self.metrics]),
            avg_gpu_memory=np.mean([m['gpu_memory'][0] for m in self.metrics]),
            avg_cpu_utilization=np.mean([m['cpu_percent'] for m in self.metrics]),
            avg_system_memory=np.mean([m['memory_percent'] for m in self.metrics]),
            errors=[]
        )
    
    def run_nlp_benchmark(self):
        """Benchmark NLP-based extraction"""
        self.logger.info("Starting NLP benchmark...")
        
        # Initialize NLP pipeline
        nlp_pipeline = pipeline(
            "ner",
            model="dslim/bert-base-NER",
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Sample texts
        sample_texts = [
            "John Doe works at Acme Corp in New York. The invoice amount is $1,234.56."
            for _ in range(self.config.batch_size)
        ]
        
        start_time = time.time()
        processed_pages = 0
        
        for doc_idx in range(self.config.num_documents):
            for page_idx in range(self.config.pages_per_doc):
                # Process batch
                results = nlp_pipeline(sample_texts, batch_size=self.config.batch_size)
                processed_pages += len(sample_texts)
                
                # Monitor resources
                if processed_pages % 100 == 0:
                    self.metrics.append(self._monitor_resources())
        
        total_time = time.time() - start_time
        
        return BenchmarkResults(
            total_documents=self.config.num_documents,
            total_pages=processed_pages,
            total_time=total_time,
            docs_per_second=self.config.num_documents / total_time,
            pages_per_second=processed_pages / total_time,
            avg_gpu_utilization=np.mean([m['gpu_utilization'][0] for m in self.metrics]),
            avg_gpu_memory=np.mean([m['gpu_memory'][0] for m in self.metrics]),
            avg_cpu_utilization=np.mean([m['cpu_percent'] for m in self.metrics]),
            avg_system_memory=np.mean([m['memory_percent'] for m in self.metrics]),
            errors=[]
        )
    
    def run_hybrid_benchmark(self):
        """Benchmark hybrid CV + NLP extraction"""
        self.logger.info("Starting hybrid benchmark...")
        
        # This would combine both OCR and NLP pipelines
        # Implementation depends on specific use case
        
        # For now, simulate with combined processing time
        ocr_results = self.run_ocr_benchmark()
        nlp_results = self.run_nlp_benchmark()
        
        # Combine results (simplified)
        return BenchmarkResults(
            total_documents=self.config.num_documents,
            total_pages=ocr_results.total_pages,
            total_time=ocr_results.total_time + nlp_results.total_time,
            docs_per_second=self.config.num_documents / (ocr_results.total_time + nlp_results.total_time),
            pages_per_second=ocr_results.total_pages / (ocr_results.total_time + nlp_results.total_time),
            avg_gpu_utilization=(ocr_results.avg_gpu_utilization + nlp_results.avg_gpu_utilization) / 2,
            avg_gpu_memory=max(ocr_results.avg_gpu_memory, nlp_results.avg_gpu_memory),
            avg_cpu_utilization=(ocr_results.avg_cpu_utilization + nlp_results.avg_cpu_utilization) / 2,
            avg_system_memory=max(ocr_results.avg_system_memory, nlp_results.avg_system_memory),
            errors=ocr_results.errors + nlp_results.errors
        )
    
    def generate_report(self, results: BenchmarkResults):
        """Generate comprehensive benchmark report"""
        report = f"""
# Document Extraction Benchmark Report

## Configuration
- Workload Type: {self.config.workload_type}
- Total Documents: {self.config.num_documents}
- Pages per Document: {self.config.pages_per_doc}
- Batch Size: {self.config.batch_size}
- Workers: {self.config.num_workers}

## Performance Results
- Total Processing Time: {results.total_time:.2f} seconds
- Documents per Second: {results.docs_per_second:.2f}
- Pages per Second: {results.pages_per_second:.2f}

## Resource Utilization
- Average GPU Utilization: {results.avg_gpu_utilization:.1f}%
- Average GPU Memory: {results.avg_gpu_memory:.1f}%
- Average CPU Utilization: {results.avg_cpu_utilization:.1f}%
- Average System Memory: {results.avg_system_memory:.1f}%

## Extrapolated Performance
- 10K documents: {10000 / results.docs_per_second / 3600:.2f} hours
- 100K documents: {100000 / results.docs_per_second / 3600:.2f} hours
- 1M documents: {1000000 / results.docs_per_second / 3600:.2f} hours

## Cost Estimation (A100 @ $3.06/hour)
- Cost per 1K documents: ${3.06 * (1000 / results.docs_per_second / 3600):.2f}
- Cost per 100K documents: ${3.06 * (100000 / results.docs_per_second / 3600):.2f}
"""
        
        # Save report
        with open('benchmark_report.md', 'w') as f:
            f.write(report)
        
        # Save detailed metrics
        with open('benchmark_metrics.json', 'w') as f:
            json.dump({
                'config': self.config.__dict__,
                'results': results.__dict__,
                'detailed_metrics': self.metrics
            }, f, indent=2)
        
        return report

# Example usage
if __name__ == "__main__":
    # Configure benchmark
    config = BenchmarkConfig(
        workload_type='hybrid',
        num_documents=100,
        pages_per_doc=5,
        batch_size=16,
        num_workers=4
    )
    
    # Run benchmark
    benchmark = DocumentExtractionBenchmark(config)
    
    if config.workload_type == 'ocr':
        results = benchmark.run_ocr_benchmark()
    elif config.workload_type == 'nlp':
        results = benchmark.run_nlp_benchmark()
    elif config.workload_type == 'hybrid':
        results = benchmark.run_hybrid_benchmark()
    
    # Generate report
    report = benchmark.generate_report(results)
    print(report)
2.2 Production-Ready Benchmark Script
[python]
# production_benchmark.py
import asyncio
import aiofiles
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
import time
from typing import List, Dict, Any
import logging
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import signal
import sys

# Prometheus metrics
docs_processed = Counter('docs_processed_total', 'Total documents processed')
processing_time = Histogram('processing_time_seconds', 'Time spent processing documents')
gpu_utilization = Gauge('gpu_utilization_percent', 'GPU utilization percentage')
gpu_memory_usage = Gauge('gpu_memory_usage_percent', 'GPU memory usage percentage')
queue_size = Gauge('queue_size', 'Number of documents in processing queue')

class DocumentDataset(Dataset):
    def __init__(self, document_paths: List[Path]):
        self.paths = document_paths
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        # Load document (implement based on format)
        return self.paths[idx]

class ProductionBenchmark:
    def __init__(self, model_name: str, batch_size: int = 32, num_workers: int = 4):
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.running = True
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Start Prometheus metrics server
        start_http_server(8000)
        
    def _signal_handler(self, sig, frame):
        logging.info('Gracefully shutting down...')
        self.running = False
        sys.exit(0)
    
    async def process_document_batch(self, batch: List[Any]):
        """Process a batch of documents"""
        start = time.time()
        
        try:
            # Your actual processing logic here
            await asyncio.sleep(0.1)  # Simulate processing
            
            # Update metrics
            docs_processed.inc(len(batch))
            processing_time.observe(time.time() - start)
            
            # Monitor GPU
            if torch.cuda.is_available():
                gpu_utilization.set(torch.cuda.utilization())
                gpu_memory_usage.set(
                    torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
                )
            
        except Exception as e:
            logging.error(f"Error processing batch: {e}")
            raise
    
    async def run_benchmark(self, document_paths: List[Path]):
        """Run the benchmark with production-like conditions"""
        dataset = DocumentDataset(document_paths)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        tasks = []
        for batch in dataloader:
            if not self.running:
                break
                
            queue_size.set(len(tasks))
            task = asyncio.create_task(self.process_document_batch(batch))
            tasks.append(task)
            
            # Limit concurrent tasks
            if len(tasks) >= 10:
                await asyncio.gather(*tasks)
                tasks = []
        
        # Process remaining tasks
        if tasks:
            await asyncio.gather(*tasks)

# Run the benchmark
async def main():
    # Get document paths (example)
    document_paths = list(Path('/data/documents').glob('*.pdf'))
    
    benchmark = ProductionBenchmark(
        model_name='layoutlmv3-base',
        batch_size=32,
        num_workers=4
    )
    
    await benchmark.run_benchmark(document_paths)

if __name__ == "__main__":
    asyncio.run(main())
__________________________________________________
