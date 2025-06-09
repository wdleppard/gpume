
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import json
from benchmark_suite import BenchmarkConfig, DocumentExtractionBenchmark

app = FastAPI()
templates = Jinja2Templates(directory="/root/templates")

@app.get("/", response_class=HTMLResponse)
def form_get(request: Request):
    return templates.TemplateResponse("form.html", {"request": request, "result": None})

@app.post("/run", response_class=HTMLResponse)
def run_benchmark(request: Request,
                 workload: str = Form(...),
                 num_documents: int = Form(1000),
                 pages_per_doc: int = Form(5),
                 batch_size: int = Form(8),
                 num_workers: int = Form(4),
                 use_fp16: str = Form("off"),
                 use_multiprocessing: str = Form("off")):
    config = BenchmarkConfig(
        workload_type=workload,
        num_documents=num_documents,
        pages_per_doc=pages_per_doc,
        batch_size=batch_size,
        num_workers=num_workers,
        use_fp16=(use_fp16 == "on"),
        use_multiprocessing=(use_multiprocessing == "on")
    )
    benchmark = DocumentExtractionBenchmark(config)
    if workload == 'ocr':
        results = benchmark.run_ocr_benchmark()
    elif workload == 'nlp':
        results = benchmark.run_nlp_benchmark()
    elif workload == 'hybrid':
        results = benchmark.run_hybrid_benchmark()
    else:
        results = None
    return templates.TemplateResponse("form.html", {"request": request, "result": json.dumps(results.__dict__, indent=2)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
