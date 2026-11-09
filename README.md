# PyGGUF

Python, PyTorch and Triton GGUF inference server. See `PyGGUF_GitHub_Kaggle.ipynb` to clone this repository and run it on a Kaggle GPU, or `PyGGUF_Kaggle_OpenAI_Server.ipynb` for the self-contained version.

## Kaggle

Select a dual-T4 accelerator. Run the GitHub-backed notebook cells in order. By default it fetches `main`, so review the repository or set `GIT_REF` to a trusted full commit SHA in the clone cell. The notebook downloads model weights separately from Hugging Face. A valid GGUF model, sufficient GPU and host memory, and network access are required. The notebook provides a stop/relaunch mechanism; relaunching reloads the model.

## Tests

Install dependencies including torch, triton, numpy, gguf, fastapi, httpx, jinja2, tokenizers and pydantic, then run:

```sh
python -m tests.test_pygguf_engine
```

T4 performance and CUDA-specific behavior should be benchmarked separately on actual hardware.
