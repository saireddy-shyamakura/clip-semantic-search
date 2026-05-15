# clip-semantic-search

clip-semantic-search is a semantic search engine that uses OpenAI CLIP to search through a local database of images using text descriptions or other images. It builds a fast, vectorized local index for rapid retrieval.

## Overview

The project loads a CLIP model (`ViT-B/32`), encodes images into vector embeddings, and stores them in a persistent index (`image_store.pkl`). It allows you to search for images by providing either a text query (Text-to-Image search) or another image (Image-to-Image search). 

## Features

- **Batch Ingestion**: Easily add single images, multiple images, or entire directories at once using `add_images()`.
- **Vectorized Search**: Fast image retrieval using NumPy matrix multiplication.
- **Persistent Index**: Images and their features are saved to `image_store.pkl` to avoid re-encoding on every run.
- **Text-to-Image Search**: Search your image database using natural language queries.
- **Image-to-Image Search**: Find visually similar images by querying with an image.
- **RGB Conversion**: Automatically handles varied image formats (e.g. PNGs with transparency) safely.

## Requirements

- Python 3.12+
- `torch`
- `Pillow`
- `clip` (OpenAI CLIP)
- `numpy`

## Installation

This project uses `uv` to manage the environment and dependencies.

1. Install `uv` if needed:

```bash
python -m pip install uv
```

2. Sync dependencies:

```bash
uv sync
```

## Usage

Run the main script using `uv`:

```bash
uv run python main.py
```

### Adding Images

You can add individual image files or entire directories to your local index. Features are automatically extracted and saved.

```python
# Add multiple specific files
add_images("images/dog.jpg", "images/car.jpg")

# Add an entire directory of images
add_images("images")
```

### Searching

**Image Search**
Find images in your index that are visually similar to a query image:
```python
results = image_search("images/query.jpg", top_k=3)
```

**Text Search**
Find images in your index that match a text description:
```python
results = text_search("a person running", top_k=3)
```

## Example Output

```text
Loaded 4 images
Added: /absolute/path/to/images/new_image.jpg

Top Matches (Image Search) :
/path/to/images/dog2.jpg -> 1.0000
/path/to/images/dog.jpg -> 0.7702

Top Matches (Text Search):
/path/to/images/person_running.jpg -> 0.2761
/path/to/images/dog.jpg -> 0.2505
```

## Configuration

To use a GPU for faster CLIP inference, update the `device` variable at the top of `main.py`:

```python
device = "cuda" # Set to "cuda" for NVIDIA GPUs or "mps" for Apple Silicon
```
