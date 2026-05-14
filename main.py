import torch
import clip
from PIL import Image
import os
import pickle
import numpy as np


# Config
STORE_PATH = "image_store.pkl"
device = "cpu"  # change to "cuda" if GPU available

# Load CLIP model
model, preprocess = clip.load("ViT-B/32", device=device)

# In-memory store
image_store = []

# Load & Save
def load_store():
    global image_store

    if os.path.exists(STORE_PATH):
        with open(STORE_PATH, "rb") as f:
            image_store = pickle.load(f)
        print(f"Loaded {len(image_store)} images")
    else:
        image_store = []
        print("No existing store found, starting fresh")


def save_store():
    with open(STORE_PATH, "wb") as f:
        pickle.dump(image_store, f)


# Feature Extraction
def extract_image_features(image_path):
    with torch.inference_mode():
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

        features = model.encode_image(image)
        features = features / features.norm(dim=-1, keepdim=True)

        # convert to numpy 
        return features.squeeze(0).cpu().numpy()

def extract_text_featuers(text):
    with torch.inference_mode():
        text = clip.tokenize([text]).to(device=device)

        features = model.encode_text(text=text)
        features = features / features.norm(dim=-1, keepdim=True)

        return features.squeeze(0).cpu().numpy()

# Add Image
def add_image(image_path):
    for img in image_store:
        if img["path"] == image_path:
            print(f"Skipped (already exists): {image_path}")
            return
        
    features = extract_image_features(image_path)

    image_store.append({
        "id": len(image_store),
        "path": image_path,
        "features": features
    })

    save_store()
    print(f"Added: {image_path}")


# Cosine Similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# Search (Image → Image)
def image_search(query_image_path, top_k=3):
    query_features = extract_image_features(query_image_path)

    results = []

    for img in image_store:
        score = cosine_similarity(query_features, img["features"])
        results.append((score, img["path"]))

    results.sort(reverse=True, key=lambda x: x[0])

    return results[:top_k]

# Search (Text → Image)
def text_search(text, top_k=3):
    query_featuers = extract_text_featuers(text)

    results = []

    for img in image_store:
        score = cosine_similarity(query_featuers,img["features"])
        results.append((score, img["path"]))

    results.sort(reverse=True, key=lambda x : x[0])

    return results[:top_k]


# Run
if __name__ == "__main__":
    load_store()

    add_image("images/dog.jpg")
    add_image("images/car.jpg")
    add_image("images/person_running.jpg")

    # Search via image
    results = image_search("images/dog2.jpg", top_k=3)

    print("\nTop Matches (Image Search) :")
    for score, path in results:
        print(f"{path} -> {score:.4f}")

    # Search via text
    results = text_search("a person running",top_k = 3)

    print("\nTop Matches (Text Search):")
    for score, path in results:
        print(f"{path} -> {score:.4f}")