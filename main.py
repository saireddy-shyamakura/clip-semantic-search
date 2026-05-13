import torch
import clip
from PIL import Image

device = "cpu"

model, preprocess = clip.load("ViT-B/32",device=device)

def upload_images(image):
    with torch.inference_mode():
        image = preprocess(Image.open(image)).unsqueeze(0).to(device=device)
        image_features = model.encode_image(image=image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

def search_images(texts_list,image_features):
    with torch.inference_mode():
        texts = clip.tokenize(texts_list).to(device=device)

        text_features = model.encode_text(text=texts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        similarity = (image_features @ text_features.T)

        best_score = -float("inf")
        best_index = -1

        for i, r in enumerate(similarity[0]):
            if r.item() > best_score:
                best_score = r.item()
                best_index = i

        print("Best match:", texts_list[best_index])
        print("Score:", best_score)


image_features = upload_images("images/dog.jpg")
search_images(["dog","car","person running"],image_features)
