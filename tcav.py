import torch
import numpy as np
from sklearn.linear_model import SGDClassifier
from xai_utils import load_models_and_processors, preprocess_image
from datasets import load_dataset
import cv2

class TCAV:
    def __init__(self, model, processor, device, concept_name, concept_func, random_images):
        """
        TCAV for a given concept.

        concept_func: function that takes an image (PIL) and returns True if it belongs to the concept
        random_images: list of PIL images for random set
        """
        self.model = model
        self.processor = processor
        self.device = device
        self.concept_name = concept_name

        # Load dataset for concept images
        dataset = load_dataset("huggan/wikiart", streaming=True)
        test_dataset = list(dataset["train"].take(1000))  # Sample more for concepts

        concept_images = []
        for example in test_dataset:
            img = example["image"].convert("RGB")
            if concept_func(img):
                concept_images.append(img)

        self.concept_images = concept_images[:50]  # Limit to 50
        self.random_images = random_images[:50]

        # Get activations
        self.concept_acts = self.get_activations(self.concept_images)
        self.random_acts = self.get_activations(self.random_images)

        # Train CAV
        self.cav = self.train_cav()

    def get_activations(self, images):
        activations = []
        for img in images:
            inputs = preprocess_image(img, self.processor, self.device)
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                # Use last hidden state before classifier
                act = outputs.hidden_states[-1].mean(dim=1).squeeze().cpu().numpy()
                activations.append(act)
        return np.array(activations)

    def train_cav(self):
        X = np.vstack([self.concept_acts, self.random_acts])
        y = np.array([1] * len(self.concept_acts) + [0] * len(self.random_acts))
        clf = SGDClassifier(random_state=42)
        clf.fit(X, y)
        return clf.coef_[0]

    def compute_tcav_score(self, image):
        """
        Compute TCAV score for a single image.
        """
        act = self.get_activations([image])[0]
        score = np.dot(self.cav, act)
        return score

# Concept functions
def brightness_concept(img):
    img_np = np.array(img)
    brightness = img_np.mean()
    return brightness > 150

def warm_color_concept(img):
    img_np = np.array(img)
    r = img_np[:, :, 0].mean()
    b = img_np[:, :, 2].mean()
    return r > b + 20

def texture_concept(img):
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = edges.mean()
    return edge_density > 20

# Precompute TCAVs
def load_tcavs(device):
    (vit_model, vit_processor), (resnet_model, resnet_processor) = load_models_and_processors(device)

    # Random images
    dataset = load_dataset("huggan/wikiart", streaming=True)
    random_images = [ex["image"].convert("RGB") for ex in dataset["train"].take(100)][50:100]

    tcavs = {}

    # For ResNet
    tcavs['resnet_brightness'] = TCAV(resnet_model, resnet_processor, device, 'brightness', brightness_concept, random_images)
    tcavs['resnet_warm_color'] = TCAV(resnet_model, resnet_processor, device, 'warm_color', warm_color_concept, random_images)
    tcavs['resnet_texture'] = TCAV(resnet_model, resnet_processor, device, 'texture', texture_concept, random_images)

    # For ViT
    tcavs['vit_brightness'] = TCAV(vit_model, vit_processor, device, 'brightness', brightness_concept, random_images)
    tcavs['vit_warm_color'] = TCAV(vit_model, vit_processor, device, 'warm_color', warm_color_concept, random_images)
    tcavs['vit_texture'] = TCAV(vit_model, vit_processor, device, 'texture', texture_concept, random_images)

    return tcavs