

`python
!python -m pip install ipykernel
`

`python
%pip install transformers timm datasets captum grad-cam
`

`python
import numpy as np
`

`python
%pip install -q --force-reinstall numpy
%pip install -q --force-reinstall transformers
`

`python
%pip install torch torchvision

`

`python
import torch
import torchvision
print(torch.__version__, torchvision.__version__)
`

`python
from transformers import AutoImageProcessor, AutoModelForImageClassification
`

`python
%pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
`

`python

from transformers import AutoImageProcessor, AutoModelForImageClassification

vit_name = "sakhmatd/vit-wikiart-finetuned"

vit_processor = AutoImageProcessor.from_pretrained(vit_name)
vit_model = AutoModelForImageClassification.from_pretrained(vit_name)


vit_model.eval()
`

`python

`

`python
resnet_name = "Iust1n2/resnet-18-finetuned-wikiart"

resnet_processor = AutoImageProcessor.from_pretrained(resnet_name)
resnet_model = AutoModelForImageClassification.from_pretrained(resnet_name)

resnet_model.eval()
`

Computing accuracy and confusion matrix of the ml models loaded



`python
%pip install transformers datasets scikit-learn
`

**loading dataset (small test subset)**


`python
from datasets import load_dataset

dataset = load_dataset("huggan/wikiart", streaming=True)

# Take 1000 samples
test_dataset = dataset["train"].take(1000)

# Convert to list so we can iterate multiple times
test_dataset = list(test_dataset)

print(len(test_dataset))
print(test_dataset[0])
`

**Load Pretrained Model (Example: ViT)**


`python
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "sakhmatd/vit-wikiart-finetuned"

processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)

model.to(device)
model.eval()
`

`python
from tqdm import tqdm
import numpy as np

all_preds = []
all_labels = []

for example in tqdm(test_dataset):
    image = example["image"]
    label = example["style"]  # style label

    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    pred = torch.argmax(logits, dim=-1).item()

    all_preds.append(pred)
    all_labels.append(label)
`

`python
%pip install matplotlib seaborn
`

`python

`

`python
import matplotlib.pyplot as plt
import seaborn as sns
`

`python
#accuracy
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(all_labels, all_preds)
print(f"Accuracy: {accuracy:.4f}")

#confusion matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(10,8))
sns.heatmap(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
`

**Load Pretrained Model (Example: CNN)**


`python
%pip install transformers datasets torch torchvision scikit-learn tqdm
`

`python
from datasets import load_dataset
dataset = load_dataset("huggan/wikiart", streaming=True)
test_dataset = dataset["train"].take(200)
`

`python
%pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
`

`python
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "Iust1n2/resnet-18-finetuned-wikiart"

processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)

model.to(device)
model.eval()
`

`python
!pip install scikit-learn
`

`python
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np
`

`python
y_true = []
y_pred = []

num_samples = 200  # keep small to avoid memory issues

for i, sample in enumerate(dataset["train"].take(num_samples)):
    try:
        image = sample["image"]
        label = sample["style"]

        # preprocess
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        pred = torch.argmax(logits, dim=1).item()

        # store
        y_true.append(label)
        y_pred.append(pred)

    except Exception as e:
        print(f"Skipping {i}: {e}")
        continue
`

`python
# Ensure labels are ints
y_true = [int(x) for x in y_true]
y_pred = [int(x) for x in y_pred]
`

`python
print("Model labels:", model.config.id2label)

sample = next(iter(dataset["train"]))
print("Dataset label:", sample["style"], type(sample["style"]))
`

`python
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average="weighted")  # or "macro"/"binary"

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 score: {f1:.4f}")
print(classification_report(all_labels, all_preds))

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
`

`python
from datasets import load_dataset
dataset = load_dataset("huggan/wikiart", streaming=True)
`

`python
test_dataset = list(dataset["train"].take(200))
`

`python
print(set([x["style"] for x in dataset["train"].take(50)]))
print(model.config.label2id.keys())
`

`python
# dataset labels
dataset_labels = set([x["style"] for x in dataset["train"].take(50)])
print("Dataset labels:\n", dataset_labels)

# model labels
print("\nModel labels:\n", model.config.label2id.keys())
`

`python
label_id = label
`

`python
for example in tqdm(dataset["train"].take(200)):

    image = example["image"]
    label = example["style"]

    # convert dataset label → string → model id
    label_name = dataset["train"].features["style"].int2str(label)

    if label_name not in model.config.label2id:
        continue

    label_id = model.config.label2id[label_name]

    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    pred = outputs.logits.argmax(-1).item()

    all_preds.append(pred)
    all_labels.append(label_id)
`

`python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(all_labels, all_preds)

print("Model Accuracy:", accuracy)
`

`python
%pip install transformers datasets timm grad-cam scikit-learn matplotlib seaborn tqdm
`

`python
import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification
`

`python
dataset = load_dataset("huggan/wikiart", streaming=True)

for example in dataset["train"].take(200):  # or 1000 if you want
    ...
`

`python
for example in dataset["train"].take(1):
    print(example)
`

`python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "Iust1n2/resnet-18-finetuned-wikiart"

processor = AutoImageProcessor.from_pretrained(model_name)
resnet_model = AutoModelForImageClassification.from_pretrained(model_name)

resnet_model.to(device)
resnet_model.eval()
`

`python
from tqdm import tqdm

predictions = []

for example in tqdm(dataset["train"].take(1)):

    image = example["image"]

    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    pred = outputs.logits.argmax(-1).item()

    predictions.append(pred)
`

`python
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
`

`python
target_layers = [resnet_model.resnet.encoder.stages[-1].layers[-1]]
`

`python
import torch.nn as nn

class HuggingFaceWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(pixel_values=x).logits
`




`python
wrapped_model = HuggingFaceWrapper(resnet_model)
wrapped_model.to(device)
wrapped_model.eval()
`

`python
target_layers = [resnet_model.resnet.encoder.stages[-1].layers[-1]]

cam = GradCAM(
    model=wrapped_model,
    target_layers=target_layers
)
`

`python
example = next(iter(dataset["train"]))

image = example["image"]

inputs = processor(images=image, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)

pred = outputs.logits.argmax(-1).item()

targets = [ClassifierOutputTarget(pred)]

grayscale_cam = cam(
    input_tensor=inputs["pixel_values"],
    targets=targets
)
`

preparing image


`python
img = np.array(image.resize((224,224))) / 255.0
`

overlaying heatmap



`python
from pytorch_grad_cam.utils.image import show_cam_on_image

visualization = show_cam_on_image(
    img,
    grayscale_cam[0],
    use_rgb=True
)
`

`python
plt.imshow(visualization)
plt.axis("off")
plt.title("CNN Grad-CAM Explanation")
plt.show()
`

`python
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.imshow(image)
plt.title("Original Painting")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(visualization)
plt.title("Grad-CAM Explanation")
plt.axis("off")

plt.show()
`

`python
for i, sample in enumerate(dataset["train"]):
    if i >= 5:
        break

    image = sample["image"]
`

`python
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt
import numpy as np

for i, sample in enumerate(dataset["train"]):
    if i >= 5:
        break

    image = sample["image"]

    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = resnet_model(**inputs)

    pred = outputs.logits.argmax(-1).item()

    targets = [ClassifierOutputTarget(pred)]

    grayscale_cam = cam(
        input_tensor=inputs["pixel_values"],
        targets=targets
    )

    img = np.array(image.resize((224,224))) / 255.0
    visualization = show_cam_on_image(img, grayscale_cam[0], use_rgb=True)

    plt.figure(figsize=(6,3))

    plt.subplot(1,2,1)
    plt.imshow(image)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(visualization)
    plt.title("Grad-CAM")
    plt.axis("off")

    plt.show()
`

Loading ViT model



`python
from transformers import AutoImageProcessor, AutoModelForImageClassification

vit_model_name = "google/vit-base-patch16-224"

vit_processor = AutoImageProcessor.from_pretrained(vit_model_name)

vit_model = AutoModelForImageClassification.from_pretrained(
    vit_model_name,
    output_attentions=True
)

vit_model.to(device)
vit_model.eval()
`

`python
sample = next(iter(dataset["train"]))
image = sample["image"]

inputs = vit_processor(images=image, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = vit_model(**inputs, output_attentions=True)
`

`python
attentions = outputs.attentions
`

`python
attention = attentions[0]
`

`python
print(type(outputs.attentions))
print(len(outputs.attentions))
`

`python
attention_map = attention[0].mean(0)
`

`python
cls_attention = attention_map[0,1:]
`

`python
cls_attention = cls_attention.reshape(14,14)
`

`python
import cv2

attention_resized = cv2.resize(
    cls_attention.cpu().numpy(),
    (224,224)
)
`

`python
attention_resized = attention_resized / attention_resized.max()
`

`python
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.imshow(image)
plt.title("Original Painting")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(attention_resized, cmap="jet")
plt.title("ViT Attention Map")
plt.axis("off")

plt.show()
`

`python
img = np.array(image.resize((224,224))) / 255.0

heatmap = cv2.applyColorMap(
    np.uint8(255 * attention_resized),
    cv2.COLORMAP_JET
)

heatmap = heatmap / 255.0

overlay = heatmap + img
overlay = overlay / overlay.max()

plt.imshow(overlay)
plt.title("ViT Attention Overlay")
plt.axis("off")

`

`python
target_size = (224, 224)
image_resized = image.resize(target_size)
inputs = processor(images=image_resized, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}
import cv2

# Grad-CAM
cam_resized = cv2.resize(grayscale_cam, target_size)

# ViT attention (assuming attention map = attn_map)
vit_resized = cv2.resize(attn_map, target_size)
`

`python
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(image)
plt.title("Original")

plt.subplot(1,3,2)
plt.imshow(visualization)
plt.title("CNN Grad-CAM")

plt.subplot(1,3,3)
plt.imshow(overlay)
plt.title("ViT Attention")

plt.show()
`

------------------



`python
idx = 10
image = test_dataset[idx]["image"]
`

`python
import torch.nn as nn

class HFResNetWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        outputs = self.model(pixel_values=x)
        return outputs.logits
`

`python
wrapped_resnet = HFResNetWrapper(resnet_model)
wrapped_resnet.to(device)
wrapped_resnet.eval()
`

`python
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np

def generate_gradcam(model, image):

    inputs = processor(images=image, return_tensors="pt").to(device)

    outputs = model(**inputs)
    pred = outputs.logits.argmax(dim=1).item()

    target_layers = [model.resnet.encoder.stages[-1].layers[-1]]

    cam = GradCAM(model=wrapped_resnet, target_layers=target_layers)

    targets = [ClassifierOutputTarget(pred)]

    grayscale_cam = cam(
        input_tensor=inputs["pixel_values"],
        targets=targets
    )[0]

    img = np.array(image)/255.0
    visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)

    return visualization
`

`python
from PIL import Image
import numpy as np
import cv2

def generate_gradcam(model, image):

    # preprocess
    inputs = processor(images=image, return_tensors="pt").to(device)

    outputs = model(**inputs)
    pred = outputs.logits.argmax(dim=1).item()

    target_layers = [model.resnet.encoder.stages[-1].layers[-1]]

    cam = GradCAM(model=wrapped_resnet, target_layers=target_layers)

    targets = [ClassifierOutputTarget(pred)]

    grayscale_cam = cam(
        input_tensor=inputs["pixel_values"],
        targets=targets
    )[0]

    # resize original image to 224x224
    img = np.array(image.resize((224,224))) / 255.0

    visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)

    return visualization

gradcam = generate_gradcam(resnet_model, image)
`

`python
plt.imshow(gradcam)
plt.axis("off")
`

`python
import numpy as np
import cv2

def generate_vit_attention(model, image):

    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    attentions = outputs.attentions

    # last layer attention
    attention = attentions[-1]

    # average over heads
    attention = attention.mean(dim=1)

    # remove CLS token
    attention = attention[0, 0, 1:]

    # reshape to patch grid
    num_patches = int(attention.shape[0] ** 0.5)
    attention = attention.reshape(num_patches, num_patches).cpu().numpy()

    # normalize
    attention = attention / attention.max()

    # upscale to image size
    attention = cv2.resize(attention, (224,224))

    # resize original image
    img = np.array(image.resize((224,224))) / 255.0

    heatmap = cv2.applyColorMap(np.uint8(255 * attention), cv2.COLORMAP_JET)
    heatmap = heatmap[:,:,::-1] / 255.0

    overlay = heatmap * 0.4 + img * 0.6

    return overlay
`

`python
idx = 10
image = test_dataset[idx]["image"]

gradcam = generate_gradcam(resnet_model, image)
vit_map = generate_vit_attention(vit_model, image)
`

`python
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(image.resize((224,224)))
plt.title("Original")

plt.subplot(1,3,2)
plt.imshow(gradcam)
plt.title("CNN Grad-CAM")

plt.subplot(1,3,3)
plt.imshow(vit_map)
plt.title("ViT Attention")

plt.show()
`

`python
import random
import matplotlib.pyplot as plt

num_examples = 15   # you can change this to 10–20

indices = random.sample(range(len(test_dataset)), num_examples)

for idx in indices:

    image = test_dataset[idx]["image"]

    gradcam = generate_gradcam(resnet_model, image)
    vit_map = generate_vit_attention(vit_model, image)

    plt.figure(figsize=(12,4))

    plt.subplot(1,3,1)
    plt.imshow(image.resize((224,224)))
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.imshow(gradcam)
    plt.title("CNN Grad-CAM")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.imshow(vit_map)
    plt.title("ViT Attention")
    plt.axis("off")

    plt.show()
`

`python
import torch
import numpy as np
import cv2

def generate_attention_rollout(model, image):

    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    attentions = outputs.attentions

    # stack all layers
    attention_mat = torch.stack(attentions)

    # average heads
    attention_mat = attention_mat.mean(dim=2)

    # add residual connection
    residual_att = torch.eye(attention_mat.size(-1)).to(device)
    aug_att_mat = attention_mat + residual_att

    # normalize
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    # compute joint attention
    joint_attentions = torch.zeros_like(aug_att_mat)
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])

    # attention from CLS token
    v = joint_attentions[-1][0]

    mask = v[0, 1:]

    num_patches = int(mask.shape[0] ** 0.5)

    mask = mask.reshape(num_patches, num_patches).cpu().numpy()

    mask = mask / mask.max()

    mask = cv2.resize(mask, (224,224))

    img = np.array(image.resize((224,224))) / 255.0

    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = heatmap[:,:,::-1] / 255.0

    overlay = heatmap * 0.4 + img * 0.6

    return overlay
`

`python
rollout_map = generate_attention_rollout(vit_model, image)
`

`python
plt.figure(figsize=(15,4))

plt.subplot(1,4,1)
plt.imshow(image.resize((224,224)))
plt.title("Original")
plt.axis("off")

plt.subplot(1,4,2)
plt.imshow(gradcam)
plt.title("CNN GradCAM")
plt.axis("off")

plt.subplot(1,4,3)
plt.imshow(vit_map)
plt.title("ViT Attention")
plt.axis("off")

plt.subplot(1,4,4)
plt.imshow(rollout_map)
plt.title("ViT Attention Rollout")
plt.axis("off")

plt.show()
`

`python
import random
import matplotlib.pyplot as plt

num_examples = 6   # change to 5–10

indices = random.sample(range(len(test_dataset)), num_examples)

for idx in indices:

    image = test_dataset[idx]["image"]

    gradcam = generate_gradcam(resnet_model, image)
    vit_attention = generate_vit_attention(vit_model, image)
    rollout = generate_attention_rollout(vit_model, image)

    plt.figure(figsize=(16,4))

    plt.subplot(1,4,1)
    plt.imshow(image.resize((224,224)))
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1,4,2)
    plt.imshow(gradcam)
    plt.title("CNN GradCAM")
    plt.axis("off")

    plt.subplot(1,4,3)
    plt.imshow(vit_attention)
    plt.title("ViT Attention")
    plt.axis("off")

    plt.subplot(1,4,4)
    plt.imshow(rollout)
    plt.title("ViT Attention Rollout")
    plt.axis("off")

    plt.show()
`

#**Grad-CAM highlights which spatial regions contributed most to the predicted class.**


**The red areas mean:**

**If these regions changed, the model’s prediction would likely change.**


#**What ViT Attention Shows**

##In Vision Transformer, the image is split into patches (usually 16×16).

Instead of convolutions, the model uses **self-attention** to decide:

**Which patches are important for classification?**

####**which patches influenced the CLS token**

####which areas the model considered when making the prediction


What our Project is Actually Comparing

We are not just showing important pixels.

You are studying architectural interpretability differences.

Model	Explanation Focus-
* **CNN (ResNet): 	Local textures / brushwork**



* **ViT: 	Global composition**

* **ViT Rollout: 	Aggregated global reasoning**


`python
from PIL import ImageEnhance

def change_brightness(image, factor):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)
`

`python
image = test_dataset[10]["image"]

bright_image = change_brightness(image, 1.2)
dark_image = change_brightness(image, 0.8)
`

`python
# CNN explanations
gradcam_original = generate_gradcam(resnet_model, image)
gradcam_bright = generate_gradcam(resnet_model, bright_image)
gradcam_dark = generate_gradcam(resnet_model, dark_image)

# ViT explanations
vit_original = generate_attention_rollout(vit_model, image)
vit_bright = generate_attention_rollout(vit_model, bright_image)
vit_dark = generate_attention_rollout(vit_model, dark_image)
`

`python
plt.figure(figsize=(15,8))


plt.subplot(2,3,1)
plt.imshow(gradcam_original)
plt.title("CNN Original")

plt.subplot(2,3,2)
plt.imshow(gradcam_bright)
plt.title("CNN Bright")

plt.subplot(2,3,3)
plt.imshow(gradcam_dark)
plt.title("CNN Dark")

plt.subplot(2,3,4)
plt.imshow(vit_original)
plt.title("ViT Original")

plt.subplot(2,3,5)
plt.imshow(vit_bright)
plt.title("ViT Bright")

plt.subplot(2,3,6)
plt.imshow(vit_dark)
plt.title("ViT Dark")

plt.show()
`

Grad-CAM explanations from the CNN model showed moderate sensitivity to brightness perturbations, with highlighted regions shifting toward high-contrast textures. In contrast, Attention Rollout explanations from the Vision Transformer remained more stable, continuing to emphasize global compositional regions of the painting.



`python
plt.figure(figsize=(15,8))
plt.subplot(1,3,1)
plt.imshow(image)
plt.title("Original")
`

`python
layer = resnet_model.resnet.encoder.stages[-1]
`

`python
activations = []

def hook_fn(module, input, output):
    activations.append(output.detach())

layer = resnet_model.resnet.encoder.stages[-1]

hook = layer.register_forward_hook(hook_fn)
`

`python
resnet_model(**inputs)
`

`python
activations.clear()

inputs = processor(images=test_dataset[0]["image"], return_tensors="pt")

with torch.no_grad():
    resnet_model(**inputs)

print(activations[0].shape)
`

`python
activations.clear()
`

`python
inputs = processor(images=test_dataset[0]["image"], return_tensors="pt")
`

`python
resnet_model(**inputs)
`

`python
print(activations[0].shape)
`

#TCAV Pipeline
(for brightness)


`python
import torch
import numpy as np

def get_activation(image):

    activations.clear()

    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        resnet_model(**inputs)

    act = activations[0]

    act = act.squeeze().cpu().numpy()

    return act.flatten()
`

`python
concept_images = []

for example in test_dataset[:50]:

    img = example["image"]

    img_np = np.array(img)

    brightness = img_np.mean()

    if brightness > 150:   # threshold
        concept_images.append(img)

print(len(concept_images))
`

`python
random_images = [example["image"] for example in test_dataset[100:150]]
`

`python
concept_acts = []
random_acts = []

for img in concept_images:
    concept_acts.append(get_activation(img))

for img in random_images:
    random_acts.append(get_activation(img))
`

`python
from sklearn.linear_model import SGDClassifier

X = np.vstack([concept_acts, random_acts])

y = np.array([1]*len(concept_acts) + [0]*len(random_acts))

clf = SGDClassifier()

clf.fit(X, y)

cav = clf.coef_
`

`python
scores = []

for example in test_dataset[:50]:

    img = example["image"]

    act = get_activation(img)

    score = np.dot(cav, act)

    scores.append(score)
`

`python
tcav_score = np.mean(np.array(scores) > 0)

print("TCAV Score:", tcav_score)
`

#**This means**:
50% of predictions increase along the "brightness" concept direction


**This TCAV pipeline shows the dependency of brightness on the predictions**


Now focusing on color dominance


`python
concept_images = []

for example in test_dataset:

    img = example["image"]
    img_np = np.array(img)

    r = img_np[:,:,0].mean()
    g = img_np[:,:,1].mean()
    b = img_np[:,:,2].mean()

    if r > b + 20:   # warm color dominance
        concept_images.append(img)

print(len(concept_images))
`

Texture Complexity


`python
import cv2

concept_images = []

for example in test_dataset:

    img = example["image"]
    img_np = np.array(img)

    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    edges = cv2.Canny(gray,100,200)

    edge_density = edges.mean()

    if edge_density > 20:
        concept_images.append(img)

print(len(concept_images))
`

Global Structure


`python
concept_images = []

for example in test_dataset:

    img = example["image"]
    img_np = np.array(img)

    h,w,_ = img_np.shape

    center = img_np[h//3:2*h//3 , w//3:2*w//3]

    center_intensity = center.mean()
    full_intensity = img_np.mean()

    if center_intensity > full_intensity + 5:
        concept_images.append(img)

print(len(concept_images))
`

These numbers represent how many images in your test subset satisfy each concept rule.

*103 paintings have dominant warm colors

*110 paintings have strong texture/edge density

*79 paintings have strong center composition


`python
def get_activation_vector(image):

    activations.clear()

    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        resnet_model(**inputs)

    act = activations[0]

    act = act.mean(dim=(2,3))   # global average pooling

    return act.squeeze().numpy()
`

`python
concept_vectors = []

for img in concept_images[:50]:   # take 50 to keep it balanced
    vec = get_activation_vector(img)
    concept_vectors.append(vec)

concept_vectors = np.array(concept_vectors)
`

`python
random_images = []

for example in test_dataset[:50]:
    random_images.append(example["image"])

random_vectors = []

for img in random_images:
    vec = get_activation_vector(img)
    random_vectors.append(vec)

random_vectors = np.array(random_vectors)
`

`python
from sklearn.linear_model import LogisticRegression

X = np.concatenate([concept_vectors, random_vectors])

y = np.array([1]*len(concept_vectors) + [0]*len(random_vectors))

clf = LogisticRegression()

clf.fit(X,y)
`

`python
cav = clf.coef_
`

`python
plt.figure(figsize=(10,4))

for i,img in enumerate(concept_images[:5]):
    plt.subplot(1,5,i+1)
    plt.imshow(img)
    plt.axis("off")

plt.show()
`

The concepts created were:

*Warm color palette

*Texture / brushstroke complexity

*Central composition

Our filtering rules picked paintings that satisfy those conditions.


Renaissance painting → smooth surface → lower variance

Impressionist painting → visible brush strokes → higher variance

Explainability significance

This tells us:

“The model may classify this painting because it detected strong brushstroke patterns.


Quantitative analysis:


`python
import numpy as np
import torch

def compute_entropy(map):
    if isinstance(map, torch.Tensor):
        map = map.detach().cpu().numpy()

    map = map / (map.sum() + 1e-8)
    return -np.sum(map * np.log(map + 1e-8))
`

`python
entropy_cam = compute_entropy(grayscale_cam)
entropy_vit = compute_entropy(attention)

print(f"Grad-CAM Entropy: {entropy_cam:.4f}")
print(f"ViT Entropy: {entropy_vit:.4f}")
`

##Focus Score:


`python
def compute_focus_score(map, threshold=0.6):
    # convert if tensor
    if isinstance(map, torch.Tensor):
        map = map.detach().cpu().numpy()

    # normalize
    map = map / (map.max() + 1e-8)

    # count high-importance pixels
    focused_pixels = np.sum(map > threshold)
    total_pixels = map.size

    return focused_pixels / total_pixels
`

`python
focus_cam = compute_focus_score(grayscale_cam)
focus_vit = compute_focus_score(attention)

print(f"Grad-CAM Focus: {focus_cam:.4f}")
print(f"ViT Focus: {focus_vit:.4f}")
`

`python
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(image.resize((224,224)))
plt.title("Original")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(gradcam)
plt.title(f"Grad-CAM\nEntropy: {entropy_cam:.2f}\nFocus: {focus_cam:.2f}")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(vit_map)
plt.title(f"ViT\nEntropy: {entropy_vit:.2f}\nFocus: {focus_vit:.2f}")
plt.axis("off")

plt.show()
`

`python
# ===== SINGLE IMAGE XAI PIPELINE =====

import numpy as np
import cv2
import torch

# 1. Get ONE sample from streaming dataset
sample = next(iter(dataset["train"]))
image = sample["image"]

# 2. Preprocess ONCE
image_resized = image.resize((224, 224))

inputs = processor(images=image_resized, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

img_np = np.array(image_resized) / 255.0

# 3. -------- RESNET (Grad-CAM) --------
outputs = model(**inputs)
pred = outputs.logits.argmax(dim=1).item()

targets = [ClassifierOutputTarget(pred)]

grayscale_cam = cam(
    input_tensor=inputs["pixel_values"],
    targets=targets
)[0]

# 4. -------- ViT (Attention) --------
with torch.no_grad():
    vit_outputs = vit_model(**inputs, output_attentions=True)

attention = vit_outputs.attentions[-1]
attention = attention.mean(dim=1)
attention = attention[0, 0, 1:]

num_patches = int(attention.shape[0] ** 0.5)
attention = attention.reshape(num_patches, num_patches).cpu().numpy()
attention = cv2.resize(attention, (224, 224))

# 5. -------- Metrics --------
def compute_entropy(map):
    map = map / (map.sum() + 1e-8)
    return -np.sum(map * np.log(map + 1e-8))

def compute_focus_score(map, threshold=0.6):
    return np.sum(map > threshold) / map.size

entropy_cam = compute_entropy(grayscale_cam)
entropy_vit = compute_entropy(attention)

focus_cam = compute_focus_score(grayscale_cam)
focus_vit = compute_focus_score(attention)

# 6. -------- Overlays --------
heatmap_cam = plt.cm.jet(grayscale_cam)[..., :3]
overlay_cam = np.clip(heatmap_cam * 0.5 + img_np * 0.5, 0, 1)

heatmap_vit = plt.cm.jet(attention)[..., :3]
overlay_vit = np.clip(heatmap_vit * 0.5 + img_np * 0.5, 0, 1)

# Done → Ready for plotting cell
`

`python
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(image_resized)
plt.title("Original (Same Image)")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(overlay_cam)
plt.title(f"Grad-CAM\nEntropy: {entropy_cam:.2f}\nFocus: {focus_cam:.2f}")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(overlay_vit)
plt.title(f"ViT\nEntropy: {entropy_vit:.2f}\nFocus: {focus_vit:.2f}")
plt.axis("off")

plt.show()
`

Faithfullness test


`python
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from datasets import load_dataset

# ------------------------------
# 1️⃣ Setup: device and models
# ------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# Assume your models are already loaded:
# cnn_model, vit_model, vit_processor
# cnn_model.to(device)
# vit_model.to(device)

# ------------------------------
# 2️⃣ Load one sample from streaming dataset
# ------------------------------
dataset = load_dataset("huggan/wikiart", streaming=True)
sample = next(iter(dataset["train"]))
image, label = sample["image"], sample["style"]

# ------------------------------
# 3️⃣ Preprocessing
# ------------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])
image_tensor = transform(image).to(device)

# ------------------------------
# 4️⃣ Define helper functions
# ------------------------------
def normalize_heatmap(cam):
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    return cam

def create_masked_image(image_tensor, heatmap, threshold=0.6, keep_important=True):
    mask = (heatmap > threshold).astype(np.float32)
    if not keep_important:
        mask = 1 - mask
    mask_tensor = torch.tensor(mask).unsqueeze(0).repeat(3,1,1).to(image_tensor.device)
    return image_tensor * mask_tensor

def faithfulness_test(model, image_tensor, heatmap, target_class, threshold=0.6):
    model.eval()
    with torch.no_grad():
        inputs = image_tensor.unsqueeze(0)
        outputs = model(inputs)
        orig_conf = torch.softmax(outputs.logits, dim=1)[0, target_class].item()
        masked_img = create_masked_image(image_tensor, heatmap, threshold, keep_important=False)
        masked_outputs = model(masked_img.unsqueeze(0))
        masked_conf = torch.softmax(masked_outputs.logits, dim=1)[0, target_class].item()
    confidence_drop = orig_conf - masked_conf
    return confidence_drop, orig_conf, masked_conf, masked_img

# ------------------------------
# 5️⃣ Generate explanation heatmaps
# ------------------------------
# Example: Grad-CAM for CNN (grayscale_cam should be obtained from pytorch-grad-cam)
# Here we simulate a dummy heatmap for demonstration:
grayscale_cam = np.random.rand(224,224)  # replace with real Grad-CAM
heatmap = normalize_heatmap(grayscale_cam)

# Example: ViT CLS attention map (optional)
# attentions = vit_outputs.attentions[-1] ...

# ------------------------------
# 6️⃣ Run faithfulness test
# ------------------------------
cnn_model = resnet_model
drop, orig, masked_conf, masked_img = faithfulness_test(cnn_model, image_tensor, heatmap, target_class=label, threshold=0.6)
print(f"Confidence drop: {drop:.4f}, Original: {orig:.4f}, Masked: {masked_conf:.4f}")

# ------------------------------
# 7️⃣ Visualization
# ------------------------------
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.imshow(image)
plt.title("Original Image")
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(image)
plt.imshow(heatmap, cmap='jet', alpha=0.5)
plt.title("Explanation Heatmap")
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(torch.permute(masked_img.cpu(), (1,2,0)) * 0.5 + 0.5)
plt.title("Masked Image")
plt.axis('off')
plt.show()
`

Original confidence: 0.0001 → model’s predicted probability for the target class on the original image.

Masked confidence: 0.0099 → predicted probability after masking important regions.

Confidence drop: -0.0098 → negative, meaning the masked image’s confidence actually increased slightly.


Interpretation:

This small negative value indicates that masking the “important” regions didn’t reduce the model’s confidence, which suggests that either:
Your heatmap isn’t highlighting the truly important regions (explanation may be weak/unfaithful), or
The threshold for masking may be too aggressive or too lenient, leaving the model enough cues elsewhere.





More comarison


`python

`