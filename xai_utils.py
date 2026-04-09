import torch
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification
import numpy as np
from PIL import Image

def normalize_label(label_name):
    """
    Normalize a label so style and artist mappings work consistently.
    """
    return label_name.replace("_", " ").replace("-", " ").strip().lower()

# Art style to era mapping
STYLE_TO_ERA = {
    "abstract expressionism": "Modern Art",
    "action painting": "Modern Art",
    "analytical cubism": "Modern Art",
    "art nouveau": "Modern Art",
    "art nouveau (modern)": "Modern Art",
    "baroque": "Baroque",
    "color field painting": "Modern Art",
    "contemporary realism": "Contemporary",
    "cubism": "Modern Art",
    "early renaissance": "Renaissance",
    "expressionism": "Modern Art",
    "fauvism": "Modern Art",
    "high renaissance": "Renaissance",
    "impressionism": "Impressionism",
    "mannerism late renaissance": "Renaissance",
    "mannerism (late renaissance)": "Renaissance",
    "minimalism": "Modern Art",
    "naive art primitivism": "Contemporary",
    "new realism": "Modern Art",
    "northern renaissance": "Renaissance",
    "pointillism": "Post-Impressionism",
    "pop art": "Modern Art",
    "post impressionism": "Post-Impressionism",
    "realism": "Realism",
    "rococo": "Rococo",
    "romanticism": "Romanticism",
    "symbolism": "Symbolism",
    "synthetic cubism": "Modern Art",
    "ukiyo e": "Ukiyo-e",
}

ARTIST_TO_ERA = {
    "unknown artist": "Unknown Era",
    "boris kustodiev": "Modern Art",
    "camille pissarro": "Impressionism",
    "childe hassam": "Impressionism",
    "claude monet": "Impressionism",
    "edgar degas": "Impressionism",
    "eugene boudin": "Impressionism",
    "gustave dore": "Romanticism",
    "ilya repin": "Realism",
    "ivan aivazovsky": "Romanticism",
    "ivan shishkin": "Realism",
    "john singer sargent": "Impressionism",
    "marc chagall": "Modern Art",
    "martiros saryan": "Modern Art",
    "nicholas roerich": "Modern Art",
    "pablo picasso": "Modern Art",
    "paul cezanne": "Post-Impressionism",
    "pierre auguste renoir": "Impressionism",
    "pyotr konchalovsky": "Impressionism",
    "raphael kirchner": "Modern Art",
    "rembrandt": "Baroque",
    "salvador dali": "Modern Art",
    "vincent van gogh": "Post-Impressionism",
    "hieronymus bosch": "Northern Renaissance",
    "leonardo da vinci": "Renaissance",
    "albrecht durer": "Northern Renaissance",
    "edouard cortes": "Modern Art",
    "sam francis": "Modern Art",
    "juan gris": "Modern Art",
    "lucas cranach the elder": "Renaissance",
    "paul gauguin": "Post-Impressionism",
    "konstantin makovsky": "Realism",
    "egon schiele": "Modern Art",
    "thomas eakins": "Realism",
    "gustave moreau": "Symbolism",
    "francisco goya": "Romanticism",
    "edvard munch": "Expressionism",
    "henri matisse": "Modern Art",
    "fra angelico": "Renaissance",
    "maxime maufra": "Modern Art",
    "jan matejko": "Romanticism",
    "mstislav dobuzhinsky": "Modern Art",
    "alfred sisley": "Impressionism",
    "mary cassatt": "Impressionism",
    "gustave loiseau": "Modern Art",
    "fernando botero": "Contemporary",
    "zinaida serebriakova": "Modern Art",
    "georges seurat": "Post-Impressionism",
    "isaac levitan": "Realism",
    "joaquin sorolla": "Impressionism",
    "jacek malczewski": "Symbolism",
    "berthe morisot": "Impressionism",
    "andy warhol": "Contemporary",
    "arkhip kuindzhi": "Realism",
    "niko pirosmani": "Modern Art",
    "james tissot": "Realism",
    "vasily polenov": "Realism",
    "valentin serov": "Impressionism",
    "pietro perugino": "Renaissance",
    "pierre bonnard": "Modern Art",
    "ferdinand hodler": "Symbolism",
    "bartolome esteban murillo": "Baroque",
    "giovanni boldini": "Realism",
    "henri martin": "Symbolism",
    "gustav klimt": "Modern Art",
    "vasily perov": "Realism",
    "odilon redon": "Symbolism",
    "tintoretto": "Renaissance",
    "gene davis": "Modern Art",
    "raphael": "Renaissance",
    "john henry twachtman": "Impressionism",
    "henri de toulouse lautrec": "Post-Impressionism",
    "antoine blanchard": "Realism",
    "david burliuk": "Modern Art",
    "camille corot": "Realism",
    "konstantin korovin": "Impressionism",
    "ivan bilibin": "Art Nouveau",
    "titian": "Renaissance",
    "maurice prendergast": "Post-Impressionism",
    "edouard manet": "Impressionism",
    "peter paul rubens": "Baroque",
    "aubrey beardsley": "Art Nouveau",
    "paolo veronese": "Renaissance",
    "joshua reynolds": "Neoclassicism",
    "kuzma petrov vodkin": "Modern Art",
    "gustave caillebotte": "Impressionism",
    "lucian freud": "Modern Art",
    "michelangelo": "Renaissance",
    "dante gabriel rossetti": "Romanticism",
    "felix vallotton": "Modern Art",
    "nikolay bogdanov belsky": "Realism",
    "georges braque": "Modern Art",
    "vasily surikov": "Realism",
    "fernand leger": "Modern Art",
    "konstantin somov": "Symbolism",
    "katsushika hokusai": "Ukiyo-e",
    "sir lawrence alma tadema": "Neoclassicism",
    "vasily vereshchagin": "Realism",
    "ernst ludwig kirchner": "Modern Art",
    "mikhail vrubel": "Symbolism",
    "orest kiprensky": "Romanticism",
    "william merritt chase": "Impressionism",
    "aleksey savrasov": "Realism",
    "hans memling": "Northern Renaissance",
    "amedeo modigliani": "Modern Art",
    "ivan kramskoy": "Realism",
    "utagawa kuniyoshi": "Ukiyo-e",
    "gustave courbet": "Realism",
    "william turner": "Romanticism",
    "theo van rysselberghe": "Post-Impressionism",
    "joseph wright": "Romanticism",
    "edward burne jones": "Symbolism",
    "koloman moser": "Art Nouveau",
    "viktor vasnetsov": "Romanticism",
    "anthony van dyck": "Baroque",
    "raoul dufy": "Modern Art",
    "frans hals": "Baroque",
    "hans holbein the younger": "Northern Renaissance",
    "ilya mashkov": "Modern Art",
    "henri fantin latour": "Symbolism",
    "m c escher": "Modern Art",
    "el greco": "Mannerism",
    "mikalojus ciurlionis": "Symbolism",
    "james mcneill whistler": "Impressionism",
    "karl bryullov": "Romanticism",
    "jacob jordaens": "Baroque",
    "thomas gainsborough": "Rococo",
    "eugene delacroix": "Romanticism",
    "canaletto": "Rococo",
}

_WIKIART_STYLE_NAMES = None
_WIKIART_STYLE_FALLBACK = [
    "Abstract_Expressionism",
    "Action_painting",
    "Analytical_Cubism",
    "Art_Nouveau_Modern",
    "Baroque",
    "Color_Field_Painting",
    "Contemporary_Realism",
    "Cubism",
    "Early_Renaissance",
    "Expressionism",
    "Fauvism",
    "High_Renaissance",
    "Impressionism",
    "Mannerism_Late_Renaissance",
    "Minimalism",
    "Naive_Art_Primitivism",
    "New_Realism",
    "Northern_Renaissance",
    "Pointillism",
    "Pop_Art",
    "Post_Impressionism",
    "Realism",
    "Rococo",
    "Romanticism",
    "Symbolism",
    "Synthetic_Cubism",
    "Ukiyo_e",
]

def _is_generic_label(label_name):
    """
    True when model labels look like placeholders such as LABEL_12.
    """
    if not isinstance(label_name, str):
        return True
    normalized = label_name.strip().lower()
    return normalized.startswith("label_") or normalized.startswith("class_") or normalized.isdigit()

def _get_wikiart_style_names():
    """
    Load WikiArt style names once and cache them for label fallback.
    """
    global _WIKIART_STYLE_NAMES
    if _WIKIART_STYLE_NAMES is not None:
        return _WIKIART_STYLE_NAMES

    try:
        dataset = load_dataset("huggan/wikiart", split="train", streaming=True)
        style_feature = dataset.features.get("style")
        if hasattr(style_feature, "names") and style_feature.names:
            _WIKIART_STYLE_NAMES = list(style_feature.names)
            return _WIKIART_STYLE_NAMES
    except Exception:
        pass

    _WIKIART_STYLE_NAMES = list(_WIKIART_STYLE_FALLBACK)
    return _WIKIART_STYLE_NAMES

def decode_prediction_label_with_source(model, pred_id):
    """
    Convert class index to a label and return which mapping source was used.
    """
    config = getattr(model, "config", None)
    if config is None:
        return f"label_{pred_id}", "fallback_literal"

    id2label = getattr(config, "id2label", {}) or {}
    raw_label = id2label.get(pred_id, id2label.get(str(pred_id)))
    if raw_label and not _is_generic_label(raw_label):
        return raw_label, "config_id2label"

    label2id = getattr(config, "label2id", {}) or {}
    inverse = {}
    for name, idx in label2id.items():
        if isinstance(idx, int):
            inverse[idx] = name
        elif isinstance(idx, str) and idx.isdigit():
            inverse[int(idx)] = name
    inverse_label = inverse.get(pred_id)
    if inverse_label and not _is_generic_label(inverse_label):
        return inverse_label, "config_label2id_inverse"

    # Manual fallback: use WikiArt's own style label names by index.
    style_names = _get_wikiart_style_names()
    if 0 <= pred_id < len(style_names):
        return style_names[pred_id], "manual_wikiart_styles"

    if raw_label:
        return str(raw_label), "config_id2label_generic"
    return f"label_{pred_id}", "fallback_literal"

def decode_prediction_label(model, pred_id):
    """
    Convert predicted class index to a human-readable label with robust fallbacks.
    """
    label, _ = decode_prediction_label_with_source(model, pred_id)
    return label

def get_era_from_style(style_name):
    """
    Map art style or artist label to historical era.
    """
    normalized = normalize_label(style_name)
    if normalized in STYLE_TO_ERA:
        return STYLE_TO_ERA[normalized]
    if normalized in ARTIST_TO_ERA:
        return ARTIST_TO_ERA[normalized]
    return "Unknown Era"

def load_wikiart_dataset(num_samples=200):
    """
    Loads a streaming portion of the huggan/wikiart dataset.
    """
    print("Loading Wikiart dataset...")
    dataset = load_dataset("huggan/wikiart", streaming=True)
    test_dataset = dataset["train"].take(num_samples)
    
    # Materialize explicitly into a list of PIL images
    images = [example["image"].convert("RGB") for example in test_dataset if example.get("image")]
    print(f"Loaded {len(images)} valid images.")
    return images

def get_random_wikiart_sample():
    """
    Returns a single random image and its style label from the wikiart dataset.
    Uses a more memory-efficient approach.
    """
    try:
        dataset = load_dataset("huggan/wikiart", split="train", streaming=True)
        import random
        
        # Use a smaller skip range to avoid memory issues
        skip_amount = random.randint(0, 100)  # Reduced from 1000 to 100
        
        # Skip a random amount to get a random-ish sample from streaming dataset
        iterator = iter(dataset)
        for _ in range(skip_amount):
            try:
                next(iterator)
            except StopIteration:
                # If we reach the end, start over
                iterator = iter(dataset)
                break
        
        # Get the next sample
        sample = next(iterator)
        # The dataset provides 'style' as an integer. We will return the raw sample.
        return sample["image"].convert("RGB"), sample["style"]
    except Exception as e:
        # Fallback: return a simple test image if dataset loading fails
        print(f"Warning: Could not load from dataset ({e}), using fallback")
        from PIL import Image
        import numpy as np
        
        # Create a simple colored square as fallback
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        fallback_img = Image.fromarray(img_array)
        return fallback_img, 0  # Return dummy style


def load_models_and_processors(device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Loads both models and their processors from Hugging Face.
    """
    print(f"Loading models to {device}...")

    vit_name = "sakhmatd/vit-wikiart-finetuned"
    resnet_name = "Iust1n2/resnet-18-finetuned-wikiart"

    # We need to set output_attentions=True to extract attention weights for ViT
    vit_model = AutoModelForImageClassification.from_pretrained(vit_name, output_attentions=True).to(device)
    vit_processor = AutoImageProcessor.from_pretrained(vit_name)

    resnet_model = AutoModelForImageClassification.from_pretrained(resnet_name).to(device)
    resnet_processor = AutoImageProcessor.from_pretrained(resnet_name)

    return (vit_model, vit_processor), (resnet_model, resnet_processor)

def preprocess_image(image, processor, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Preprocesses a PIL Image using the given feature extractor/processor.
    """
    inputs = processor(images=image, return_tensors="pt")
    # Move all input tensors to the target device
    return {k: v.to(device) for k, v in inputs.items()}
