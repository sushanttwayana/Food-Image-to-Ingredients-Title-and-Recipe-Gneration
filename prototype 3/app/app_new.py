import streamlit as st
import torch
import json
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
from torchvision import models
import os
from transformers import BlipProcessor, BlipForConditionalGeneration, T5Tokenizer, T5ForConditionalGeneration
import tempfile
import re

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
st.write(f"Using device: {device}")

# ===== Model and utility code =====

class IngredientVocabularyBuilder:
    def __init__(self):
        self.word_to_idx = {}
        self.idx_to_word = {}
    
    def load_vocabulary(self, vocab_file):
        try:
            with open(vocab_file, 'r') as f:
                vocab = json.load(f)
            self.word_to_idx = vocab['word_to_idx']
            self.idx_to_word = {int(k): v for k, v in vocab['idx_to_word'].items()}
            st.info(f"Vocabulary size: {len(self.word_to_idx)}")
            return self.word_to_idx, self.idx_to_word
        except Exception as e:
            st.error(f"Error loading vocabulary file {vocab_file}: {str(e)}")
            return None, None

class IngredientPredictor(nn.Module):
    def __init__(self, num_ingredients, num_classes=52, backbone='efficientnet_b0'):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=None)
        feature_dim = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        self.classifier_class = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        self.classifier_ingredients = nn.Sequential(
            nn.Linear(feature_dim + num_classes, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_ingredients),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        features = self.backbone(x)
        class_logits = self.classifier_class(features)
        class_probs = torch.softmax(class_logits, dim=1)
        combined_features = torch.cat([features, class_probs], dim=1)
        ingredient_probs = self.classifier_ingredients(combined_features)
        return class_logits, ingredient_probs

def load_food_map(food_map_path):
    """Load the food mapping JSON file"""
    try:
        with open(food_map_path, 'r') as f:
            food_map = json.load(f)
        return food_map
    except Exception as e:
        st.error(f"Error loading food map: {e}")
        return {}

def normalize_text(text):
    """Normalize text for better matching (lowercase, remove special chars)"""
    return re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower().strip())

def find_best_food_match(title, food_map):
    """Find the best matching food item from the food map based on title"""
    normalized_title = normalize_text(title)
    
    # Direct match first
    for food_name in food_map.keys():
        normalized_food = normalize_text(food_name)
        if normalized_food == normalized_title:
            return food_name
    
    # Partial match - check if title contains food name or vice versa
    best_match = None
    max_match_score = 0
    
    for food_name in food_map.keys():
        normalized_food = normalize_text(food_name)
        if normalized_food in normalized_title or normalized_title in normalized_food:
            overlap = len(set(normalized_food.split()) & set(normalized_title.split()))
            if overlap > max_match_score:
                max_match_score = overlap
                best_match = food_name
    
    return best_match

def enhance_ingredients_with_food_map(predicted_ingredients, title, food_map):
    """Enhance predicted ingredients by adding missing key ingredients from food map"""
    enhanced_ingredients = predicted_ingredients.copy()
    added_ingredients = []
    
    matched_food = find_best_food_match(title, food_map)
    
    if matched_food:
        required_ingredients_str = food_map[matched_food]
        required_ingredients = [ing.strip().lower() for ing in required_ingredients_str.split(',')]
        normalized_predicted = [ing.lower().replace('_', ' ') for ing in predicted_ingredients]
        
        for req_ing in required_ingredients:
            found = False
            for pred_ing in normalized_predicted:
                if req_ing in pred_ing or pred_ing in req_ing:
                    found = True
                    break
            if not found:
                formatted_ingredient = req_ing.replace(' ', '_')
                enhanced_ingredients.append(formatted_ingredient)
                added_ingredients.append(formatted_ingredient)
    
    return enhanced_ingredients, matched_food, added_ingredients

def load_image_for_ingredients(image_path):
    """Load and preprocess the input image for ingredient prediction"""
    try:
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)
        transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        transformed = transform(image=image_np)
        image_tensor = transformed['image'].float().unsqueeze(0)
        return image_tensor
    except Exception as e:
        st.error(f"Error loading image for ingredient prediction: {str(e)}")
        return None

def predict_ingredients(image_path, model_path, vocab_path, food_map_path, threshold=0.5, device=device):
    """Predict ingredients for a given image"""
    vocab_builder = IngredientVocabularyBuilder()
    word_to_idx, idx_to_word = vocab_builder.load_vocabulary(vocab_path)
    if word_to_idx is None or idx_to_word is None:
        return None
    
    try:
        model = IngredientPredictor(num_ingredients=len(word_to_idx), num_classes=52).to(device)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        checkpoint_state_dict = checkpoint['model_state_dict']
        
        model_state_dict = model.state_dict()
        new_state_dict = {}
        
        for key, value in checkpoint_state_dict.items():
            if key == 'classifier_class.3.weight' or key == 'classifier_class.3.bias':
                new_state_dict[key] = value
            elif key == 'classifier_ingredients.0.weight':
                expected_input_dim = 1280 + 52
                current_input_dim = 1280 + 52
                if value.size(1) != current_input_dim:
                    st.warning(f"Adjusting classifier_ingredients.0.weight from {value.size()} to match new input dim")
                    new_weight = value[:, :current_input_dim]
                    new_state_dict[key] = new_weight
                else:
                    new_state_dict[key] = value
            elif key == 'classifier_ingredients.3.weight':
                if value.size(0) != len(word_to_idx):
                    st.warning(f"Adjusting classifier_ingredients.3.weight from {value.size()} to {len(word_to_idx)}")
                    new_weight = torch.zeros(len(word_to_idx), value.size(1))
                    min_size = min(value.size(0), len(word_to_idx))
                    new_weight[:min_size] = value[:min_size]
                    new_state_dict[key] = new_weight
                else:
                    new_state_dict[key] = value
            elif key == 'classifier_ingredients.3.bias':
                if value.size(0) != len(word_to_idx):
                    st.warning(f"Adjusting classifier_ingredients.3.bias from {value.size()} to {len(word_to_idx)}")
                    new_bias = torch.zeros(len(word_to_idx))
                    min_size = min(value.size(0), len(word_to_idx))
                    new_bias[:min_size] = value[:min_size]
                    new_state_dict[key] = new_bias
                else:
                    new_state_dict[key] = value
            else:
                new_state_dict[key] = value
        
        model.load_state_dict(new_state_dict, strict=False)
        model.eval()
        
        best_threshold = checkpoint.get('best_threshold', threshold)
        st.info(f"Using threshold for ingredient prediction: {best_threshold}")
        
        image_tensor = load_image_for_ingredients(image_path)
        if image_tensor is None:
            return None
        
        with torch.no_grad():
            image_tensor = image_tensor.to(device)
            _, ingredient_probs = model(image_tensor)
            ingredient_probs = ingredient_probs.cpu().numpy()[0]
        
        predicted_indices = np.where(ingredient_probs > best_threshold)[0]
        predicted_ingredients = [idx_to_word[idx] for idx in predicted_indices if idx in idx_to_word]
        predicted_ingredients = [ing for ing in predicted_ingredients if ing not in ['<PAD>', '<UNK>']]
        
        return predicted_ingredients
    except Exception as e:
        st.error(f"Error in ingredient prediction: {str(e)}")
        return None

def generate_title(image_path, blip_processor_path, blip_model_path, device=device):
    """Generate a title for the given image using the fine-tuned BLIP model"""
    try:
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        model.load_state_dict(torch.load(blip_model_path, map_location=device, weights_only=False))
        model.to(device)
        model.eval()
        
        image = Image.open(image_path).convert("RGB")
        inputs = processor(image, return_tensors="pt").to(device)
        
        generated_ids = model.generate(
            **inputs,
            max_length=20,
            num_beams=5,
            length_penalty=1.0,
            early_stopping=True
        )
        title = processor.decode(generated_ids[0], skip_special_tokens=True).capitalize()
        return title
    except Exception as e:
        st.error(f"Error generating title: {str(e)}")
        return None

def generate_recipe_instructions(image_path, predicted_ingredients, title, t5_model_path, device=device):
    """Generate recipe instructions using the fine-tuned T5 model"""
    try:
        t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
        t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")
        t5_model.load_state_dict(torch.load(t5_model_path, map_location=device, weights_only=False))
        t5_model.to(device)
        t5_model.eval()
        
        ingredients_str = ", ".join(predicted_ingredients) if predicted_ingredients else "unknown ingredients"
        t5_input_text = f"title: {title}. ingredients: {ingredients_str}. instructions:"
        t5_inputs = t5_tokenizer(
            t5_input_text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        t5_inputs = {key: val.to(device) for key, val in t5_inputs.items()}
        
        with torch.no_grad():
            t5_outputs = t5_model.generate(
                input_ids=t5_inputs['input_ids'],
                attention_mask=t5_inputs['attention_mask'],
                max_length=256,
                num_beams=4,
                length_penalty=1.0,
                early_stopping=True
            )
        recipe_instructions = t5_tokenizer.decode(t5_outputs[0], skip_special_tokens=True)
        return recipe_instructions
    except Exception as e:
        st.error(f"Error generating recipe instructions: {str(e)}")
        return None

# ===== Streamlit UI =====

st.title("Food Image to Recipe Generator")
st.markdown("Upload a food image to get predicted ingredients (enhanced with food mapping), a dish title, and a full cooking recipe.")

with st.sidebar:
    st.header("Model File Paths")
    INGREDIENT_MODEL_PATH = st.text_input("Ingredient model (.pth)", value="best_ingredient_model_new.pth")
    VOCAB_PATH = st.text_input("Vocabulary (.json)", value="ingredient_vocabulary_new.json")
    FOOD_MAP_PATH = st.text_input("Food mapping (.json)", value="food_map.json")
    BLIP_PROCESSOR_PATH = st.text_input("BLIP processor (not used, default applied)", value="blip_processor", disabled=True)
    BLIP_MODEL_PATH = st.text_input("BLIP title model (.pth)", value="blip_finetuned_title_gen.pth")
    T5_MODEL_PATH = st.text_input("T5 recipe model (.pth)", value="t5_small_recipe_best.pth")

uploaded_file = st.file_uploader("Choose a food image...", type=["jpg", "jpeg", "png"], accept_multiple_files=False)

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.read())
        image_path = tmp_file.name

    st.image(Image.open(image_path), caption="Uploaded Image", width=300)

    if st.button("Predict and Generate Recipe"):
        with st.spinner("Processing image..."):
            # Check if model files exist
            for path, name in [
                (INGREDIENT_MODEL_PATH, "Ingredient model file"),
                (VOCAB_PATH, "Vocabulary file"),
                (FOOD_MAP_PATH, "Food map file"),
                (BLIP_MODEL_PATH, "BLIP model file"),
                (T5_MODEL_PATH, "T5 model file")
            ]:
                if not os.path.exists(path):
                    st.error(f"{name} {path} does not exist.")
                    break
            else:
                # Step 1: Generate title
                title = generate_title(image_path, BLIP_PROCESSOR_PATH, BLIP_MODEL_PATH, device=device)
                if title and title.lower().strip() == "pizza dough":
                    title = "Pizza"
                
                st.subheader("Generated Dish Title")
                st.write(title if title else "Unknown")

                # Step 2: Predict ingredients
                predicted_ingredients = predict_ingredients(
                    image_path, INGREDIENT_MODEL_PATH, VOCAB_PATH, FOOD_MAP_PATH, device=device
                )
                
                if not predicted_ingredients:
                    st.warning("Could not predict ingredients. Please check model files or image.")
                else:
                    # Step 3: Enhance ingredients with food map
                    food_map = load_food_map(FOOD_MAP_PATH)
                    if food_map:
                        enhanced_ingredients, matched_food, added_ingredients = enhance_ingredients_with_food_map(
                            predicted_ingredients, title, food_map
                        )
                        final_ingredients = enhanced_ingredients
                    else:
                        final_ingredients = predicted_ingredients
                    
                    st.subheader("Predicted Ingredients")
                    st.write(", ".join(final_ingredients) if final_ingredients else "None")

                    # Step 4: Generate recipe instructions
                    recipe_instructions = generate_recipe_instructions(
                        image_path, final_ingredients, title, T5_MODEL_PATH, device=device
                    )
                    st.subheader("Recipe Instructions")
                    st.write(recipe_instructions if recipe_instructions else "Could not generate recipe.")

    # Clean up temporary file
    if 'image_path' in locals():
        try:
            os.unlink(image_path)
        except:
            pass