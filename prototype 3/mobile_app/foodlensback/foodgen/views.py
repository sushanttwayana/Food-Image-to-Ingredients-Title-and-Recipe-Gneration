import os
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
from .predictor import predict_and_generate  # your model function

class RecipePredictView(APIView):
    parser_classes = [MultiPartParser]

    def post(self, request, *args, **kwargs):
        image_file = request.data.get('image')

        if not image_file:
            return Response({'error': 'No image uploaded'}, status=400)

        # Save image temporarily
        image_path = f'media/{image_file.name}'
        with open(image_path, 'wb+') as f:
            for chunk in image_file.chunks():
                f.write(chunk)

        # Model paths
        INGREDIENT_MODEL_PATH = './best_ingredient_model (3).pth'
        VOCAB_PATH = './ingredient_vocabulary.json'
        BLIP_PROCESSOR_PATH = './blip_processor'
        BLIP_MODEL_PATH = './blip_finetuned_title_gen.pth'
        T5_MODEL_PATH = './t5_recipe_instr_ep10.pth'


# Updated model paths to match your Streamlit app
        INGREDIENT_MODEL_PATH = './best_ingredient_model_new.pth'
        VOCAB_PATH = './ingredient_vocabulary_new.json'
        FOOD_MAP_PATH = './food_map.json'
        BLIP_PROCESSOR_PATH = './blip_processor'  # Not used but keeping for compatibility
        BLIP_MODEL_PATH = './blip_finetuned_title_gen.pth'
        T5_MODEL_PATH = './t5_base_recipe_instr.pth'  # Default to T5-small
        MODEL_TYPE = "T5-base"  # You can make this configurable via request parameter

        try:
            predicted_ingredients, title, recipe_instructions = predict_and_generate(
    image_path,
    INGREDIENT_MODEL_PATH,
    VOCAB_PATH,
    BLIP_PROCESSOR_PATH,
    BLIP_MODEL_PATH,
    T5_MODEL_PATH,
    food_map_path=FOOD_MAP_PATH,
    # model_type=MODEL_TYPE  # Remove this line to auto-detect
)
        except Exception as e:
            # Clean up
            if os.path.exists(image_path):
                os.remove(image_path)
            print(f"Fatal prediction error: {str(e)}")
            return Response({'error': str(e)}, status=500)

        # Clean up uploaded image
        if os.path.exists(image_path):
            os.remove(image_path)

        return Response({
            'ingredients': predicted_ingredients,
            'title': title,
            'instructions': recipe_instructions
        })
