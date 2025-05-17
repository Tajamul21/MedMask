import os
import numpy as np
import torch
from PIL import Image
from modeling.BaseModel import BaseModel
from modeling import build_model
from utilities.distributed import init_distributed
from utilities.arguments import load_opt_from_config_files
from utilities.constants import BIOMED_CLASSES
from inference_utils.inference import interactive_infer_image

opt = load_opt_from_config_files(["configs/biomedparse_inference.yaml"])
opt = init_distributed(opt)

pretrained_pth = 'pretrained/biomed_parse.pt'
model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()
with torch.no_grad():
    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(BIOMED_CLASSES + ["background"], is_eval=True)

# Prompts
input_dir = '/home/suhaib/Research/Drive/Datasets/BCD_INBreast/coco_uniform_ts/train2017'
embeddings_dir = '/home/suhaib/Research/Drive/Outputs/embeddings/train_inbreast'
prompts = ['malignant cancer in the breast']

os.makedirs(embeddings_dir, exist_ok=True)
for filename in os.listdir(input_dir):
    if filename.endswith('.png'):
        try:
            image_path = os.path.join(input_dir, filename)
            image = Image.open(image_path).convert('RGB')
            
            _, embedding = interactive_infer_image(model, image, prompts)
            
            embedding_filename = os.path.splitext(filename)[0] + '_embedding.npy'
            embedding_path = os.path.join(embeddings_dir, embedding_filename)
            np.save(embedding_path, embedding.cpu().numpy())  # Save embedding as .npy
            
            print(f"Embedding saved at: {embedding_path}")
        
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            import traceback
            traceback.print_exc()
            continue

print("Embedding export completed.")
