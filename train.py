import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from utils.dataloader import LabelMeSojaDataset
from utils.metrics import DiceLoss, calculate_iou
from torch.utils.data import DataLoader
from tqdm import tqdm


dataset = LabelMeSojaDataset(data_dir="data/train")
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
# 1. Dataset Customizado para Soja
class SojaDataset(Dataset):
    def __init__(self, img_dir, mask_dir, model_type="vit_b"):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_names = os.listdir(img_dir)
        self.transform = ResizeLongestSide(1024) # Tamanho padrão do SAM

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        # Carregar imagem e máscara
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        mask_path = os.path.join(self.mask_dir, self.img_names[idx].replace(".jpg", ".png"))
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.float32) # Binarizar

        # Preparar para o SAM
        input_image = self.transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image).permute(2, 0, 1).contiguous()
        
        # Simular um prompt (Bounding Box) a partir da máscara real
        y, x = np.where(mask > 0)
        bbox = np.array([np.min(x), np.min(y), np.max(x), np.max(y)])
        bbox_torch = self.transform.apply_boxes(bbox, image.shape[:2])
        
        return {
            "image": input_image_torch,
            "mask": torch.as_tensor(mask).unsqueeze(0),
            "bbox": torch.as_tensor(bbox_torch, dtype=torch.float)
        }

# 2. Configurações de Treino
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry["vit_b"](checkpoint="weights/sam_vit_b_01ec64.pth")
sam.to(device)

# Congelar tudo, exceto o Mask Decoder
for name, param in sam.named_parameters():
    if "mask_decoder" not in name:
        param.requires_grad = False

optimizer = optim.Adam(sam.mask_decoder.parameters(), lr=1e-5, weight_decay=0)
loss_fn = DiceLoss() # Para simplificar, ou use DiceLoss + BCE

# 3. Loop de Treinamento
dataset = LabelMeSojaDataset(data_dir="data/train")
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)


sam.train()
for epoch in range(10): 
    epoch_loss = 0
    
    # 1. Envolvemos o dataloader com o tqdm
    loop_lotes = tqdm(dataloader, desc=f"Época {epoch+1}/10")
    
    # 2. O for agora usa a nossa barra de progresso
    for batch in loop_lotes:
        imgs = batch["image"].to(device)
        gt_masks = batch["mask"].to(device)
        bboxes = batch["bbox"].to(device)

        # --- INÍCIO DA PARTE QUE HAVIA SUMIDO ---
        # Extrair características (Embeddings)
        with torch.no_grad():
            imgs_processadas = sam.preprocess(imgs) # Normaliza as cores
            image_embedding = sam.image_encoder(imgs_processadas)
            
        # Predição com o Decoder (Treinável)
        sparse_embeddings, dense_embeddings = sam.prompt_encoder(
            points=None, boxes=bboxes, masks=None
        )
        
        low_res_masks, iou_predictions = sam.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        # Calcular Perda (AQUI É ONDE A VARIÁVEL 'loss' NASCE!)
        upscaled_masks = sam.postprocess_masks(low_res_masks, (1024, 1024), (1024, 1024))
        loss = loss_fn(upscaled_masks, gt_masks)
        
        # Backpropagation (Aprender com o erro)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # --- FIM DA PARTE QUE HAVIA SUMIDO ---

        # Agora sim a variável 'loss' existe para ser somada!
        epoch_loss += loss.item()
        
        # Atualizamos a barra para mostrar a Loss caindo em tempo real
        loop_lotes.set_postfix(erro_atual=loss.item())
    
    print(f"\nResumo da Época {epoch+1} -> Erro médio total: {epoch_loss/len(dataloader)}\n")
    
    # Salva o checkpoint da época
    nome_arquivo = f"weights/sam_soja_epoca_{epoch+1}.pth"
    torch.save(sam.state_dict(), nome_arquivo)


# Salvar pesos ajustados
torch.save(sam.state_dict(), "weights/sam_soja_finetuned.pth")