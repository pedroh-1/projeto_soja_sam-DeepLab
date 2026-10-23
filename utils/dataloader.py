import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from segment_anything.utils.transforms import ResizeLongestSide
import torch.nn.functional as F

class LabelMeSojaDataset(Dataset):
    """
    Dataset customizado para carregar imagens e polígonos do LabelMe 
    e prepará-los para o treinamento do Segment Anything Model (SAM).
    """
    def __init__(self, data_dir, target_size=1024):
        """
        Args:
            data_dir (str): Caminho para a pasta onde estão as imagens (.jpg/.png) e os .json juntos.
            target_size (int): Tamanho padrão que o Image Encoder do SAM espera (1024).
        """
        self.data_dir = data_dir
        self.transform = ResizeLongestSide(target_size)
        
        # Filtra apenas os arquivos de imagem (ignorando os .json nesta lista)
        self.image_files = [
            f for f in os.listdir(data_dir) 
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 1. Nomes e Caminhos
        img_name = self.image_files[idx]
        img_path = os.path.join(self.data_dir, img_name)
        
        # O arquivo do LabelMe tem o mesmo nome da imagem, mas com extensão .json
        json_name = os.path.splitext(img_name)[0] + '.json'
        json_path = os.path.join(self.data_dir, json_name)

        # 2. Carregar Imagem
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_h, original_w = image.shape[:2]

        # 3. Gerar Máscara Binária a partir do Polígono do LabelMe
        mask = np.zeros((original_h, original_w), dtype=np.uint8)
        
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            for shape in data['shapes']:
                # Extrai os pontos do polígono (x, y)
                points = np.array(shape['points'], dtype=np.int32)
                
                # Desenha o polígono preenchido com 1 (folha de soja)
                cv2.fillPoly(mask, [points], 1)

        # 4. Extrair a Bounding Box (Caixa delimitadora) para usar como Prompt
        y_indices, x_indices = np.where(mask > 0)
        
        if len(x_indices) > 0 and len(y_indices) > 0:
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            # Adicionar uma pequena margem (ruído) pode ajudar o modelo a generalizar melhor
            bbox = np.array([x_min, y_min, x_max, y_max])
        else:
            # Fallback caso a imagem não tenha anotação (bbox cobre a imagem toda)
            bbox = np.array([0, 0, original_w, original_h])

        # 5. Aplicar Transformações específicas do SAM (ResizeLongestSide)
        input_image = self.transform.apply_image(image)
        
        # A máscara precisa de um reshape temporário para a transformação
        mask_expanded = np.expand_dims(mask, axis=-1)
        input_mask = self.transform.apply_image(mask_expanded).squeeze()
        
        input_bbox = self.transform.apply_boxes(bbox, (original_h, original_w))

        # 6. Converter para Tensores do PyTorch
        image_tensor = torch.as_tensor(input_image, dtype=torch.float32).permute(2, 0, 1).contiguous()
        mask_tensor = torch.as_tensor(input_mask, dtype=torch.float32).unsqueeze(0) # (1, H, W)
        bbox_tensor = torch.as_tensor(input_bbox[0], dtype=torch.float32)

        # 7. Aplicar Padding (Preenchimento) para garantir 1024x1024
        # Pegamos o tamanho atual (altura e largura)
        h, w = image_tensor.shape[-2:]
        
        # Calculamos quanto falta para chegar em 1024
        padh = 1024 - h
        padw = 1024 - w
        
        # Aplicamos o padding (esquerda, direita, cima, baixo)
        image_tensor = F.pad(image_tensor, (0, padw, 0, padh))
        mask_tensor = F.pad(mask_tensor, (0, padw, 0, padh))

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "bbox": bbox_tensor,
            "original_size": (original_h, original_w)
        }

# --- Teste Rápido do DataLoader ---
if __name__ == "__main__":
    # Suponha que suas imagens e jsons estejam na pasta "data/train"
    dataset = LabelMeSojaDataset(data_dir="data/train")
    
    # Criar o DataLoader do PyTorch
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    print(f"Total de imagens no dataset: {len(dataset)}")
    
    # Pegar um lote (batch) para verificar
    for batch in dataloader:
        print("Formato da Imagem:", batch["image"].shape) # Esperado: [Batch, 3, H, W]
        print("Formato da Máscara:", batch["mask"].shape) # Esperado: [Batch, 1, H, W]
        print("Formato da BBox:", batch["bbox"].shape)    # Esperado: [Batch, 4]
        break