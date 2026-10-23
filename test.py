import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from segment_anything import sam_model_registry

import matplotlib
matplotlib.use('Agg')

# Importando as ferramentas que já criamos
from utils.dataloader import LabelMeSojaDataset
from utils.metrics import calculate_iou, calculate_dice_coef

def evaluate_model():
    # 1. Configurações Iniciais
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # IMPORTANTE: Coloque aqui o nome exato do arquivo da época que você salvou
    peso_treinado = "weights/sam_soja_epoca_1.pth" 
    pasta_teste = "data/test"
    
    print(f"Carregando modelo para avaliação no {device}...")
    sam = sam_model_registry["vit_b"](checkpoint=peso_treinado)
    sam.to(device)
    
    # Desliga o "modo de aprendizado" e liga o "modo de prova"
    sam.eval() 

    # 2. Carregar o Dataset de Teste
    dataset_test = LabelMeSojaDataset(data_dir=pasta_teste)
    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)

    # Listas para guardar as notas de cada imagem
    todas_ious = []
    todos_dices = []
    
    print(f"Iniciando teste em {len(dataset_test)} imagens...\n")

    # 3. Loop de Inferência e Avaliação
    # O torch.no_grad() bloqueia a matemática de treino, economizando VRAM
    with torch.no_grad(): 
        for i, batch in enumerate(dataloader_test):
            imgs = batch["image"].to(device)
            gt_masks = batch["mask"].to(device)
            bboxes = batch["bbox"].to(device)

            # --- A Rede Neural Trabalhando ---
            imgs_processadas = sam.preprocess(imgs)
            image_embedding = sam.image_encoder(imgs_processadas)
            
            sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                points=None, boxes=bboxes, masks=None
            )
            
            low_res_masks, _ = sam.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            upscaled_masks = sam.postprocess_masks(low_res_masks, (1024, 1024), (1024, 1024))
            
            # --- Avaliação (Boletim Escolar) ---
            iou_score = calculate_iou(upscaled_masks, gt_masks)
            dice_score = calculate_dice_coef(upscaled_masks, gt_masks)
            
            todas_ious.append(iou_score)
            todos_dices.append(dice_score)

            print(f"Imagem {i+1} - IoU: {iou_score:.4f} | Dice: {dice_score:.4f}")

            # Salva um comparativo visual da primeira e da última imagem
            if i == 0 or i == len(dataset_test) - 1:
                salvar_grafico_visual(imgs[0], gt_masks[0], upscaled_masks[0], iou_score, i)

    # 4. Médias Finais
    iou_medio = np.mean(todas_ious)
    dice_medio = np.mean(todos_dices)
    
    print("\n" + "=" * 40)
    print("🎓 RESULTADOS FINAIS DO TESTE:")
    print(f"IoU Médio:  {iou_medio:.4f} ({iou_medio*100:.2f}%)")
    print(f"Dice Médio: {dice_medio:.4f} ({dice_medio*100:.2f}%)")
    print("=" * 40)
    
    # Gera o gráfico de linhas e salva na pasta
    gerar_grafico_metricas(todas_ious, todos_dices)

# --- Funções Auxiliares para Plotar os Gráficos ---

def salvar_grafico_visual(imagem_tensor, mascara_real, mascara_prevista, iou, indice):
    """Gera uma imagem dividida em 3 partes: Original, Gabarito e Predição"""
    # Preparar tensores para virarem imagens
    img = imagem_tensor.permute(1, 2, 0).cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min()) # Normalizar cores
    
    mask_real = mascara_real.squeeze().cpu().numpy()
    mask_pred = torch.sigmoid(mascara_prevista).squeeze().cpu().numpy() > 0.5
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("1. Imagem Original")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(img)
    plt.imshow(mask_real, alpha=0.5, cmap='Greens')
    plt.title("2. Seu Polígono (Gabarito)")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(img)
    plt.imshow(mask_pred, alpha=0.5, cmap='Reds')
    plt.title(f"3. IA - Segmentação SAM (IoU: {iou:.3f})")
    plt.axis('off')
    
    plt.tight_layout()
    nome_foto = f"resultado_visual_{indice}.png"
    plt.savefig(nome_foto)
    print(f"-> Foto comparativa salva como: {nome_foto}")
    plt.close()

def gerar_grafico_metricas(ious, dices):
    """Gera o gráfico de linhas mostrando a constância do modelo"""
    plt.figure(figsize=(10, 6))
    plt.plot(ious, marker='o', label='IoU', color='dodgerblue')
    plt.plot(dices, marker='s', label='Coeficiente Dice', color='darkorange')
    
    # Linhas de média
    plt.axhline(y=np.mean(ious), color='dodgerblue', linestyle='--', alpha=0.5)
    plt.axhline(y=np.mean(dices), color='darkorange', linestyle='--', alpha=0.5)
    
    plt.title("Desempenho da Segmentação por Imagem (Dataset de Teste)", fontsize=14)
    plt.xlabel("Índice da Imagem Analisada", fontsize=12)
    plt.ylabel("Score (0.0 a 1.0)", fontsize=12)
    plt.ylim(0, 1.1)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig("grafico_metricas_geral.png", dpi=300)
    print("-> Gráfico de métricas salvo como: grafico_metricas_geral.png")

if __name__ == "__main__":
    evaluate_model()