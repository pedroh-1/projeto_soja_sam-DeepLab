import os
import cv2
import json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from segment_anything import sam_model_registry, SamPredictor

# ==============================================================================
# 1. IMPORTAÇÃO DA SUA ARQUITETURA DEEPLABV3+
# Substitua 'arquivo_deeplab' pelo nome do seu arquivo .py
# Exemplo: from modelo_deeplab import DeepLabV3Plus
# ==============================================================================
from model_factory import get_deeplab_model

DEEPLAB_CHECKPOINT = "weights_DeepLab/best_deeplab_soja.pth"

def carregar_checkpoint_deeplab(caminho_checkpoint, device):
    checkpoint = torch.load(caminho_checkpoint, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        image_size = checkpoint.get("image_size", 256)
        threshold = checkpoint.get("best_threshold", 0.5)
        best_iou = checkpoint.get("best_iou")

        print(
            "Checkpoint DeepLab detectado "
            f"(image_size={image_size}, threshold={threshold}, best_iou={best_iou})."
        )
        return state_dict, int(image_size), float(threshold)

    if isinstance(checkpoint, dict):
        checkpoint.pop("mask_values", None)

    return checkpoint, 256, 0.5

def selecionar_device():
    if not torch.cuda.is_available():
        return "cpu"

    if torch.version.hip:
        props = torch.cuda.get_device_properties(0)
        arch = getattr(props, "gcnArchName", "")
        if arch == "gfx1102":
            print(
                "GPU ROCm detectada como gfx1102, mas esta instalacao do PyTorch "
                "nao tem biblioteca rocBLAS compativel. Usando CPU."
            )
            return "cpu"

    return "cuda"

def carregar_modelos(device):
    print(f"Carregando modelos no device: {device}...")
    
    # --- Carrega o SAM (Vit-B Fine-Tuned) ---
    sam_checkpoint = "weights/sam_soja_epoca_1.pth" 
    sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    # --- Carrega o DeepLabV3+ ---
    print("Injetando pesos do DeepLabV3+...")
    
    # ATENÇÃO: Ajuste os parâmetros de inicialização conforme a sua classe!
    # Assumindo 2 classes (Fundo e Folha) assim como foi na U-Net
    deeplab = get_deeplab_model(num_classes=1) 
    
    # Coloque o nome correto do seu arquivo de pesos (.pth) do DeepLabV3+
    state_dict, image_size, threshold = carregar_checkpoint_deeplab(DEEPLAB_CHECKPOINT, device)
        
    deeplab.load_state_dict(state_dict)
    deeplab.to(device)
    deeplab.eval()
    
    print("Todos os modelos foram carregados com sucesso!\n")
    return predictor, deeplab, image_size, threshold

def processar_imagem_automatica(caminho_imagem, predictor, deeplab, device, image_size, threshold):
    print(f"Processando: {os.path.basename(caminho_imagem)}")
    
    image_bgr = cv2.imread(caminho_imagem)
    if image_bgr is None:
        print("Erro ao carregar a imagem. Verifique o caminho.")
        return
    
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    altura_orig, largura_orig = image_rgb.shape[:2]

    # ==========================================================================
    # 2. PASSO DEEPLAB: INFERÊNCIA COM MÓDULO ASPP
    # ==========================================================================
    print("-> 1/3: Executando inferência no DeepLabV3+...")
    
    # Mantenha a mesma resolução usada no treinamento do DeepLab
    tamanho_entrada = (image_size, image_size) 
    img_redimensionada = cv2.resize(image_rgb, tamanho_entrada)
    
    transformacao = transforms.Compose([transforms.ToTensor()])
    tensor_img = transformacao(img_redimensionada).unsqueeze(0).to(device)

    with torch.no_grad():
        predicao = deeplab(tensor_img)
        
        # Como temos apenas 1 classe, usamos a função Sigmoid
        predicao_sigmoid = torch.sigmoid(predicao)
        
        # Usa o limiar salvo no checkpoint quando disponivel.
        mask_pequena = (predicao_sigmoid > threshold).squeeze().cpu().numpy()

    mask_deeplab = cv2.resize(
        mask_pequena.astype(np.uint8), 
        (largura_orig, altura_orig), 
        interpolation=cv2.INTER_NEAREST
    )

    # ==========================================================================
    # 3. EXTRAÇÃO AUTOMÁTICA DA BOUNDING BOX
    # ==========================================================================
    print("-> 2/3: Extraindo Bounding Box...")
    mask_uint8 = (mask_deeplab * 255).astype(np.uint8)
    
    contornos, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contornos) == 0:
        print("O DeepLab não detectou folha nesta imagem. Pulando o SAM.")
        return
    
    maior_contorno = max(contornos, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(maior_contorno)
    caixa_automatica = np.array([x, y, x + w, y + h])

    # ==========================================================================
    # 4. PASSO SAM: REFINAMENTO
    # ==========================================================================
    print("-> 3/3: Executando SAM com o prompt da caixa...")
    predictor.set_image(image_rgb)
    
    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=caixa_automatica[None, :],
        multimask_output=False
    )
    
    mascara_final_sam = masks[0]
    print(f"-> Sucesso! Confiança final do SAM: {scores[0]:.3f}\n")

    gerar_comparativo_visual(image_rgb, mask_deeplab, caixa_automatica, mascara_final_sam)
    gerar_comparativo_mascaras(caminho_imagem, image_rgb, mascara_final_sam)

def carregar_mascara_anotada(caminho_imagem, altura, largura):
    """Carrega a anotacao LabelMe correspondente a imagem e retorna uma mascara binaria."""
    caminho_json = os.path.splitext(caminho_imagem)[0] + ".json"
    if not os.path.exists(caminho_json):
        print(f"[AVISO] Anotacao manual nao encontrada: '{caminho_json}'")
        return None

    with open(caminho_json, "r") as arquivo_json:
        dados_labelme = json.load(arquivo_json)

    mascara_manual = np.zeros((altura, largura), dtype=np.uint8)
    for shape in dados_labelme.get("shapes", []):
        pontos = np.array(shape.get("points", []), dtype=np.int32)
        if len(pontos) >= 3:
            cv2.fillPoly(mascara_manual, [pontos], 1)

    return mascara_manual

def calcular_iou(mascara_predita, mascara_manual):
    """Calcula IoU entre duas mascaras binarias."""
    pred = mascara_predita.astype(bool)
    manual = mascara_manual.astype(bool)

    intersecao = np.logical_and(pred, manual).sum()
    uniao = np.logical_or(pred, manual).sum()

    if uniao == 0:
        return 1.0 if pred.sum() == 0 else 0.0

    return intersecao / uniao

def gerar_comparativo_mascaras(caminho_imagem, img, mask_sam):
    """Gera uma imagem comparando a mascara do modelo com a anotacao manual."""
    altura, largura = img.shape[:2]
    mascara_manual = carregar_mascara_anotada(caminho_imagem, altura, largura)
    if mascara_manual is None:
        return

    mascara_sam = mask_sam.astype(np.uint8)
    iou = calcular_iou(mascara_sam, mascara_manual)

    sobreposicao = np.zeros((altura, largura, 3), dtype=np.uint8)
    sobreposicao[np.logical_and(mascara_sam == 1, mascara_manual == 1)] = [255, 255, 0]
    sobreposicao[np.logical_and(mascara_sam == 1, mascara_manual == 0)] = [0, 255, 0]
    sobreposicao[np.logical_and(mascara_sam == 0, mascara_manual == 1)] = [255, 0, 0]

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    axs[0].imshow(img)
    axs[0].imshow(mascara_sam, alpha=0.55, cmap="Greens")
    axs[0].set_title("Mascara gerada pelo modelo", fontsize=14)
    axs[0].axis("off")

    axs[1].imshow(img)
    axs[1].imshow(mascara_manual, alpha=0.55, cmap="Blues")
    axs[1].set_title("Mascara anotada manualmente", fontsize=14)
    axs[1].axis("off")

    axs[2].imshow(img)
    axs[2].imshow(sobreposicao, alpha=0.65)
    axs[2].set_title(f"Comparacao - IoU: {iou:.4f}", fontsize=14)
    axs[2].axis("off")

    plt.tight_layout()
    nome_saida = "comparativo_mascaras_deeplab.png"
    plt.savefig(nome_saida, dpi=300)
    print(f"-> Comparativo de mascaras salvo como: '{nome_saida}'")
    print("   Legenda: amarelo = intersecao | verde = so modelo | vermelho = so anotacao")
    print(f"   IoU: {iou:.4f}")
    plt.close()

def gerar_comparativo_visual(img, mask_inicial, bbox, mask_sam):
    """Gera um gráfico lado a lado do processo híbrido"""
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    axs[0].imshow(img)
    axs[0].set_title("1. Imagem Original", fontsize=14)
    axs[0].axis('off')
    
    axs[1].imshow(img)
    axs[1].imshow(mask_inicial, alpha=0.5, cmap='Blues') # Usando azul para diferenciar do script da U-Net
    x_min, y_min, x_max, y_max = bbox
    rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                         fill=False, edgecolor='blue', linewidth=3)
    axs[1].add_patch(rect)
    axs[1].set_title("2. DeepLabV3+ + Bounding Box", fontsize=14)
    axs[1].axis('off')
    
    axs[2].imshow(img)
    axs[2].imshow(mask_sam, alpha=0.5, cmap='Greens')
    axs[2].set_title("3. SAM (Bordas Refinadas)", fontsize=14)
    axs[2].axis('off')
    
    plt.tight_layout()
    nome_saida = "resultado_hibrido_deeplab.png"
    plt.savefig(nome_saida, dpi=300)
    print(f"-> Imagem salva como: '{nome_saida}'")
    plt.close()

if __name__ == "__main__":
    device = selecionar_device()
    
    try:
        predictor, deeplab, image_size, threshold = carregar_modelos(device)
        
        # Ajuste para uma imagem real da sua pasta de teste
        imagem_teste = "/home/vicom02/Downloads/Projeto_Vicom_Pedro/projeto_soja_sam + DeepLab/data/test/IMG_20260322_131221.jpg" 
        
        processar_imagem_automatica(imagem_teste, predictor, deeplab, device, image_size, threshold)
        
    except ImportError as e:
        print("\n[ERRO] Falha ao importar a arquitetura.")
        print("Verifique a linha 10 com o nome correto do seu arquivo do DeepLab.")
        print(f"Detalhe do erro: {e}")
