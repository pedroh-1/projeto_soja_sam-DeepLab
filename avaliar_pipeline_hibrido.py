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

# Importando a fábrica do DeepLabV3+
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

# ==============================================================================
# FUNÇÕES MATEMÁTICAS (MÉTRICAS)
# ==============================================================================
def calcular_iou_dice(predicao, ground_truth):
    pred = predicao.astype(bool)
    gt = ground_truth.astype(bool)
    
    intersecao = np.logical_and(pred, gt).sum()
    uniao = np.logical_or(pred, gt).sum()
    
    iou = (intersecao / uniao) if uniao > 0 else (1.0 if pred.sum() == 0 else 0.0)
    
    soma_areas = pred.sum() + gt.sum()
    dice = (2. * intersecao / soma_areas) if soma_areas > 0 else 1.0
        
    return iou, dice

# ==============================================================================
# CARREGAMENTO DOS MODELOS
# ==============================================================================
def carregar_modelos(device):
    print("Carregando o SAM...")
    sam = sam_model_registry["vit_b"](checkpoint="weights/sam_soja_epoca_1.pth")
    sam.to(device=device)
    predictor = SamPredictor(sam)

    print("Carregando o DeepLabV3+...")
    # Usando 1 classe, conforme descobrimos no erro anterior
    deeplab = get_deeplab_model(num_classes=1) 
    
    # ATENÇÃO: COLOQUE AQUI O CAMINHO CORRETO PARA OS PESOS DO SEU DEEPLAB
    state_dict, image_size, threshold = carregar_checkpoint_deeplab(DEEPLAB_CHECKPOINT, device)
        
    deeplab.load_state_dict(state_dict)
    deeplab.to(device)
    deeplab.eval()
    
    return predictor, deeplab, image_size, threshold

# ==============================================================================
# PIPELINE DE AVALIAÇÃO EM LOTE COM LEITURA DE JSON (LABELME)
# ==============================================================================
def avaliar_dataset(pasta_teste, predictor, deeplab, device, image_size, threshold):
    lista_imagens = [f for f in os.listdir(pasta_teste) if f.lower().endswith('.jpg')]
    lista_imagens.sort()
    
    resultados_iou = []
    resultados_dice = []
    tamanho_deeplab = (image_size, image_size)
    transformacao = transforms.Compose([transforms.ToTensor()])

    print(f"\nIniciando avaliação de {len(lista_imagens)} imagens...\n")

    for nome_img in lista_imagens:
        caminho_img = os.path.join(pasta_teste, nome_img)
        nome_json = nome_img.replace('.jpg', '.json') 
        caminho_json = os.path.join(pasta_teste, nome_json)
        
        if not os.path.exists(caminho_json):
            print(f"[AVISO] Arquivo JSON não encontrado para {nome_img}. Pulando...")
            continue

        # 1. Lê a imagem
        image_bgr = cv2.imread(caminho_img)
        if image_bgr is None:
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        altura_orig, largura_orig = image_rgb.shape[:2]
        
        # 2. Lê o JSON do LabelMe
        with open(caminho_json, 'r') as f:
            dados_labelme = json.load(f)
            
        ground_truth_binario = np.zeros((altura_orig, largura_orig), dtype=np.uint8)
        for shape in dados_labelme['shapes']:
            pontos = np.array(shape['points'], dtype=np.int32)
            cv2.fillPoly(ground_truth_binario, [pontos], 1)

        # 3. Roda o DeepLab para achar o núcleo (com Sigmoid)
        img_redimensionada = cv2.resize(image_rgb, tamanho_deeplab)
        tensor_img = transformacao(img_redimensionada).unsqueeze(0).to(device)

        with torch.no_grad():
            predicao = deeplab(tensor_img)
            predicao_sigmoid = torch.sigmoid(predicao)
            mask_pequena = (predicao_sigmoid > threshold).squeeze().cpu().numpy()

        mask_deeplab = cv2.resize(mask_pequena.astype(np.uint8), (largura_orig, altura_orig), interpolation=cv2.INTER_NEAREST)

        # 4. Extrai Bounding Box
        contornos, _ = cv2.findContours(mask_deeplab, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contornos) == 0:
            print(f"[{nome_img}] Falha: Nenhuma folha detectada pelo DeepLab.")
            resultados_iou.append(0.0)
            resultados_dice.append(0.0)
            continue
            
        maior_contorno = max(contornos, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(maior_contorno)
        caixa_automatica = np.array([x, y, x + w, y + h])

        # 5. Roda o SAM guiado pela Caixa
        predictor.set_image(image_rgb)
        masks, _, _ = predictor.predict(
            point_coords=None, point_labels=None,
            box=caixa_automatica[None, :], multimask_output=False
        )
        mascara_final_sam = masks[0].astype(np.uint8)

        # 6. Calcula as Métricas
        iou, dice = calcular_iou_dice(mascara_final_sam, ground_truth_binario)
        resultados_iou.append(iou)
        resultados_dice.append(dice)
        print(f"[{nome_img}] IoU: {iou:.4f} | Dice: {dice:.4f}")

    return resultados_iou, resultados_dice

# ==============================================================================
# GERAÇÃO DOS GRÁFICOS
# ==============================================================================
def gerar_graficos(resultados_iou, resultados_dice):
    media_iou = np.mean(resultados_iou)
    media_dice = np.mean(resultados_dice)
    print(f"\n=====================================")
    print(f"RESULTADO FINAL: DEEPLAB + SAM")
    print(f"Média IoU: {media_iou:.4f} ({(media_iou*100):.2f}%)")
    print(f"Média Dice: {media_dice:.4f} ({(media_dice*100):.2f}%)")
    print(f"=====================================\n")

    # Gráfico de Barras
    plt.figure(figsize=(6, 5))
    barras = plt.bar(['IoU', 'Dice Score'], [media_iou, media_dice], color=['#ff7f0e', '#d62728']) # Cores quentes para diferenciar da U-Net
    plt.ylim(0, 1.1)
    plt.title('Desempenho Médio - Híbrido (DeepLab + SAM)')
    plt.ylabel('Score')
    
    for barra in barras:
        altura = barra.get_height()
        plt.text(barra.get_x() + barra.get_width()/2., altura + 0.02,
                 f"{altura:.4f}", ha='center', va='bottom', fontsize=11, fontweight='bold')
                 
    plt.savefig('deeplab_sam_grafico_medias.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Gráfico de Dispersão
    plt.figure(figsize=(10, 5))
    x = np.arange(len(resultados_iou))
    plt.plot(x, resultados_iou, label='IoU', marker='o', linestyle='-', color='#ff7f0e', alpha=0.7)
    plt.plot(x, resultados_dice, label='Dice Score', marker='s', linestyle='--', color='#d62728', alpha=0.7)
    plt.ylim(0, 1.1)
    plt.title('Desempenho por Imagem - Híbrido (DeepLab + SAM)')
    plt.xlabel('Índice da Imagem de Teste')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig('deeplab_sam_grafico_dispersao.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Gráficos salvos: 'deeplab_sam_grafico_medias.png' e 'deeplab_sam_grafico_dispersao.png'")

if __name__ == "__main__":
    device = selecionar_device()
    
    # --- AJUSTES NECESSÁRIOS AQUI ---
    PASTA_TESTE = "/home/vicom02/Downloads/Projeto_Vicom_Pedro/projeto_soja_sam + DeepLab/data/test/" 
    
    predictor, deeplab, image_size, threshold = carregar_modelos(device)
    
    lista_iou, lista_dice = avaliar_dataset(PASTA_TESTE, predictor, deeplab, device, image_size, threshold)
    
    if len(lista_iou) > 0:
        gerar_graficos(lista_iou, lista_dice)