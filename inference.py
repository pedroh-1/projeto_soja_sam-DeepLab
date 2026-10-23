import numpy as np
import cv2
import torch
from segment_anything import sam_model_registry, SamPredictor

# --- Mantenha suas funções show_mask e show_points se quiser salvar imagens depois ---

def interactive_inference(image_path, model_type, checkpoint_path, device):
    print(f"Carregando modelo no {device}...")
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print("Erro: Imagem não encontrada!")
        return
    
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    # --- CONFIGURAÇÃO DA JANELA ---
    # Usamos WINDOW_NORMAL para permitir redimensionamento livre
    nome_janela = "Interacao Soja SAM - Clique em uma folha"
    cv2.namedWindow(nome_janela, cv2.WINDOW_NORMAL) 
    
    # Define um tamanho inicial razoável (ex: 1280x720)
    cv2.resizeWindow(nome_janela, 1280, 720)

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Captura o tamanho atual da janela onde o usuário clicou
            rect = cv2.getWindowImageRect(nome_janela)
            win_w, win_h = rect[2], rect[3]

            # Mapeia o clique da janela para a imagem real
            real_x = int(x * (image_bgr.shape[1] / win_w))
            real_y = int(y * (image_bgr.shape[0] / win_h))

            print(f"Clique na tela: {x},{y} -> Real na Imagem: {real_x},{real_y}")

            input_point = np.array([[real_x, real_y]])
            input_label = np.array([1])

            masks, scores, _ = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=False,
            )

            mask = masks[0]
            display_img = image_bgr.copy()
            overlay = np.zeros_like(display_img)
            overlay[mask] = [0, 255, 0]
            
            final_view = cv2.addWeighted(display_img, 0.7, overlay, 0.3, 0)
            cv2.circle(final_view, (real_x, real_y), 15, (0, 0, 255), -1)
            
            cv2.imshow(nome_janela, final_view)

    cv2.setMouseCallback(nome_janela, on_click)
    cv2.imshow(nome_janela, image_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    CONFIG = {
        "image_path": "data/test/aruco.jpg",
        "model_type": "vit_b",
        "checkpoint": "weights/sam_soja_epoca_8.pth", # Use o seu modelo treinado!
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

    interactive_inference(
        CONFIG["image_path"], 
        CONFIG["model_type"], 
        CONFIG["checkpoint"], 
        CONFIG["device"]
    )