import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    """
    Calcula a Dice Loss. 
    Ideal para datasets desbalanceados (muito fundo, pouco objeto).
    A fórmula base é: 1 - (2 * Interseção / (Soma das Áreas))
    """
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth # Evita divisão por zero

    def forward(self, inputs, targets):
        # O SAM geralmente cospe 'logits' (valores sem ativação). 
        # Precisamos passar uma Sigmoid para transformar em probabilidades (0 a 1).
        inputs = torch.sigmoid(inputs)
        
        # Achatar (flatten) os tensores para 1D
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Calcular a interseção
        intersection = (inputs * targets).sum()                            
        
        # Calcular o coeficiente Dice
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)  
        
        # Como queremos minimizar o erro no treino, retornamos 1 - Dice
        return 1 - dice


def calculate_iou(preds, targets, threshold=0.5, smooth=1e-5):
    """
    Calcula o IoU (Intersection over Union), também conhecido como Índice de Jaccard.
    Usado principalmente para validação e teste.
    """
    # Aplicar Sigmoid e binarizar as predições baseado no limiar (threshold)
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    
    # Achatar os tensores
    preds = preds.view(-1)
    targets = targets.view(-1)
    
    # Interseção: onde a predição e o alvo são 1
    intersection = (preds * targets).sum()
    
    # União: Soma de tudo menos a interseção
    union = preds.sum() + targets.sum() - intersection
    
    # Calcular IoU
    iou = (intersection + smooth) / (union + smooth)
    
    # Retorna o valor numérico puro para facilitar o log/print
    return iou.item()


def calculate_dice_coef(preds, targets, threshold=0.5, smooth=1e-5):
    """
    Calcula o Coeficiente Dice (Apenas a métrica, não a Loss) para validação.
    """
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    
    preds = preds.view(-1)
    targets = targets.view(-1)
    
    intersection = (preds * targets).sum()
    dice = (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
    
    return dice.item()