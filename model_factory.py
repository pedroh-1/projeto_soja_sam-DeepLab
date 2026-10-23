import segmentation_models_pytorch as smp

def get_deeplab_model(num_classes=1):
    model = smp.DeepLabV3Plus(
        encoder_name="mobilenet_v2", 
        encoder_weights="imagenet", 
        in_channels=3, 
        classes=num_classes,
        activation=None 
    )
    return model
