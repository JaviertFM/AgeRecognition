# ===== IMPORTAR LIBRERÍAS =====
if __name__ == '__main__':
    from fastai.vision.all import *
    import torch
    from sklearn.metrics import mean_absolute_error, confusion_matrix, ConfusionMatrixDisplay, classification_report
    import matplotlib.pyplot as plt

    # ===== PARÁMETROS =====
    DATASET_PATH = Path('dataset_rangos_10_balanced')  # Ruta al dataset
    IMG_SIZE = (224, 224)  # Tamaño de entrada
    BATCH_SIZE = 32        # Tamaño del batch
    EPOCHS = 15            # Número de épocas
    LR = 1e-3              # Learning rate inicial

    # ===== CUDA CHECK =====
    print("CUDA Available: ", torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ===== CARGAR DATOS =====
    # Configurar DataBlock
    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),  # División train/val
        get_y=parent_label,
        item_tfms=Resize(IMG_SIZE),  # Redimensionar imágenes
        batch_tfms=aug_transforms(  # Aumentación de datos
            do_flip=True,
            max_rotate=10.0,  # Rotación máxima en grados
            max_zoom=1.2,  # Zoom máximo
            max_warp=0.2,  # Deformación geométrica
            max_lighting=0.4,  # Controla brillo y contraste (reemplazo de brightness y contrast)
            p_lighting=0.75  # Probabilidad de aplicar cambios de iluminación
        )
    )

    # Crear Dataloaders
    dls = dblock.dataloaders(DATASET_PATH, bs=BATCH_SIZE, num_workers=0)  # num_workers=0 para Windows

    # Mostrar ejemplos del dataset
    dls.show_batch(max_n=9, figsize=(8, 8))

    # ===== CREAR EL MODELO =====
    learn = vision_learner(
        dls,
        arch=models.efficientnet_b4,  # Modelo EfficientNetB4 preentrenado
        metrics=[accuracy, mae],           # Métrica: Precisión
        loss_func=CrossEntropyLossFlat()
    ).to_fp16()  # Entrenamiento en precisión mixta (fp16)

    # ===== BÚSQUEDA DEL LEARNING RATE =====
    lr_min, lr_steep = learn.lr_find(suggest_funcs=(minimum, steep))
    print(f"Learning Rate recomendado: {lr_min:.6f}, {lr_steep:.6f}")

    # ===== ENTRENAMIENTO =====
    # Fine-tuning con early stopping
    learn.fine_tune(EPOCHS, base_lr=LR)

    # ===== EVALUACIÓN =====
    # Obtener predicciones en el conjunto de validación
    preds, targs = learn.get_preds()
    pred_labels = preds.argmax(dim=1)

    # ===== MÉTRICAS PERSONALIZADAS =====
    # 1. MAE (Mean Absolute Error)
    mae = mean_absolute_error(targs, pred_labels)
    print(f"Mean Absolute Error (MAE): {mae:.4f}")

    # 2. Matriz de confusión
    cm = confusion_matrix(targs, pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dls.vocab)
    disp.plot(cmap='Blues', xticks_rotation='vertical')
    plt.title('Confusion Matrix')
    plt.show()

    # 3. Clasificación detallada
    print(classification_report(targs, pred_labels, target_names=dls.vocab))

    # ===== GUARDAR MODELO =====
    learn.save('efficientnet_b4_balanced')

    # ===== GRAFICAR MÉTRICAS =====
    # Precisión y Pérdida por Época
    plt.figure(figsize=(12, 4))

    # Precisión
    plt.subplot(1, 2, 1)
    plt.plot(learn.recorder.metrics[0], label='Train Accuracy')
    plt.plot(learn.recorder.metrics[1], label='Validation Accuracy')
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Pérdida
    plt.subplot(1, 2, 2)
    plt.plot(learn.recorder.losses, label='Train Loss')
    plt.plot(learn.recorder.values[:, 1], label='Validation Loss')
    plt.title('Loss per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()
