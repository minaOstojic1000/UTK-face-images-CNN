import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score
from pathlib import Path

dataset_path = Path("./utk_ethnicity")

IMG_SIZE = (96, 96)
BATCH_SIZE = 64

def ucitavanje_podataka():
    # =====================
    # UČITAVANJE PODATAKA
    # =====================
    # Prvo učitavamo cijeli dataset
    full_dataset = image_dataset_from_directory(
        dataset_path,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=123
    )

    classes = full_dataset.class_names
    num_classes = len(classes)
    print(f'Dataset ima sledece klase: {classes}')
    print(f'Broj klasa: {num_classes}')

    # Ukupan broj batcheva
    total_batches = tf.data.experimental.cardinality(full_dataset).numpy()
    print(f'Ukupno batcheva: {total_batches}')

    train_size = int(0.70 * total_batches)
    val_size = int(0.15 * total_batches)
    test_size = total_batches - train_size - val_size

    Xtrain = full_dataset.take(train_size)
    Xval = full_dataset.skip(train_size).take(val_size)
    Xtest = full_dataset.skip(train_size + val_size).take(test_size)

    print(f'Trening batcheva: {train_size}')
    print(f'Validacioni batcheva: {val_size}')
    print(f'Test batcheva: {test_size}')

    return Xtrain, Xval, Xtest, num_classes, classes

def class_weighting(Xtrain, num_classes):
    # =====================
    # CLASS WEIGHTING
    # =====================
    labels = np.concatenate([y for x, y in Xtrain], axis=0)

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.array(range(num_classes)),
        y=labels
    )

    class_weight_dict = dict(enumerate(class_weights))
    print("Tezine klasa:", class_weight_dict)

    return class_weight_dict

def preprocesiranje(Xtrain, Xval, Xtest):
    # =====================
    # PREPROCESIRANJE
    # =====================
    normalization_layer = tf.keras.layers.Rescaling(1. / 255)

    Xtrain = Xtrain.map(lambda x, y: (normalization_layer(x), y))
    Xval = Xval.map(lambda x, y: (normalization_layer(x), y))
    Xtest = Xtest.map(lambda x, y: (normalization_layer(x), y))

    # Augmentacija - samo na trening skupu
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomTranslation(0.1, 0.1)
    ])

    Xtrain = Xtrain.map(lambda x, y: (data_augmentation(x, training=True), y))

    # Optimizacija učitavanja
    AUTOTUNE = tf.data.AUTOTUNE
    Xtrain = Xtrain.prefetch(buffer_size=AUTOTUNE)
    Xval = Xval.prefetch(buffer_size=AUTOTUNE)
    Xtest = Xtest.prefetch(buffer_size=AUTOTUNE)

    print("Preprocesiranje zavrseno!")

    return Xtrain, Xval, Xtest

def transfer_learning(img_size, num_classes):
    base_model = MobileNetV2(
        input_shape=(img_size[0], img_size[1], 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss=SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    return model

def train_model(model, Xtrain, Xval, class_weight_dict):
    es = EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    checkpoint = ModelCheckpoint(
        'best_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )

    print("Treniranje modela...")
    history = model.fit(
        Xtrain,
        validation_data=Xval,
        epochs=30,
        callbacks=[es, checkpoint],
        class_weight=class_weight_dict,
        verbose=1
    )

    return history
def prikaz_performansi(history):
    plt.figure(figsize=(12, 5))

    plt.subplot(121)
    plt.plot(history.history['accuracy'], label='train accuracy')
    plt.plot(history.history['val_accuracy'], label='val accuracy')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(122)
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.title('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('performanse.png')
    print("Grafik sacuvan kao performanse.png")

def evaluacija_modela(model, Xtrain, Xval, Xtest, classes):
    for skup, naziv in [(Xtrain, 'Trening'), (Xtest, 'Test')]:
        y_true = np.array([])
        y_pred = np.array([])

        for img, lab in skup:
            y_true = np.concatenate([y_true, lab.numpy()])
            y_pred = np.concatenate([y_pred, np.argmax(model.predict(img, verbose=0), axis=1)])

        print(f"\n--- {naziv} skup ---")
        print(f"Tacnost (Accuracy):     {100 * accuracy_score(y_true, y_pred):.2f}%")
        print(f"Preciznost (Precision): {100 * precision_score(y_true, y_pred, average='weighted'):.2f}%")
        print(f"Osjetljivost (Recall):  {100 * recall_score(y_true, y_pred, average='weighted'):.2f}%")
        print(f"F1-skor:                {100 * f1_score(y_true, y_pred, average='weighted'):.2f}%")

        cm = confusion_matrix(y_true, y_pred, normalize='true')
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        disp.plot(cmap='Blues')
        plt.title(f"Matrica konfuzije - {naziv} skup")
        plt.tight_layout()
        plt.savefig(f"matrica_konfuzije_{naziv.lower()}.png")
        print(f"Matrica konfuzije sacuvana kao matrica_konfuzije_{naziv.lower()}.png")

def prikaz_primjera(model, Xtest, classes):
    dobre_slike, lose_slike = [], []
    dobre_labele, lose_labele = [], []
    dobre_predikcije, lose_predikcije = [], []

    for img_batch, lab_batch in Xtest:
        predikcije = np.argmax(model.predict(img_batch, verbose=0), axis=1)
        for i in range(len(lab_batch)):
            slika = img_batch[i].numpy()
            true = int(lab_batch[i].numpy())
            pred = int(predikcije[i])
            if true == pred and len(dobre_slike) < 5:
                dobre_slike.append(slika)
                dobre_labele.append(true)
                dobre_predikcije.append(pred)
            elif true != pred and len(lose_slike) < 5:
                lose_slike.append(slika)
                lose_labele.append(true)
                lose_predikcije.append(pred)
        if len(dobre_slike) == 5 and len(lose_slike) == 5:
            break

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle("Dobro i lose klasifikovane slike", fontsize=14)

    for i in range(5):
        axes[0, i].imshow(dobre_slike[i])
        axes[0, i].set_title(f"T: {classes[dobre_labele[i]]}\nP: {classes[dobre_predikcije[i]]}", color='green', fontsize=8)
        axes[0, i].axis('off')

        axes[1, i].imshow(lose_slike[i])
        axes[1, i].set_title(f"T: {classes[lose_labele[i]]}\nP: {classes[lose_predikcije[i]]}", color='red', fontsize=8)
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig("primjeri_klasifikacije.png")
    print("Slika sacuvana kao primjeri_klasifikacije.png")

# =====================
# MAIN
# =====================
Xtrain, Xval, Xtest, num_classes, classes = ucitavanje_podataka()
class_weight_dict = class_weighting(Xtrain, num_classes)
Xtrain, Xval, Xtest = preprocesiranje(Xtrain, Xval, Xtest)
model = transfer_learning(IMG_SIZE, num_classes)
# model.summary()
# history = train_model(model, Xtrain, Xval, class_weight_dict)
# prikaz_performansi(history)
best_model = tf.keras.models.load_model('best_model.keras')
evaluacija_modela(best_model, Xtrain, Xval, Xtest, classes)
prikaz_primjera(best_model, Xtest, classes)