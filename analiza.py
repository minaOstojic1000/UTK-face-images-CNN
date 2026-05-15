import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

dataset_path = Path("./utk_ethnicity")

classes = ['asian', 'black', 'indian', 'other', 'white']

# =====================
# HISTOGRAM PO KLASAMA
# =====================
brojevi = []

for klasa in classes:
    folder = os.path.join(dataset_path, klasa)
    n = len(os.listdir(folder))
    brojevi.append(n)
    print(f"{klasa}: {n} slika")

print(f"\nUkupno: {sum(brojevi)} slika")

plt.figure(figsize=(8, 5))
plt.bar(classes, brojevi, color='steelblue', edgecolor='black')
plt.title("Broj uzoraka po klasama")
plt.xlabel("Klasa")
plt.ylabel("Broj slika")
plt.tight_layout()
plt.savefig("histogram_klasa.png")
print("Histogram sacuvan kao histogram_klasa.png")

# ==============================
# PO JEDAN PRIMJERAK IZ KLASE
# ==============================
fig, axes = plt.subplots(1, 5, figsize=(15, 4))

for i, klasa in enumerate(classes):
    folder = os.path.join(dataset_path, klasa)
    prva_slika = os.path.join(folder, os.listdir(folder)[0])
    img = mpimg.imread(prva_slika)
    axes[i].imshow(img)
    axes[i].set_title(klasa)
    axes[i].axis('off')

plt.suptitle("Po jedan primjerak iz svake klase", fontsize=14)
plt.tight_layout()
plt.savefig("primjerci_po_klasama.png")
print("Slika sacuvana kao primjerci_po_klasama.png")