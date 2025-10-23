import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

def plot_curves(train_losses, val_losses, val_accs):
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label="Train Loss", marker='o')
    plt.plot(val_losses, label="Val Loss", marker='o')
    plt.legend(); plt.title("Loss Curves")

    plt.subplot(1,2,2)
    plt.plot(val_accs, label="Val Accuracy", marker='o', color='green')
    plt.legend(); plt.title("Validation Accuracy")
    plt.show()


def visualize_embeddings(model, data, labels, max_samples=3000):
    model.eval()
    with torch.no_grad():
        emb = model.gat1(data.x.to("cpu"), data.edge_index.to("cpu")).cpu().numpy()
    idx = np.random.choice(emb.shape[0], min(max_samples, emb.shape[0]), replace=False)
    tsne = TSNE(n_components=2, random_state=42)
    emb2d = tsne.fit_transform(emb[idx])
    plt.scatter(emb2d[:,0], emb2d[:,1], c=labels[idx], cmap='coolwarm', s=5)
    plt.title("t-SNE of GAT Embeddings")
    plt.show()
