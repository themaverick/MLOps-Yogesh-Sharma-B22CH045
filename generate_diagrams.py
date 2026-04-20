import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

def create_architecture_diagram(output_path):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Helper to draw a box
    def draw_box(x, y, w, h, text, color='lightblue'):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3", 
                                      linewidth=2, edgecolor='navy', facecolor=color)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=10, fontweight='bold')

    # Draw components
    draw_box(40, 90, 20, 8, "User Query", 'lightyellow')
    
    # Retrieval Path
    draw_box(10, 70, 20, 8, "Vector Search\n(FAISS)", 'skyblue')
    draw_box(70, 70, 20, 8, "Keyword Search\n(BM25)", 'skyblue')
    
    # Arrows from Query
    ax.annotate('', xy=(20, 78), xytext=(45, 90), arrowprops=dict(arrowstyle="->", color='gray'))
    ax.annotate('', xy=(80, 78), xytext=(55, 90), arrowprops=dict(arrowstyle="->", color='gray'))

    # Fusion
    draw_box(40, 50, 20, 8, "Reciprocal Rank\nFusion (RRF)", 'lightgreen')
    ax.annotate('', xy=(45, 58), xytext=(25, 70), arrowprops=dict(arrowstyle="->", color='gray'))
    ax.annotate('', xy=(55, 58), xytext=(75, 70), arrowprops=dict(arrowstyle="->", color='gray'))

    # Reranker
    draw_box(40, 30, 20, 8, "Cross-Encoder\nReranker", 'orange')
    ax.annotate('', xy=(50, 38), xytext=(50, 50), arrowprops=dict(arrowstyle="->", color='gray'))

    # Output
    draw_box(40, 10, 20, 8, "Final Results", 'lightgrey')
    ax.annotate('', xy=(50, 18), xytext=(50, 30), arrowprops=dict(arrowstyle="->", color='gray'))

    plt.title("System Architecture: Hybrid Search & Reranking", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Architecture diagram saved to {output_path}")

def create_drift_diagram(output_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Simulate high-dimensional embeddings in 2D
    np.random.seed(42)
    corpus = np.random.normal(0, 1, (100, 2))
    queries = np.random.normal(3, 1, (20, 2)) # Shifted to show drift
    
    ax.scatter(corpus[:, 0], corpus[:, 1], alpha=0.3, label='Indexed Corpus', color='blue')
    ax.scatter(queries[:, 0], queries[:, 1], alpha=0.5, label='Recent queries', color='orange')
    
    c_centroid = np.mean(corpus, axis=0)
    q_centroid = np.mean(queries, axis=0)
    
    ax.scatter([c_centroid[0]], [c_centroid[1]], color='darkblue', s=100, marker='*', label='Corpus Centroid')
    ax.scatter([q_centroid[0]], [q_centroid[1]], color='darkred', s=100, marker='X', label='Query Centroid')
    
    # Arrow for drift
    ax.annotate('Drift (Cosine Distance)', xy=(q_centroid[0], q_centroid[1]), 
                xytext=(c_centroid[0], c_centroid[1]),
                arrowprops=dict(arrowstyle="<->", color='red', lw=2))
    
    ax.set_title("Embedding Drift Detection Concept", fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Drift diagram saved to {output_path}")

def create_contract_diagram(output_path):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis('off')

    def draw_box(x, y, w, h, text, color='white'):
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='black', facecolor=color)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=9)

    draw_box(5, 15, 15, 10, "Raw Data\n(ArXiv API)")
    draw_box(30, 10, 20, 20, "Data Contract\n(Pydantic)", color='mistyrose')
    draw_box(60, 15, 15, 10, "Preprocessed\nRecords")
    draw_box(85, 15, 12, 10, "Index\n(FAISS)")

    # Arrows
    ax.annotate('', xy=(30, 20), xytext=(20, 20), arrowprops=dict(arrowstyle="->"))
    ax.annotate('', xy=(60, 20), xytext=(50, 20), arrowprops=dict(arrowstyle="->"))
    ax.annotate('', xy=(85, 20), xytext=(75, 20), arrowprops=dict(arrowstyle="->"))

    plt.title("Data Ingestion Pipeline with Contract Validation", fontsize=12)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Contract diagram saved to {output_path}")

if __name__ == "__main__":
    output_dir = "./diagrams"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    create_architecture_diagram(os.path.join(output_dir, "arch_diag.png"))
    create_drift_diagram(os.path.join(output_dir, "drift_diag.png"))
    create_contract_diagram(os.path.join(output_dir, "contract_diag.png"))
    
    print(f"\nAll diagrams have been saved to: {os.path.abspath(output_dir)}")
