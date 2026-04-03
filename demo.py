"""
Demo Script — Video Content Gap Analyzer
=========================================

Run this script to demo the plugin to judges. It loads a safety
behaviours video dataset from HuggingFace Hub, launches the FiftyOne
App, and walks you through the operator workflow.

Usage:
    export TWELVELABS_API_KEY="your_key_here"
    python demo.py
"""

import fiftyone as fo
from fiftyone.utils.huggingface import load_from_hub

DATASET_NAME = "safe-unsafe-behaviours-demo"
MAX_SAMPLES = 40

# Expected safety categories to check for gaps
EXPECTED_CATEGORIES = (
    "person falling, fire evacuation, forklift moving, "
    "hard hat compliance, slip and fall, chemical spill"
)


def main():
    # ------------------------------------------------------------------
    # 1. Load dataset (reuses if already downloaded)
    # ------------------------------------------------------------------
    if fo.dataset_exists(DATASET_NAME):
        print(f"Reusing existing dataset '{DATASET_NAME}'")
        dataset = fo.load_dataset(DATASET_NAME)
    else:
        print(f"Loading {MAX_SAMPLES} samples from Voxel51/Safe_and_Unsafe_Behaviours ...")
        dataset = load_from_hub(
            "Voxel51/Safe_and_Unsafe_Behaviours",
            max_samples=MAX_SAMPLES,
            name=DATASET_NAME,
        )
    print(f"Dataset: {dataset.name} — {len(dataset)} samples\n")

    # ------------------------------------------------------------------
    # 2. Launch FiftyOne App
    # ------------------------------------------------------------------
    session = fo.launch_app(dataset)

    # ------------------------------------------------------------------
    # 3. Print demo walkthrough for the presenter
    # ------------------------------------------------------------------
    print("=" * 60)
    print("  VIDEO CONTENT GAP ANALYZER — DEMO WALKTHROUGH")
    print("=" * 60)
    print()
    print("The FiftyOne App is now open in your browser.")
    print()
    print("STEP 1: Run the Analysis")
    print("  - Press the backtick key (`) to open the operator menu")
    print("  - Search for 'Analyze Coverage'")
    print("  - Set parameters:")
    print("      num_clusters     = 0   (auto-select via silhouette)")
    print("      expected_categories = " + EXPECTED_CATEGORIES)
    print("      use_pegasus      = True")
    print("      max_samples      = 0   (process all)")
    print("  - Click Execute and watch the 4-stage progress bar")
    print()
    print("STEP 2: View the Coverage Panel")
    print("  - Click the '+' tab in the App panel bar")
    print("  - Select 'Coverage Map'")
    print("  - See the UMAP scatter plot with clusters color-coded")
    print("  - Red diamonds = missing categories (gaps)")
    print("  - Click any point to jump to that sample")
    print()
    print("STEP 3: Read the Gap Report")
    print("  - Press ` again, search for 'Show Gap Report'")
    print("  - See the coverage score, sparse clusters, and missing categories")
    print()
    print("STEP 4: Explore the Data")
    print("  - Filter by tag 'sparse_cluster' to see underrepresented videos")
    print("  - Sort by 'centroid_distance' to find outliers")
    print("  - Group by 'cluster_label' to browse by topic")
    print()
    print("=" * 60)
    print("  Suggested talking points for judges:")
    print("  • Data curation > model complexity (CV4Smalls insight)")
    print("  • Structural gaps (sparse/isolated) + category gaps")
    print("  • Marengo 3.0 multimodal embeddings (visual + audio)")
    print("  • Pegasus 1.2 generates human-readable cluster labels")
    print("  • Coverage score quantifies dataset completeness")
    print("=" * 60)
    print()

    session.wait()


if __name__ == "__main__":
    main()
