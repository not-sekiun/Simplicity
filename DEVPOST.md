# Simplicity

Simplicity is a modern AI image detection model that uses a linear classifier trained on the image embeddings of a vision foundation model
to reliably detect AI images robustly across a range of adversarial transforms. Training data was augmented using the stated transforms to improve
in the wild robustness. All images were ran through the backbone model to obtain embeddings which the linear classifier was then trained on.

For more information on installation, usage, and setup see the demo video: https://www.youtube.com/watch?v=y0ximH3GmGM or the project page: https://github.com/not-sekiun/Simplicity

Tools: VS Code, git, `uv` (dependency + venv management).

Models: PE-Core-L (`timm/vit_pe_core_large_patch14_336.fb`), frozen. Raced against DINOv3-L and MetaCLIP2-H.

Libraries: PyTorch, torchvision, timm, HuggingFace `datasets` + `hub`, scikit-learn, pandas, NumPy, Pillow, tqdm; FastAPI + uvicorn for the demo.

Datasets: Tiny-GenImage (training + heldout), AIGC-Detection-Benchmark (OOD eval; a disjoint later slice for `train_ext`), SID_Set (reals), Unsplash (`wtcherr/unsplash_5k`), WildRF (arXiv:2406.09398), COCO val2017 + WildFake DALL·E Advanced (demo-val). Photoroom/midjourney-v6-recap, bitmind/nano-banana, OpenDatasets/dalle-3-dataset

Papers: "Simplicity Prevails" (arXiv:2602.01738) — the frozen-backbone + linear-probe recipe. UniversalFakeDetect, Ojha et al. CVPR 2023 — linear probes generalise where deep classifiers do not. WildRF, arXiv:2406.09398.
