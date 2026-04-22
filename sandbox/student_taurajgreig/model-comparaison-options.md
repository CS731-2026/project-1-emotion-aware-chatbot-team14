Comparison categories for RAF-DB

We organize the comparison into three categories:

1) Dataset-specific FER models
These are FER models already established on RAF-DB in the literature. They are useful because they give a dataset-grounded reference point rather than a generic baseline. RAF-DB itself is a widely used in-the-wild facial-expression benchmark with around 30K images, so it is reasonable to anchor one category around models already reported on this dataset.  [oai_citation:0‡whdeng.cn](https://www.whdeng.cn/RAF/model1.html?utm_source=chatgpt.com)

Recommended models:
- POSTER++
  - Good choice for the main dataset-specific reference
  - Reported 92.21% on RAF-DB
  - Also reports 8.4G FLOPs and 43.7M parameters
  - Reference: Mao et al., “POSTER++: A simpler and stronger facial expression recognition network.”  [oai_citation:1‡arXiv](https://arxiv.org/abs/2301.12149?utm_source=chatgpt.com)
- POSTER
  - Older but still strong RAF-DB benchmark model
  - Reported 92.05% on RAF-DB
  - Reference: Zheng et al., “POSTER: A Pyramid Cross-Fusion Transformer Network for Facial Expression Recognition.”  [oai_citation:2‡arXiv](https://arxiv.org/abs/2301.12149?utm_source=chatgpt.com)
- APViT
  - Good transformer-style RAF-DB benchmark used in later comparison tables
  - Reported at 91.98% on RAF-DB in MPA-FER’s comparison table
  - Reference path: cited in the ICCV 2025 MPA-FER benchmark table.  [oai_citation:3‡openaccess.thecvf.com](https://openaccess.thecvf.com/content/ICCV2025/papers/Ma_Multimodal_Prompt_Alignment_for_Facial_Expression_Recognition_ICCV_2025_paper.pdf?utm_source=chatgpt.com)

Best pick for this category:
- POSTER++

2) General-purpose classification backbones
These are standard image-classification architectures adapted to FER through transfer learning. They are not FER-specific, which makes them strong and defensible baselines. This category answers the question: “How well do standard computer-vision backbones perform when adapted to RAF-DB?” The category is academically clean because it separates generic transfer-learning baselines from FER-specialized models. A modern public RAF-DB training repository also supports standard families such as ResNet, EfficientNet, ViT, Swin, and others, which reinforces that these are legitimate comparison backbones.  [oai_citation:4‡whdeng.cn](https://www.whdeng.cn/RAF/model1.html?utm_source=chatgpt.com)

Recommended models:
- ResNet-18
  - Best simple baseline
  - Easiest to train, debug, and explain
  - Best first model for a course project
  - Reference: He et al., “Deep Residual Learning for Image Recognition.” Standard residual backbone used throughout vision literature.
- ResNet-50
  - Stronger and deeper generic baseline
  - Useful if you want a “small CNN vs larger CNN” comparison
  - Reference: same ResNet paper by He et al.
- EfficientNet-B0
  - Compact and efficient general-purpose classifier
  - Better efficiency/accuracy tradeoff than many older CNN baselines
  - Reference: Tan and Le, “EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.”
- MobileNetV3
  - Efficient hardware-aware image classifier
  - Useful if you want a lighter generic architecture
  - Reference: Howard et al., “Searching for MobileNetV3.”

Best pick for this category:
- ResNet-18

3) General FER models not tied to RAF-DB
These are FER-specific models, but not chosen because they are specifically “RAF-DB models.” Instead, they are proposed as more general FER methods and then evaluated on RAF-DB among other datasets. This category avoids overlap with category 1 because the point is not “already established on RAF-DB,” but rather “FER-specific method proposed more generally.” This is a much cleaner category than “high-performance FER models,” which overlaps too heavily with dataset-specific literature models.

Recommended models:
- MPA-FER
  - Strong modern FER-specific model
  - Reported 93.74% on RAF-DB
  - Also reports 68.89% on AffectNet-7
  - Uses multimodal prompt alignment with a frozen CLIP backbone
  - Reports very small learnable parameter budgets: 0.218 MB with ViT-B/16 and 0.443 MB with ViT-L/14
  - Reference: Ma et al., “Multimodal Prompt Alignment for Facial Expression Recognition.” ICCV 2025.  [oai_citation:5‡openaccess.thecvf.com](https://openaccess.thecvf.com/content/ICCV2025/papers/Ma_Multimodal_Prompt_Alignment_for_Facial_Expression_Recognition_ICCV_2025_paper.pdf?utm_source=chatgpt.com)
- Ada-DF
  - FER-specific method based on adaptive label-distribution fusion
  - Good if you want a model with a strong methodological FER story
  - Evaluated on RAF-DB, AffectNet, and SFEW
  - Reference: Liu et al., “Ada-DF: An Adaptive Label Distribution Fusion Network For Facial Expression Recognition.”  [oai_citation:6‡arXiv](https://arxiv.org/abs/2404.15714?utm_source=chatgpt.com)

Best pick for this category:
- MPA-FER

Recommended final comparison for the project

Cleanest version:
- Category 1: POSTER++
- Category 2: ResNet-18
- Category 3: MPA-FER

Why this set works:
- POSTER++ gives you a dataset-specific RAF-DB literature anchor
- ResNet-18 gives you a standard transfer-learning baseline
- MPA-FER gives you a general FER-specific modern method

Report-ready write-up

We compare three types of models on RAF-DB. First, we include dataset-specific FER models, meaning models already established on RAF-DB in prior literature, to provide a benchmark grounded in this dataset. Second, we include general-purpose classification backbones, which are standard image-classification architectures adapted to facial-expression recognition through transfer learning, to serve as fair and interpretable baselines. Third, we include general FER models not tied specifically to RAF-DB, meaning methods proposed for facial-expression recognition more broadly and then evaluated on RAF-DB among other benchmarks. This structure separates dataset-established references, generic vision baselines, and FER-specific methods in a way that is clearer and less overlapping than a category such as “high-performance FER models.”  [oai_citation:7‡whdeng.cn](https://www.whdeng.cn/RAF/model1.html?utm_source=chatgpt.com)

Short version

- Dataset-specific FER models: POSTER++, POSTER
- General-purpose classification backbones: ResNet-18, ResNet-50, EfficientNet-B0, MobileNetV3
- General FER models not tied to RAF-DB: MPA-FER, Ada-DF