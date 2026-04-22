RAF-DB comparison categories + model recommendations
Dataset note:
- RAF-DB is a widely used in-the-wild FER benchmark with around 30K images.
- Good choice for a course project because it has many published baselines and strong comparison papers.

======================================================================
1) Dataset-specific models
Definition:
- models explicitly reported and benchmarked on RAF-DB in FER literature
- useful as literature-grounded dataset references

Recommended options:

A) POSTER++
Why:
- strong RAF-DB-specific reference model
- simpler and lighter successor to POSTER
- reported 92.21% accuracy on RAF-DB
- also reports 8.4G FLOPs and 43.7M parameters
Use in project:
- best "dataset-specific reference" pick

Academic references:
- Mao, Jiawei; Xu, Rui; Yin, Xuesong; Chang, Yuanqi; Nie, Binling; Huang, Aibin.
  "POSTER++: A simpler and stronger facial expression recognition network."
  arXiv:2301.12149, 2023.
- Pattern Recognition version of POSTER++ (same model line; useful if your lecturer prefers journal-style references).

B) POSTER
Why:
- classic RAF-DB benchmark model
- landmark-image cross-fusion transformer specifically designed for FER
- reported 92.05% accuracy on RAF-DB
Use in project:
- older but established literature reference

Academic references:
- Zheng, Ce; Mendieta, Matias; Chen, Chen.
  "POSTER: A Pyramid Cross-Fusion Transformer Network for Facial Expression Recognition."
  arXiv:2204.04083, ICCV Workshop / CVF Open Access version, 2022/2023.

C) APViT
Why:
- strong transformer-style FER benchmark on RAF-DB
- useful as another dataset-specific literature comparison
- reported 91.98% on RAF-DB in later comparison tables
Use in project:
- good if you want one transformer benchmark that is simpler to explain than prompt-learning methods

Academic references:
- Xue, Fanglei; Wang, Qiangchang; Tan, Zichang; Ma, Zhongsong; Guo, Guodong.
  "Vision Transformer with Attentive Pooling for Robust Facial Expression Recognition."
  arXiv:2212.05463, 2022.
- IEEE Transactions on Affective Computing version (commonly cited as APViT).

Best pick for category 1:
- POSTER++

======================================================================
2) General-purpose classification backbones
Definition:
- standard image classification architectures
- not specifically built for FER
- useful as fair baselines

Recommended options:

A) ResNet-18
Why:
- strongest simple baseline
- easiest to train, debug, and explain
- very common transfer-learning baseline
Use in project:
- main baseline model

Academic references:
- He, Kaiming; Zhang, Xiangyu; Ren, Shaoqing; Sun, Jian.
  "Deep Residual Learning for Image Recognition."
  CVPR 2016.
- Fair-Consistent-Affect-Analysis repository includes resnet18 and RAF-DB support.

B) ResNet-50
Why:
- stronger and deeper general-purpose CNN baseline
- very common in FER papers as a backbone/reference
Use in project:
- optional second CNN baseline if you want a "small vs bigger CNN" comparison

Academic references:
- He, Kaiming; Zhang, Xiangyu; Ren, Shaoqing; Sun, Jian.
  "Deep Residual Learning for Image Recognition."
  CVPR 2016.
- Many FER papers and comparison tables still include ResNet-family backbones.

C) EfficientNet-B0
Why:
- compact, efficient, and strong for transfer learning
- better efficiency/accuracy tradeoff than many older CNNs
Use in project:
- best compact general backbone alternative to ResNet-18

Academic references:
- Tan, Mingxing; Le, Quoc V.
  "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks."
  arXiv:1905.11946 / ICML 2019.
- Fair-Consistent-Affect-Analysis repository includes efficientnet_b0 and RAF-DB support.

D) MobileNetV3
Why:
- hardware-aware lightweight architecture
- good if you want a practical deployment baseline
Use in project:
- real-time / lower-compute baseline

Academic references:
- Howard, Andrew; Sandler, Mark; Chu, Grace; Chen, Liang-Chieh; Bo Chen; Tan, Mingxing; Wang, Weijun; Zhu, Yukun; Pang, Ruoming; Vasudevan, Vijay; Le, Quoc V.; Adam, Hartwig.
  "Searching for MobileNetV3."
  arXiv:1905.02244, 2019.

Best pick for category 2:
- ResNet-18
Optional extra baseline:
- EfficientNet-B0

======================================================================
3) Task-specific or high-performance FER models
Definition:
- advanced FER-oriented models designed to push performance
- often use facial priors, cross-modal guidance, or FER-specific design choices

Recommended options:

A) MPA-FER
Why:
- modern high-performance FER method
- reported 93.74% on RAF-DB
- uses multimodal prompt alignment with a frozen CLIP backbone
- very parameter-efficient: reported 0.218 MB learnable parameters with ViT-B/16 and 0.443 MB with ViT-L/14
Use in project:
- best "advanced FER model" if you want a strong modern paper

Academic references:
- Ma, Fuyan; He, Yiran; Sun, Bin; Li, Shutao.
  "Multimodal Prompt Alignment for Facial Expression Recognition."
  ICCV 2025.
- CVF Open Access ICCV 2025 paper version.

B) Ada-DF
Why:
- FER-specific method designed around label-distribution learning
- explicitly targets annotation ambiguity in FER
Use in project:
- useful if you want a model with a strong methodological FER story
Note:
- more niche than POSTER++ or MPA-FER, but academically interesting

Academic references:
- Liu, Shu; Xu, Yan; Wan, Tongming; Kui, Xiaoyan.
  "Ada-DF: An Adaptive Label Distribution Fusion Network For Facial Expression Recognition."
  arXiv:2404.15714, 2024.

C) POSTER++
Why:
- also fits this category because it is a specialized high-performance FER architecture
Use in project:
- safer practical choice than MPA-FER if you want a more conventional architecture
Note:
- if you already use POSTER++ in category 1, do not repeat it here in the final comparison

Best pick for category 3:
- MPA-FER
Safer alternative:
- Ada-DF or POSTER++ (but avoid duplication across categories)

======================================================================
Best final 3-model comparison for your report

Recommended clean setup:
- Category 1 (Dataset-specific): POSTER++
- Category 2 (General-purpose backbone): ResNet-18
- Category 3 (Task-specific / high-performance FER): MPA-FER

Why this set works:
- POSTER++ = strong RAF-DB literature anchor
- ResNet-18 = clean and defensible baseline
- MPA-FER = modern advanced FER model with a strong academic story

Easier implementation version:
- Category 1: POSTER
- Category 2: ResNet-18
- Category 3: Ada-DF

Lowest-risk practical version:
- Category 1: POSTER++
- Category 2: ResNet-18
- Category 3: EfficientNet-B0
Note:
- this third option is less pure, because EfficientNet-B0 is really a backbone, not a FER-specific model

======================================================================
Extra helpful references for write-up / justification

Dataset / protocol references:
- Li, Shan et al.
  "Reliable Crowdsourcing and Deep Locality-Preserving Learning for Expression Recognition in the Wild."
  RAF-DB paper / related RAF benchmark literature.
- Official RAF-DB dataset webpage.
- Kollias, Dimitrios et al.
  "Protocol Towards Fair and Consistent Affect Analysis."
  2024.
- Kollias, Dimitrios et al.
  "Rethinking affect analysis: A protocol for ensuring fairness and consistency."
  2025.

Benchmark / survey-style support:
- Tutuianu, G. I. et al.
  "Benchmarking Deep Facial Expression Recognition."
  2023.
- SynFER paper comparison tables are useful for quick RAF-DB / AffectNet benchmark numbers across models.

======================================================================
Short recommendation sentence you can paste into your report

"We compare a RAF-DB-specific literature benchmark (POSTER++), a general-purpose image classification backbone (ResNet-18), and a high-performance FER-specific model (MPA-FER) to study trade-offs between standard transfer learning, dataset-grounded FER design, and modern task-specific performance."