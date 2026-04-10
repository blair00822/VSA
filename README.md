# Learning Forgery-Aware Lip Representations Without Forgery Priors (CVPR 2026)
Official PyTorch implementation of paper "Learning Forgery-Aware Lip Representations Without Forgery Priors".

## 📝 **Abstract**
<details>
  <summary>📌 Click for FULL Abstract</summary>
<sub>
Visual Speaker Authentication (VSA) verifies identity by analyzing lip dynamics during prompted speech, offering enhanced privacy compared to full-face methods while maintaining discriminability for high-security applications. However, recent advances in personalized talking face generation (TFG) have enabled realistic forgeries that closely mimic lip dynamics in sync with speech, posing severe threats to VSA systems. Prevailing defenses rely heavily on supervised classifiers trained on known forgeries via empirical risk minimization, resulting in poor generalization to unseen attacks, dependency on continuously updated fake data, and complete failure in the absence of effective forgery priors. In this paper, we revisit the design of forgery detectors and argue that over-reliance on fake priors hinders the exploitation of rich authenticity signals inherently present in real videos. We propose a novel detector trained exclusively on authentic data, learning forgery-aware representations through three key components: (1) lightweight modules that capture forgery-indicative statistics from real videos; (2) an asymmetric contrastive objective that compacts real samples while repelling potential forgeries in representation space; and (3) a theoretically grounded regularizer that shapes
real representations into a tractable, isotropic Gaussian. To support rigorous evaluation, we introduce a benchmark suite spanning diverse TFG forgeries. Across eight modern forgery attacks and ten state-of-the-art (SOTA) detectors, we achieve over a 10% reduction in error rates while preserving identity-verification capability with minimal overhead, and demonstrate robust generalization under diverse and complex real-world conditions.
</sub>

</details>

