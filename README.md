# RIFT

**RIFT: Disentangled Unsupervised Image Translation via Restricted Information Flow** </br>
*IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2023*</br>
<a href="https://ai.bu.edu/rift/">project page</a>

> Unsupervised image-to-image translation methods aim to map images from one domain into plausible examples from another domain while preserving the structure shared across two domains. In the many-to-many setting, an additional guidance example from the target domain is used to determine the domain-specific factors of variation of the generated image. In the absence of attribute annotations, methods have to infer which factors of variation are specific to each domain from data during training. In this paper, we show that many state-of-the-art architectures implicitly treat textures and colors as always being domain-specific, and thus fail when they are not. We propose a new method called RIFT that does not rely on such inductive architectural biases and instead infers which attributes are domain-specific vs shared directly from data. As a result, RIFT achieves consistently high cross-domain manipulation accuracy across multiple datasets spanning a wide variety of domain-specific and shared factors of variation.

<img src="https://ai.bu.edu/rift/rift-web-image.png" alt="metapose task" style="width:600px;"/>

If you use any of this code or its derivatives, please consider citing our work:

```
@inproceedings{usman2023rift,
    author    = {Usman, Ben and  Bashkirova, Dina and Saenko, Kate},
    title     = {{RIFT}: Disentangled Unsupervised Image Translation via Restricted Information Flow},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2023}
}

```

---

Training instructions, checkpoints, configs and datasets will be released shortly.

This implementation is heavily based on https://github.com/LynnHo/CycleGAN-Tensorflow-2 .
