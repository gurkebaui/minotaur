# Kitten-D Spec Document

Architecture:
[x]Diffusion Block Transformer at core
[x]MoE for efficiency
[ ]Looping for reasoning
[ ]implimenting sleep framework for cl
[ ]fast and slow weights and paths (look at HOPE for inspo)
[ ]look at CTM for inspo on looping
[ ]elegibility trace for cl
[ ]Gate (for cl)
[ ]base detectors (hard wired signals for rewarding and punishing)


inputs:
text for the first prototype, later images and audio as well

outputs:
text for the first prototype, later (maybe) images and audio as well

target: beat lfm2.5 750M base

1.part .
result: diffusion Blocks is way worse than baseline transformer. will not make it into final architecture as is.

new ways:
[ ] conv blocks for looping
[ ] hybrid attention
[ ] research results S ranks
[ ] research results A ranks