# Aligning object detector bounding boxes with human preference

## Description
Official repository for the paper "Aligning object detector bounding boxes with human preference".

![Figure1.jpg](figures%2FFigure1.jpg)

*Previous work shows that humans tend to prefer large bounding boxes over small bounding boxes with the same
IoU. However, we show here that commonly used object detectors predict large and small boxes equally often. In this
work, we investigate how to align automatic detected object boxes with human preference and study whether this
improves human quality perception. We evaluate the performance of three commonly used object detectors through
a user study with more than 120 participants. We find that humans prefer object detections that are upscaled with factors
of 1.5 or 2, even if the corresponding AP is close to 0. Motivated by this result, we propose an asymmetric bounding
box regression loss that encourages large over small predicted bounding boxes. Our evaluation study shows that
object detectors fine-tuned with the asymmetric loss are better aligned with human preference and are preferred over
fixed scaling factors. A qualitative evaluation shows that the human preference might be influenced by some object 
characteristics, like object shape.*

![asymmetric_loss_figures.jpg](figures%2Fasymmetric_loss_figures.jpg)

## Requirements 

- Object detectors implementation: [Meta Detectron2](https://github.com/facebookresearch/detectron2).
- MS COCO dataset: pycocotools.

## Survey images

The survey images can be found on Google Drive. 
- Comparing object detections scaled with fixed factors: [link](https://drive.google.com/drive/folders/1Y2FVuTRDX1oNAcftV2H_tcKOjGt-tSXK?usp=share_link) 
- Comparing asymmetric loss and fixed scaling factors: [link](https://drive.google.com/drive/folders/11P8MmKYg6WpeQF3FjnFR05tcY2J_QvtA?usp=share_link)

## Support
For technical questions and support: [o.strafforello@tudelft.nl](mailto:o.strafforello@tudelft.nl).

## Authors and acknowledgment
Ombretta Strafforello, Osman S. Kayhan, Oana Inel, Klamer Schutte and Jan van Gemert.
