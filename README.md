# Aligning object detector bounding boxes with human preference

## Description
Official repository for the paper "Aligning object detector bounding boxes with human preference".

![AP_vs_scaling_factor_vs_human_preference_with_images.png](figures%2FAP_vs_scaling_factor_vs_human_preference_with_images.png)

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

![asymmetric_loss_figures.png](figures%2Fasymmetric_loss_figures.png)
![asymmetric_loss_figures.jpg](figures%2Fasymmetric_loss_figures.jpg)

## Support
For technical questions and support: [o.strafforello@tudelft.nl](mailto:o.strafforello@tudelft.nl).

## Authors and acknowledgment
Ombretta Strafforello, Osman S. Kayhan, Oana Inel, Klamer Schutte and Jan van Gemert.
