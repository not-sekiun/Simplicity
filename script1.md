[each paragraph is a slide]

[intro]
Hello judges, this is my submission of Simplicity, my solution to TikTok tech jam 2026's Track 5 on detecting ai generated content under transforms

[demo]
Simplicity ships with both an inference model along with a browser extension to demonstrate practical viability that can automatically detect ai generated images in real time while browsing the web. First heres the extension in action... <do ad hoc video recording>

[key ideas]
Now lets examine the inference model. Simplicity is based off of research which demonstrates two key idea. One that While specialized detectors for ai generated images achieve high accuracy on benchmarks, their performance drops precipitously in real life scenarios. and two: that Simpler architectures perform more robustly than specialized ones: in particular, training, a simple linear classifier against a frozen vision foundation model tends to match specialized models in lab benchmarks while outperforming them on in the wild datasets

[architecture]
Simplicity's architecture is based on those findings. It uses a frozen VFM model, PE-Core-L to create embeddings from images that have an aspect preserving resize and crop applied to them and then runs the embeddings through a simple linear classifier to predict a probability.

[training Pipeline and obtaining VFM embeddings]
To make simplicity more robust to real life scenarios, a data pipeline was built augmenting training images. Single chain transforms like blurs, resizes, noise and crops were used following the competition guidelines to augment the data. Those transforms were also composed with varying levels of depth to create multi chain transforms like a noise and then blur or a crop then jitter and then a blur to simulate real life conditions like a screenshot on a social media post. Before the images are fed in they have an aspect preserving crop and resize applied to them deliberately to prevent the model from learning that square images are ai since the majority of diffusion models output square images and a naive resize would cause distortions that could be learned.


[Training: Transforms]
Heres the list of transforms used on the images. For each transform only a single severity level was trained on so that the other severities would be extrapolations. The training chains and evaluation chains also differed and were used to simulate more realistic scenarios and transforms that may occur on images in the wild. Notably the third train chain omitted a jpeg encode as its last step to avoid teaching the model that jpeg encoded chains were ai

[Training: Datasets]
These are the datasets used for training and evaluation, they contained a mix of real and AI generated images spanning multiple generators and genres like photorealism or illustration and art. For the evaluation data set a mix of datasets was used including the official COCO val2017 and WildFake dataset from the competition track, but also wildrf to test against social media style images, dalle3_holdout and an out of distrubution, ood, dataset which was to test if the model could generalize to unseen generators.

[Results: 1]
Here are the results, first the training loss, by two epochs the loss already started to flatten considerably

[Results: 2]
This is the AUC, noticeably the the second Epoch caused clean data AUC to rise slightly but lowered robustness hence the final model trains a single epoch

[Results: 3]
Even though the model provides a numerical prediction probability, a threshold of 0.985 was chose as it empirically gave a good balance between the TPR and FPR

[Results: 4]
Based on that threshold here are the results of per generator recall. Noticably DALLE3, a modern diffusion model, was held out entirely from the training set to check to see how well the model could generalize to an unseen ai generators. The model also generalized well to GAN models despite them being obsoleted by diffusion models today.

Note that DALLE2 and ADM suffer degraded recall performance but only because of the earlier tradeoff we made setting threshold to 0.985 to reduce false positive rates. At 0.5 threshold recall for DALLE2 and ADM sit at around 0.85 and above

[Results: 5]
Finally the validation benchmarks. demo_val is the competitions self reported benchmark, the COCO and DALLE advanced datasets.

The other 3 are benchmarks sourced myself. OOD contains 10 unseen generators absent from training data

During empirical testing I noticed that my browser extension would repeatedly trip false positives on social media photos since the real images being classified in the data set did not look like the sort of images that would normally be posted on social media sites, hence I included WildRF to calibrate for that

DALLE3 is another separate unseen held out generator test set I included as well

Notably across all benchmarks, the worst single transform is adding heavy noise to the image.

[Error analysis notes and closing remarks]
Currently the model still has a significantly higher FPR rate on social media style photos, concentrated in enthusiast style photography and edited images. Possibly due to the absence of such images in my training set which included more casual snapshot style photos plus the fact that enthusiast style photography (those that include bokeh and shallow DOF) tend to look AI adjaecent visually.

Given the limitations of my single GPU setup and time constraints, I believe this is something that could be remedied with a more diverse and focused data set rather than it being a limitation of the model itself.

Regardless, Simplicity's performance shows good promise
