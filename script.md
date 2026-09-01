
[intro]
Hello judges, this is my submission of Simplicity, my solution to TikTok TechJam 2026's Track 5 on detecting AI-generated content under transforms.

[demo]
Simplicity ships with both an inference model and a browser extension to demonstrate practical viability, automatically detecting AI-generated images in real time while browsing the web. First, here's the extension in action... ‹ad hoc video recording›

[key ideas]
Now let's examine the inference model. Simplicity is based off research which demonstrates two key ideas. One: while specialized detectors for AI-generated images achieve high accuracy on benchmarks, their performance drops precipitously in real-life scenarios. Two: simpler architectures perform more robustly than specialized ones — in particular, training a simple linear classifier against a frozen vision foundation model tends to match specialized models in lab benchmarks while outperforming them on in-the-wild datasets.

[architecture]
Simplicity's architecture is based on those findings. It uses a frozen VFM, PE-Core-L, to create embeddings from images that have an aspect-preserving resize and crop applied to them, then runs the embeddings through a simple linear classifier to predict a probability. The classifier is 1,025 trainable parameters — 1,024 weights and a bias — against a 316-million-parameter backbone that is never fine-tuned.

[training pipeline and obtaining VFM embeddings]
To make Simplicity more robust to real-life scenarios, a data pipeline was built augmenting training images. Single transforms like blurs, resizes, and noise were used. Those transforms were also composed to create multi-chain transforms — like a noise then a jpeg encode — to simulate real-life conditions like a screenshot of a social media post. Before the images are fed in, they have an aspect-preserving crop and resize applied deliberately, to prevent the model from learning that square images are AI since the majority of diffusion models output square images, and a naive resize would cause distortions that could be learned instead.

[training: transforms]  ← rewritten, this inverted
Here's the list of transforms used. The model trains on every severity of every family but a separate chain of transforms that it was not trained on was used to evaluate its robustness.

[training: datasets]
These are the datasets used for training and evaluation. They contained a mix of real and AI-generated images spanning multiple generators and genres like photorealism and art. For the evaluation datasets a mix was used, including the official COCO val2017 and WildFake datasets, but also WildRF to test against social-media-style images, and some out-of-distribution datasets to test whether the model could generalize to unseen generators.

[results: 1]  ← "by two epochs" was wrong
Here are the results — this is the training loss. It flattens well before the first epoch even ends, at around step 1,000 of 6,200. The epoch means are 0.138 and 0.121, so the second epoch buys almost nothing in loss. That matters for the next slide.

[results: 2]  ← this claim has inverted; the new version is a stronger beat
This is the validation AUC. A second epoch actually makes validation look slightly better — pooled robustness goes from 0.9832 to 0.9838. But evlauation on the two held-out data sets, DALLE3 and the out of drop. So the final model trains a single epoch.

[results: 3]
Even though the model provides a numerical prediction probability, a threshold of 0.980 was chosen as it empirically gave a good balance between TPR and FPR.

[results: 4]
Based on that threshold, here are the per-generator recall results. DALL·E 3, was held out entirely from the training set to check how well the model generalizes to unseen AI generators — it reaches 0.992. The model also generalized well to GAN models at 0.982 across eight GAN generators.

Note that DALLE2 and ADM suffer degraded recall — 0.573 and 0.472 — but only because of the earlier trade-off setting the threshold high to reduce false positives. Their AUC rankings are 0.986 and 0.995 respectively. The scores are just compressed below a deliberately strict cut-off. At a 0.5 threshold their recall is 0.870 and 0.886

[results: 5]
Finally the validation benchmarks. demo_val is the competition's self-reported benchmark, the COCO and DALL·E Advanced datasets.

The other three are benchmarks I sourced myself. OOD contains 10 unseen generators absent from training data.

During empirical testing I noticed my browser extension would repeatedly trip false positives on social media photos, since the real images in the dataset did not look like the sort of images normally posted on social media. Hence I included WildRF to calibrate for that.

DALL·E 3 is another separate unseen held-out generator test set I included as well.

Across all benchmarks the worst single transform is still heavy noise. But on unseen generators the deepest chain is now just as damaging — 0.888 versus 0.890 — and that chain is the one thing the model never trains on. Composition, not severity, is the remaining frontier.

[error analysis notes and closing remarks]
The model still has a significantly higher false-positive rate on social-media-style photos, concentrated in enthusiast photography and edited images. Reddit sits at 0.8%, but Twitter is 4.4% and Facebook 5.6%. Possibly due to the absence of such images in my training set, which included more casual snapshot-style photos, plus the fact that enthusiast photography — bokeh, shallow depth of field — tends to look AI-adjacent visually.

Given the limitations of my single-GPU setup and time constraints, I believe this could be remedied with a more diverse and focused dataset rather than being a limitation of the model itself.

Regardless, Simplicity's performance shows good promise.