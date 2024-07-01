# Diffusion-Prior

This repository is a streamlined version of the `dalle2-pytorch` repository, retaining only the Prior model while removing components related to the Unet and decoder.

At of this commit, the training part of the diffusion prior has also been removed; only **a polished version of the inference code** for diffusion prior has been kept.

This repository provides an accessible entry point for people who desire to deeply explore the implementation of the diffusion prior code.

## What is a Prior Model?

Introduced in OpenAI's DALLE-2 paper, a Prior model is a network that converts text embeddings to image embeddings. Specifically, it correlates CLIP text embeddings with CLIP image embeddings.

<div style="text-align: center;">
    <img src="./diagram3.png" width="450px">
</div>


## Why Do We Need a Prior Model?

In typical diffusion-based text-to-image generation pipelines, text information guides the image generation process. This is usually achieved by using a text encoder and injecting the text embeddings into the Unet for guidance, ensuring the image output follows the instructions from the text (a task referred to as text alignment). While pioneering work has utilized CLIP text embeddings, CLIP text embeddings and CLIP image embeddings exist in different spaces. Research has shown that image embeddings encode better semantics and thus serve as superior conditions in conditional text-to-image generation tasks.

This motivates the creation of a module that can convert CLIP text embeddings to CLIP image embeddings. During inference, the pipeline involves obtaining the CLIP text embedding from the original prompt, then using the Prior model to convert it into a CLIP image embedding. The CLIP image embedding is then used as a condition in generating the image using diffusion models, provided the model was trained with CLIP image embeddings.

The following digram shows a simple use case of the image embedding - we use CLIP image embedding as an input condition in the Stable Diffusio pipeline.

<div style="text-align: center;">
    <img src="./diagram4.png" width="450px">
</div>

## The Architecture of the Prior Model

At a high level, a Prior model is a diffusion model implemented alongside a transformer. The input has dimensions [batch, concate_seq_length, 768], and the output has dimensions [batch, 768]. The structure can be represented as follows:

<div style="text-align: center;">
    <img src="./diagram1.png" width="450px">
</div>


Let's run through an example of one forward pass. Suppose we have text embeddings from CLIP, with a maximum length of 77 tokens. The text embedding tensor has a shape of [batch_size, 77, 768]. The time embedding, image embedding (starting with Gaussian noise), and learned embedding all have shapes of [batch_size, 1, 768]. By concatenating them along the sequence length dimension, we get a tensor of shape [batch_size, 80, 768] as input to the transformer. The transformer's output is also a tensor of shape [batch_size, 80, 768], and we select the last token (corresponding to the special learned embedding) as the prediction for the image embedding (of shape [batch_size, 768]). This explains one step of the diffusion process, which is repeated for a fixed number of steps (DDIM).

(Note: This process explains how to obtain the CLIP image embedding from the CLIP text embedding. To obtain an image from the CLIP image embedding, one needs a decoder or a type of diffusion model that was trained with the CLIP image embedding as a condition.)

## Important Classes Overview

**`diffusion_prior.py`**

This file defines the essential components for constructing a diffusion prior model. Specifically, the two classes `DiffusionPriorNetwork` and `DiffusionPrior` might seem confusing, so here are some key differences:

#### • DiffusionPriorNetwork

This class defines the transformer network shown in the diagram. In its forward function, it takes the input tokens and outputs the predicted image embedding. Note that this is a one-time forward pass and does not involve anything related to diffusion.

#### • DiffusionPrior

This class defines the entire diffusion process. In its initialization, a `DiffusionPriorNetwork` object is required. In this class, the forward function is responsible for computing the forward pass and the loss. This encapsulates the entire process of adding noise to the image embeddings, predicting the output, and calculating the loss based on the predictions. While this setup is uncommon, it can streamline certain parts of the training process, especially when the forward pass and loss computation are closely related and need to share intermediate computations or states.

## Example Usage


### Inference

A detailed inference script is available in the file `prior_inference.py`.


## Advanced Capabilities of a Prior Model

### Adding Additional Attributes as Conditions

Given the transformer's nature, we can easily add other conditions to the prior model. For instance, as shown in the diagram below, we can prepend an extra token embedding representing an additional feature of the image, such as an aesthetic score, crowd description, or number of people. This requires a simple layer to convert the numerical value of the description to the proper token size [batch_size, 1, 768].

<div style="text-align: center;">
    <img src="./diagram2.png" width="450px">
</div>


### Connecting More Powerful Text Embeddings to Image Embeddings

Research has shown that more powerful text embeddings are needed to achieve better text alignment. Studies have demonstrated that diffusion models conditioned on advanced text encoders, like T5, perform better. It would be interesting to see if we can develop a prior model that correlates T5 text embeddings with CLIP image embeddings. Since conditioning on image embeddings is intuitively better than conditioning only on text embeddings, this approach holds potential. 

However, a potential issue is that CLIP image embeddings might not be powerful enough, acting as a bottleneck in providing accurate instructions to the diffusion models. This doesn't mean the prior model is a dead end; it indicates that modifications are possible by replacing CLIP image embeddings with more advanced image embedding models.
