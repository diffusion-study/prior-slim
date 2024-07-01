import json

import clip
import torch
import torch.nn.functional as F

from diffusion_prior import DiffusionPrior, DiffusionPriorNetwork


# data class
class PriorConfig:
    def __init__(self, config_data):
        self.dim = config_data.get("dim")
        self.depth = config_data.get("depth")
        self.max_text_len = config_data.get("max_text_len")
        self.dim_head = config_data.get("dim_head")
        self.heads = config_data.get("heads")
        self.normformer = config_data.get("normformer")
        self.rotary_emb = config_data.get("rotary_emb")
        self.image_embed_dim = config_data.get("image_embed_dim")
        self.image_size = config_data.get("image_size")
        self.image_channels = config_data.get("image_channels")
        self.timesteps = config_data.get("timesteps")
        self.loss_type = config_data.get("loss_type")
        self.predict_x_start = config_data.get("predict_x_start")
        self.beta_schedule = config_data.get("beta_schedule")
        self.condition_on_text_encodings = config_data.get(
            "condition_on_text_encodings"
        )
        self.num_iters = config_data.get("num_iters")


def load_prior_model(
    prior_model_path: str,
    device: str,
    prior_config: PriorConfig,
    clip: torch.nn.module,
) -> DiffusionPrior:
    """
    Function to load the DiffusionPrior object for inference
    Note that there are two objects involved: one of DiffusionPriorNetwork and one of DiffusionPrior:
        1) DiffusionPriorNetwork is the module that defines the main transformer structure
        2) DiffusionPrior is a broader idea and includes the sampling loop on the trainsformer
    """
    loaded_obj = torch.load(prior_model_path, map_location="cpu")
    state_dict = loaded_obj["ema_model" if "ema_model" in loaded_obj else "model"]

    # A module that defines the main transformer stucture
    prior_network = DiffusionPriorNetwork(
        dim=prior_config.dim,
        depth=prior_config.depth,
        dim_head=prior_config.dim_head,
        heads=prior_config.heads,
        normformer=prior_config.normformer,
        num_timesteps=prior_config.timesteps,
        max_text_len=prior_config.max_text_len,
    )

    state_dict_net = {
        k.replace("net.", ""): v for k, v in state_dict.items() if k.startswith("net.")
    }

    prior_network.load_state_dict(state_dict_net, strict=False)
    prior_network.to(device)

    diffusion_prior = DiffusionPrior(
        net=prior_network,
        clip=clip,
        image_embed_dim=prior_config.dim,
        timesteps=1000,
        cond_drop_prob=0,
        loss_type="l2",
        condition_on_text_encodings=True,
    ).to(device)

    state_dict_prior = {k: v for k, v in state_dict.items() if not k.startswith("net.")}
    diffusion_prior.load_state_dict(state_dict_prior, strict=False)

    return diffusion_prior


def prior_inference(
    text_prompt: str,
    diffusion_prior: DiffusionPrior,
    prior_config: PriorConfig,
    device: str,
) -> torch.Tensor:
    """
    How this function works:
        1) compute text_embedding and text_encodings for the input text_prompt
        2) with text_embedding and text_encodings as conditions, run DDIM on the transformer

    Explanation on the parameter num_iters:
        This refers to how many inference you would like to do for each input batch.
        The DALLE2 paper mentioned that they do num_iters of times and pick the run that has the largest similarity score with the text_embedding
    """

    # text_embedding is the global feature representation, which is of size [batch_size, 768]
    # text_encodings is the token-wise feature representation, which is of size [batch_size, max_text_len, 768]

    text_embedding, text_encodings = diffusion_prior.clip.embed_text(text_prompt).to(
        device
    )

    def run_prior_with_text_embedding(
        diffusion_prior: DiffusionPrior,
        text_embedding: torch.Tensor,
        text_encodings: torch.Tensor,
        prior_config: PriorConfig,
    ) -> torch.Tensor:
        batch_size = text_embedding.shape[0]

        num_iters = prior_config.num_iters

        # based on the DALLE2 paper, the do multiple times (num_iters, default is 2) per batch.
        # so here we repeat each batch to expand it multiple times
        if num_iters != 1:
            text_embedding = torch.repeat_interleave(text_embedding, num_iters, dim=0)
            text_encodings = torch.repeat_interleave(text_encodings, num_iters, dim=0)

        # the img_embedding will have size [batch_size * num_iters, 768]
        augmented_img_embedding_shape = torch.Size(
            [batch_size * num_iters, text_embedding.shape[-1]]
        )

        # main function - run sampling loop
        # img_embeddings is of size [batch_size * num_iters, 768]
        img_embeddings = diffusion_prior.p_sample_loop(
            shape=augmented_img_embedding_shape,
            text_cond={
                "text_embed": text_embedding,
                "text_encodings": text_encodings,
            },
            timesteps=prior_config.timesteps,
            cond_scale=prior_config.cond_scale,
            eta=prior_config.eta,
        )

        img_embeddings_norm = F.normalize(img_embeddings, dim=-1)

        # Pick the one that has the best match to the text embedding
        # Before split, weights has a shape of [batch_size*num_iters]
        # After split, weights is a list of tensors, where each tensor contains num_iters scores
        weights = torch.split(
            (text_embedding * img_embeddings_norm).sum(dim=-1).flatten(), num_iters
        )

        img_embeddings_list = []
        for index, weight in enumerate(weights):
            img_embeddings_list.append(
                img_embeddings[index * num_iters + weight.topk(1).indices]
            )
        img_embeddings = torch.concat(img_embeddings_list, dim=0)

        # return shape [batch_size, 1, 768]
        return img_embeddings.unsqueeze(1)

    img_embeddings = run_prior_with_text_embedding(
        diffusion_prior=diffusion_prior,
        text_embedding=text_embedding,
        text_encodings=text_encodings,
        prior_config=prior_config,
    )

    return img_embeddings


if __name__ == "__main__":
    """
    Example Usage to get the CLIP image embedding for a input text prompt.
    """

    prior_config_path = "configs/prior_inference_config.json"
    prior_model_path = "tmp.pth"  # a model ckpt trained with the same param setting as inference params
    device = "cuda" if torch.cuda.is_available() else "cpu"

    text_prompt = "a beautiful campus."

    # load inference params
    try:
        with open(prior_config_path) as config_file:
            config = json.load(config_file)
        prior_config = PriorConfig(config["prior"])
    except FileNotFoundError:
        print("Configuration file not found.")
        exit(1)

    # load clip model
    clip_model, _ = clip.load("ViT-L/14")

    # load prior model
    prior_model = load_prior_model(
        prior_model_path=prior_model_path,
        device=device,
        prior_config=prior_config,
        clip=clip_model,
    )

    # main inference
    image_embed = prior_inference(
        text_prompt=text_prompt,
        diffusion_prior=prior_model,
        prior_config=prior_config,
        device=device,
    )
