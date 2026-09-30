import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import PIL.Image
from einops import rearrange, repeat
from dataclasses import dataclass
import copy
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin, IPAdapterMixin, LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, ImageProjection
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
    BaseOutput
)
from diffusers.utils.torch_utils import is_compiled_module, is_torch_version, randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, StableDiffusionMixin
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UniPCMultistepScheduler,
)

from dfr_libs.unet_2d_condition import UNet2DConditionModel
from dfr_libs.brushnet_CA import BrushNetModel
from .fast_config import FAST

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def vae_encode_sampled(vae, pixel_values, chunk=4, rng_slice=4):
    """分块做 VAE encode 并采样 latent，chunk 会向上对齐到 rng_slice 的整数倍。"""
    chunk = max(rng_slice, (int(chunk) // rng_slice) * rng_slice)
    if chunk == rng_slice:
        return torch.cat(
            [vae.encode(pixel_values[i : i + rng_slice]).latent_dist.sample()
             for i in range(0, pixel_values.shape[0], rng_slice)],
            dim=0,
        )
    out = []
    for i in range(0, pixel_values.shape[0], chunk):
        dist = vae.encode(pixel_values[i : i + chunk]).latent_dist
        n = dist.mean.shape[0]
        if chunk != rng_slice and n > rng_slice:
            noise = torch.cat(
                [torch.randn_like(dist.mean[j : j + rng_slice]) for j in range(0, n, rng_slice)],
                dim=0,
            )
        else:
            noise = torch.randn_like(dist.mean)
        out.append(dist.mean + dist.std * noise)
    return torch.cat(out, dim=0)


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    **kwargs,
):
    """调用 scheduler 的 set_timesteps，返回 (timesteps, num_inference_steps)。"""
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def get_frames_context_swap(total_frames=192, overlap=4, num_frames_per_clip=24):
    if total_frames<num_frames_per_clip:
        num_frames_per_clip = total_frames
    context_list = []
    context_list_swap = []
    for i in range(1, 2):
        sample_interval = np.array(range(0,total_frames,i))
        n = len(sample_interval)
        if n>num_frames_per_clip:
            for k in range(0,n-num_frames_per_clip,num_frames_per_clip-overlap):
                context_list.append(sample_interval[k:k+num_frames_per_clip])
            if k+num_frames_per_clip < n and i==1:
                context_list.append(sample_interval[n-num_frames_per_clip:n])
            context_list_swap.append(sample_interval[0:num_frames_per_clip])
            for k in range(num_frames_per_clip//2, n-num_frames_per_clip, num_frames_per_clip-overlap):
                context_list_swap.append(sample_interval[k:k+num_frames_per_clip])
            if k+num_frames_per_clip < n and i==1:
                context_list_swap.append(sample_interval[n-num_frames_per_clip:n])
        if n==num_frames_per_clip:
            context_list.append(sample_interval[n-num_frames_per_clip:n])
            context_list_swap.append(sample_interval[n-num_frames_per_clip:n])
    return context_list, context_list_swap


@dataclass
class DiffuEraserPipelineOutput(BaseOutput):
    frames: Union[torch.Tensor, np.ndarray]
    latents: Union[torch.Tensor, np.ndarray]


class StableDiffusionDiffuEraserPipeline(
    DiffusionPipeline,
    StableDiffusionMixin,
    TextualInversionLoaderMixin,
    LoraLoaderMixin,
    IPAdapterMixin,
    FromSingleFileMixin,
):
    r"""基于视频扩散模型与 BrushNet 引导的修复 pipeline。"""

    model_cpu_offload_seq = "text_encoder->image_encoder->unet->vae"
    _optional_components = ["safety_checker", "feature_extractor", "image_encoder"]
    _exclude_from_cpu_offload = ["safety_checker"]
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        brushnet: BrushNetModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        image_encoder: CLIPVisionModelWithProjection = None,
        requires_safety_checker: bool = True,
    ):
        super().__init__()

        if safety_checker is None and requires_safety_checker:
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            brushnet=brushnet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True)
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
        **kwargs,
    ):
        deprecation_message = "`_encode_prompt()` is deprecated and it will be removed in a future version. Use `encode_prompt()` instead. Also, be aware that the output format changed from a concatenated tensor to a tuple."
        deprecate("_encode_prompt()", "1.0.0", deprecation_message, standard_warn=False)

        prompt_embeds_tuple = self.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            **kwargs,
        )

        prompt_embeds = torch.cat([prompt_embeds_tuple[1], prompt_embeds_tuple[0]])

        return prompt_embeds

    def encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    ):
        r"""将 prompt 编码为 text encoder 的 hidden states。"""

        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale

            if not USE_PEFT_BACKEND:
                adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)
            else:
                scale_lora_layers(self.text_encoder, lora_scale)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            if clip_skip is None:
                prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=attention_mask)
                prompt_embeds = prompt_embeds[0]
            else:
                prompt_embeds = self.text_encoder(
                    text_input_ids.to(device), attention_mask=attention_mask, output_hidden_states=True
                )

                prompt_embeds = prompt_embeds[-1][-(clip_skip + 1)]

                prompt_embeds = self.text_encoder.text_model.final_layer_norm(prompt_embeds)

        if self.text_encoder is not None:
            prompt_embeds_dtype = self.text_encoder.dtype
        elif self.unet is not None:
            prompt_embeds_dtype = self.unet.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape

        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        if isinstance(self, LoraLoaderMixin) and USE_PEFT_BACKEND:
            unscale_lora_layers(self.text_encoder, lora_scale)

        return prompt_embeds, negative_prompt_embeds

    def encode_image(self, image, device, num_images_per_prompt, output_hidden_states=None):
        dtype = next(self.image_encoder.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor(image, return_tensors="pt").pixel_values

        image = image.to(device=device, dtype=dtype)
        if output_hidden_states:
            image_enc_hidden_states = self.image_encoder(image, output_hidden_states=True).hidden_states[-2]
            image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
            uncond_image_enc_hidden_states = self.image_encoder(
                torch.zeros_like(image), output_hidden_states=True
            ).hidden_states[-2]
            uncond_image_enc_hidden_states = uncond_image_enc_hidden_states.repeat_interleave(
                num_images_per_prompt, dim=0
            )
            return image_enc_hidden_states, uncond_image_enc_hidden_states
        else:
            image_embeds = self.image_encoder(image).image_embeds
            image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            uncond_image_embeds = torch.zeros_like(image_embeds)

            return image_embeds, uncond_image_embeds

    def decode_latents(self, latents, weight_dtype):
        latents = 1 / self.vae.config.scaling_factor * latents
        video = []
        for t in range(latents.shape[0]):
            video.append(self.vae.decode(latents[t:t+1, ...].to(weight_dtype)).sample)
        video = torch.concat(video, dim=0)

        video = video.float()
        return video

    def prepare_ip_adapter_image_embeds(
        self, ip_adapter_image, ip_adapter_image_embeds, device, num_images_per_prompt, do_classifier_free_guidance
    ):
        if ip_adapter_image_embeds is None:
            if not isinstance(ip_adapter_image, list):
                ip_adapter_image = [ip_adapter_image]

            if len(ip_adapter_image) != len(self.unet.encoder_hid_proj.image_projection_layers):
                raise ValueError(
                    f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {len(self.unet.encoder_hid_proj.image_projection_layers)} IP Adapters."
                )

            image_embeds = []
            for single_ip_adapter_image, image_proj_layer in zip(
                ip_adapter_image, self.unet.encoder_hid_proj.image_projection_layers
            ):
                output_hidden_state = not isinstance(image_proj_layer, ImageProjection)
                single_image_embeds, single_negative_image_embeds = self.encode_image(
                    single_ip_adapter_image, device, 1, output_hidden_state
                )
                single_image_embeds = torch.stack([single_image_embeds] * num_images_per_prompt, dim=0)
                single_negative_image_embeds = torch.stack(
                    [single_negative_image_embeds] * num_images_per_prompt, dim=0
                )

                if do_classifier_free_guidance:
                    single_image_embeds = torch.cat([single_negative_image_embeds, single_image_embeds])
                    single_image_embeds = single_image_embeds.to(device)

                image_embeds.append(single_image_embeds)
        else:
            repeat_dims = [1]
            image_embeds = []
            for single_image_embeds in ip_adapter_image_embeds:
                if do_classifier_free_guidance:
                    single_negative_image_embeds, single_image_embeds = single_image_embeds.chunk(2)
                    single_image_embeds = single_image_embeds.repeat(
                        num_images_per_prompt, *(repeat_dims * len(single_image_embeds.shape[1:]))
                    )
                    single_negative_image_embeds = single_negative_image_embeds.repeat(
                        num_images_per_prompt, *(repeat_dims * len(single_negative_image_embeds.shape[1:]))
                    )
                    single_image_embeds = torch.cat([single_negative_image_embeds, single_image_embeds])
                else:
                    single_image_embeds = single_image_embeds.repeat(
                        num_images_per_prompt, *(repeat_dims * len(single_image_embeds.shape[1:]))
                    )
                image_embeds.append(single_image_embeds)

        return image_embeds

    def run_safety_checker(self, image, device, dtype):
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            if torch.is_tensor(image):
                feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
            else:
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        return image, has_nsfw_concept

    def decode_latents(self, latents, weight_dtype):
        latents = 1 / self.vae.config.scaling_factor * latents
        chunk = max(1, int(FAST["vae_decode_chunk"]))
        video = []
        for t in range(0, latents.shape[0], chunk):
            video.append(self.vae.decode(latents[t:t+chunk, ...].to(weight_dtype)).sample)
        video = torch.concat(video, dim=0)

        video = video.float()
        return video

    def prepare_extra_step_kwargs(self, generator, eta):
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        images,
        masks,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        ip_adapter_image=None,
        ip_adapter_image_embeds=None,
        brushnet_conditioning_scale=1.0,
        control_guidance_start=0.0,
        control_guidance_end=1.0,
        callback_on_step_end_tensor_inputs=None,
    ):
        if callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        is_compiled = hasattr(F, "scaled_dot_product_attention") and isinstance(
            self.brushnet, torch._dynamo.eval_frame.OptimizedModule
        )
        if (
            isinstance(self.brushnet, BrushNetModel)
            or is_compiled
            and isinstance(self.brushnet._orig_mod, BrushNetModel)
        ):
            self.check_image(images, masks, prompt, prompt_embeds)
        else:
            assert False

        if (
            isinstance(self.brushnet, BrushNetModel)
            or is_compiled
            and isinstance(self.brushnet._orig_mod, BrushNetModel)
        ):
            if not isinstance(brushnet_conditioning_scale, float):
                raise TypeError("For single brushnet: `brushnet_conditioning_scale` must be type `float`.")
        else:
            assert False

        if not isinstance(control_guidance_start, (tuple, list)):
            control_guidance_start = [control_guidance_start]

        if not isinstance(control_guidance_end, (tuple, list)):
            control_guidance_end = [control_guidance_end]

        if len(control_guidance_start) != len(control_guidance_end):
            raise ValueError(
                f"`control_guidance_start` has {len(control_guidance_start)} elements, but `control_guidance_end` has {len(control_guidance_end)} elements. Make sure to provide the same number of elements to each list."
            )

        for start, end in zip(control_guidance_start, control_guidance_end):
            if start >= end:
                raise ValueError(
                    f"control guidance start: {start} cannot be larger or equal to control guidance end: {end}."
                )
            if start < 0.0:
                raise ValueError(f"control guidance start: {start} can't be smaller than 0.")
            if end > 1.0:
                raise ValueError(f"control guidance end: {end} can't be larger than 1.0.")

        if ip_adapter_image is not None and ip_adapter_image_embeds is not None:
            raise ValueError(
                "Provide either `ip_adapter_image` or `ip_adapter_image_embeds`. Cannot leave both `ip_adapter_image` and `ip_adapter_image_embeds` defined."
            )

        if ip_adapter_image_embeds is not None:
            if not isinstance(ip_adapter_image_embeds, list):
                raise ValueError(
                    f"`ip_adapter_image_embeds` has to be of type `list` but is {type(ip_adapter_image_embeds)}"
                )
            elif ip_adapter_image_embeds[0].ndim not in [3, 4]:
                raise ValueError(
                    f"`ip_adapter_image_embeds` has to be a list of 3D or 4D tensors but is {ip_adapter_image_embeds[0].ndim}D"
                )

    def check_image(self, images, masks, prompt, prompt_embeds):
        for image in images:
            image_is_pil = isinstance(image, PIL.Image.Image)
            image_is_tensor = isinstance(image, torch.Tensor)
            image_is_np = isinstance(image, np.ndarray)
            image_is_pil_list = isinstance(image, list) and isinstance(image[0], PIL.Image.Image)
            image_is_tensor_list = isinstance(image, list) and isinstance(image[0], torch.Tensor)
            image_is_np_list = isinstance(image, list) and isinstance(image[0], np.ndarray)

            if (
                not image_is_pil
                and not image_is_tensor
                and not image_is_np
                and not image_is_pil_list
                and not image_is_tensor_list
                and not image_is_np_list
            ):
                raise TypeError(
                    f"image must be passed and be one of PIL image, numpy array, torch tensor, list of PIL images, list of numpy arrays or list of torch tensors, but is {type(image)}"
                )
        for mask in masks:
            mask_is_pil = isinstance(mask, PIL.Image.Image)
            mask_is_tensor = isinstance(mask, torch.Tensor)
            mask_is_np = isinstance(mask, np.ndarray)
            mask_is_pil_list = isinstance(mask, list) and isinstance(mask[0], PIL.Image.Image)
            mask_is_tensor_list = isinstance(mask, list) and isinstance(mask[0], torch.Tensor)
            mask_is_np_list = isinstance(mask, list) and isinstance(mask[0], np.ndarray)

            if (
                not mask_is_pil
                and not mask_is_tensor
                and not mask_is_np
                and not mask_is_pil_list
                and not mask_is_tensor_list
                and not mask_is_np_list
            ):
                raise TypeError(
                    f"mask must be passed and be one of PIL image, numpy array, torch tensor, list of PIL images, list of numpy arrays or list of torch tensors, but is {type(mask)}"
                )

        if image_is_pil:
            image_batch_size = 1
        else:
            image_batch_size = len(image)

        if prompt is not None and isinstance(prompt, str):
            prompt_batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            prompt_batch_size = len(prompt)
        elif prompt_embeds is not None:
            prompt_batch_size = prompt_embeds.shape[0]

        if image_batch_size != 1 and image_batch_size != prompt_batch_size:
            raise ValueError(
                f"If image batch size is not 1, image batch size must be same as prompt batch size. image batch size: {image_batch_size}, prompt batch size: {prompt_batch_size}"
            )

    def prepare_image(
        self,
        images,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance=False,
        guess_mode=False,
    ):
        images_new = []
        for image in images:
            image = self.image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)
            image_batch_size = image.shape[0]

            if image_batch_size == 1:
                repeat_by = batch_size
            else:
                repeat_by = num_images_per_prompt

            image = image.repeat_interleave(repeat_by, dim=0)

            image = image.to(device=device, dtype=dtype)

            images_new.append(image)

        return images_new

    def prepare_latents(self, batch_size, num_channels_latents, num_frames, height, width, dtype, device, generator, latents=None):
        shape = (
            batch_size,
            num_channels_latents,
            num_frames,
            height // self.vae_scale_factor, 
            width // self.vae_scale_factor
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            noise = rearrange(randn_tensor(shape, generator=generator, device=device, dtype=dtype), "b c t h w -> (b t) c h w")
        else:
            noise = latents.to(device)

        latents = noise * self.scheduler.init_noise_sigma
        return latents, noise

    @staticmethod
    def temp_blend(a, b, overlap):
        factor = torch.arange(overlap).to(b.device).view(overlap, 1, 1, 1) / (overlap - 1)
        a[:overlap, ...] = (1 - factor) * a[:overlap, ...] + factor * b[:overlap, ...]
        a[overlap:, ...] = b[overlap:, ...]
        return a

    def get_guidance_scale_embedding(self, w, embedding_dim=512, dtype=torch.float32):
        """生成 timestep 的 guidance scale embedding。"""
        assert len(w.shape) == 1
        w = w * 1000.0

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def clip_skip(self):
        return self._clip_skip

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None

    @property
    def cross_attention_kwargs(self):
        return self._cross_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @torch.no_grad()
    def __call__(
        self,
        num_frames: Optional[int] = 24,
        prompt: Union[str, List[str]] = None,
        images: PipelineImageInput = None,
        masks: PipelineImageInput = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.FloatTensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        brushnet_conditioning_scale: Union[float, List[float]] = 1.0,
        guess_mode: bool = False,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):
        r"""执行生成的主调用函数。"""

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )

        brushnet = self.brushnet._orig_mod if is_compiled_module(self.brushnet) else self.brushnet

        if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
            control_guidance_start = len(control_guidance_end) * [control_guidance_start]
        elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
            control_guidance_end = len(control_guidance_start) * [control_guidance_end]
        elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
            control_guidance_start, control_guidance_end = (
                [control_guidance_start],
                [control_guidance_end],
            )

        self.check_inputs(
            prompt,
            images,
            masks,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            brushnet_conditioning_scale,
            control_guidance_start,
            control_guidance_end,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        global_pool_conditions = (
            brushnet.config.global_pool_conditions
            if isinstance(brushnet, BrushNetModel)
            else brushnet.nets[0].config.global_pool_conditions
        )
        guess_mode = guess_mode or global_pool_conditions
        video_length = len(images)

        text_encoder_lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=self.clip_skip,
        )

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

        if isinstance(brushnet, BrushNetModel):
            images = self.prepare_image(
                images=images,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=brushnet.dtype,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                guess_mode=guess_mode,
            )
            original_masks = self.prepare_image(
                images=masks,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=brushnet.dtype,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                guess_mode=guess_mode,
            )
            original_masks_new = []
            for original_mask in original_masks:
                original_mask=(original_mask.sum(1)[:,None,:,:] < 0).to(images[0].dtype) 
                original_masks_new.append(original_mask)
            original_masks = original_masks_new

            height, width = images[0].shape[-2:]
        else:
            assert False

        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        self._num_timesteps = len(timesteps)

        num_channels_latents = self.unet.config.in_channels
        latents, noise = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            num_frames,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        images = torch.cat(images)
        images = images.to(dtype=images[0].dtype)
        conditioning_latents = vae_encode_sampled(
            self.vae, images, chunk=FAST["vae_encode_chunk"], rng_slice=4
        )

        conditioning_latents = conditioning_latents * self.vae.config.scaling_factor

        original_masks = torch.cat(original_masks) 
        masks = torch.nn.functional.interpolate(
            original_masks, 
            size=(
                latents.shape[-2], 
                latents.shape[-1]
            )
        )

        conditioning_latents=torch.concat([conditioning_latents,masks],1)

        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        added_cond_kwargs = (
            {"image_embeds": image_embeds}
            if ip_adapter_image is not None or ip_adapter_image_embeds is not None
            else None
        )

        brushnet_keep = []
        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            brushnet_keep.append(keeps[0] if isinstance(brushnet, BrushNetModel) else keeps)

        overlap = int(FAST["win_overlap"])
        if overlap < 0:
            overlap = num_frames//4
        overlap = max(0, min(overlap, num_frames - 1))
        context_list, context_list_swap = get_frames_context_swap(video_length, overlap=overlap, num_frames_per_clip=num_frames)
        scheduler_status = [copy.deepcopy(self.scheduler.__dict__)] * len(context_list)
        scheduler_status_swap = [copy.deepcopy(self.scheduler.__dict__)] * len(context_list_swap)
        count = torch.zeros_like(latents)
        value = torch.zeros_like(latents)

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        is_unet_compiled = is_compiled_module(self.unet)
        is_brushnet_compiled = is_compiled_module(self.brushnet)
        is_torch_higher_equal_2_1 = is_torch_version(">=", "2.1")
        n_win = max(1, len(context_list), len(context_list_swap))
        with self.progress_bar(total=len(timesteps) * n_win) as progress_bar:
            for i, t in enumerate(timesteps):
                count.zero_()
                value.zero_()

                if (i%2==1):
                    context_list_choose = context_list_swap
                    scheduler_status_choose = scheduler_status_swap
                else:
                    context_list_choose = context_list
                    scheduler_status_choose = scheduler_status

                for j, context in enumerate(context_list_choose):
                    self.scheduler.__dict__.update(scheduler_status_choose[j])

                    latents_j = latents[context, :, :, :]

                    if (is_unet_compiled and is_brushnet_compiled) and is_torch_higher_equal_2_1:
                        torch._inductor.cudagraph_mark_step_begin()

                    latent_model_input = torch.cat([latents_j] * 2) if self.do_classifier_free_guidance else latents_j
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    if guess_mode and self.do_classifier_free_guidance:
                        control_model_input = latents_j
                        control_model_input = self.scheduler.scale_model_input(control_model_input, t)
                        brushnet_prompt_embeds = prompt_embeds.chunk(2)[1]
                        brushnet_prompt_embeds = rearrange(repeat(brushnet_prompt_embeds, "b c d -> b t c d", t=num_frames), 'b t c d -> (b t) c d')
                    else:
                        control_model_input = latent_model_input
                        brushnet_prompt_embeds = prompt_embeds
                        if self.do_classifier_free_guidance:
                            neg_brushnet_prompt_embeds, brushnet_prompt_embeds = brushnet_prompt_embeds.chunk(2)
                            brushnet_prompt_embeds = rearrange(repeat(brushnet_prompt_embeds, "b c d -> b t c d", t=num_frames), 'b t c d -> (b t) c d')
                            neg_brushnet_prompt_embeds = rearrange(repeat(neg_brushnet_prompt_embeds, "b c d -> b t c d", t=num_frames), 'b t c d -> (b t) c d')
                            brushnet_prompt_embeds = torch.cat([neg_brushnet_prompt_embeds, brushnet_prompt_embeds])
                        else:
                            brushnet_prompt_embeds = rearrange(repeat(brushnet_prompt_embeds, "b c d -> b t c d", t=num_frames), 'b t c d -> (b t) c d')

                    if isinstance(brushnet_keep[i], list):
                        cond_scale = [c * s for c, s in zip(brushnet_conditioning_scale, brushnet_keep[i])]
                    else:
                        brushnet_cond_scale = brushnet_conditioning_scale
                        if isinstance(brushnet_cond_scale, list):
                            brushnet_cond_scale = brushnet_cond_scale[0]
                        cond_scale = brushnet_cond_scale * brushnet_keep[i]

                    down_block_res_samples, mid_block_res_sample, up_block_res_samples = self.brushnet(
                        control_model_input,
                        t,
                        encoder_hidden_states=brushnet_prompt_embeds,
                        brushnet_cond=torch.cat([conditioning_latents[context, :, :, :]]*2) if self.do_classifier_free_guidance else conditioning_latents[context, :, :, :],
                        conditioning_scale=cond_scale,
                        guess_mode=guess_mode,
                        return_dict=False,
                    )

                    if guess_mode and self.do_classifier_free_guidance:
                        down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
                        mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])
                        up_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in up_block_res_samples]

                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        timestep_cond=timestep_cond,
                        cross_attention_kwargs=self.cross_attention_kwargs,
                        down_block_add_samples=down_block_res_samples,
                        mid_block_add_sample=mid_block_res_sample,
                        up_block_add_samples=up_block_res_samples,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                        num_frames=num_frames,
                    )[0]

                    if self.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                    latents_j = self.scheduler.step(noise_pred, t, latents_j, **extra_step_kwargs, return_dict=False)[0]

                    count[context, ...] += 1

                    if j==0:
                        value[context, ...] += latents_j
                    else:
                        if FAST["blend_sync"]:
                            overlap_index_list = [idx for idx, v in
                                                  enumerate(count[context, 0, 0, 0].tolist()) if v > 1]
                        else:
                            overlap_index_list = [index for index, value in enumerate(count[context, 0, 0, 0]) if value > 1]
                        overlap_cur = len(overlap_index_list)
                        ratio_next = torch.linspace(0, 1, overlap_cur+2)[1:-1]
                        ratio_pre = 1-ratio_next
                        for i_overlap in overlap_index_list:
                            value[context[i_overlap], ...] = value[context[i_overlap], ...]*ratio_pre[i_overlap] + latents_j[i_overlap, ...]*ratio_next[i_overlap]
                        value[context[i_overlap:num_frames], ...] = latents_j[i_overlap:num_frames, ...]

                    progress_bar.update()

                latents = value.clone()

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.unet.to("cpu")
            self.brushnet.to("cpu")
            torch.cuda.empty_cache()

        if  output_type == "latent":
            image = latents
            has_nsfw_concept = None
            return DiffuEraserPipelineOutput(frames=image, nsfw_content_detected=has_nsfw_concept)

        video_tensor = self.decode_latents(latents, weight_dtype=prompt_embeds.dtype)

        if output_type == "pt":
            video = video_tensor
        else:
            video = self.image_processor.postprocess(video_tensor, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (video, has_nsfw_concept)

        return DiffuEraserPipelineOutput(frames=video, latents=latents)
