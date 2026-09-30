# 模型下载

本文是 Detext 全部模型权重的下载方式。


---

## 1. HuggingFace 的四个仓库

用 `huggingface_hub` 自带的 `hf` 命令（装完 `requirements.txt` 就有了）。



```bash
# DiffuEraser 主权重：只要 brushnet 和 unet_main 两个子目录
#   -> models/diffuEraser/brushnet/  +  models/diffuEraser/unet_main/
hf download lixiaowen/diffuEraser \
  --include "brushnet/*" "unet_main/*" \
  --local-dir models/diffuEraser

# SD1.5 底座：整仓 30 G+，推理只需要下面这 6 项，约 4.8 G
#   -> models/stable-diffusion-v1-5/
hf download stable-diffusion-v1-5/stable-diffusion-v1-5 \
  --include "feature_extractor/*" "model_index.json" "safety_checker/*" \
            "scheduler/*" "text_encoder/*" "tokenizer/*" \
  --local-dir models/stable-diffusion-v1-5

# 独立 VAE
#   -> models/sd-vae-ft-mse/
hf download stabilityai/sd-vae-ft-mse \
  --local-dir models/sd-vae-ft-mse

# PCM LoRA：默认（--ckpt 2-Step）只需要这一个文件，约 129 M
#   -> models/PCM_Weights/sd15/pcm_sd15_smallcfg_2step_converted.safetensors
hf download wangfuyun/PCM_Weights \
  --include "sd15/pcm_sd15_smallcfg_2step_converted.safetensors" \
  --local-dir models/PCM_Weights
```

**用其它加速档位**：`run.py --ckpt` 还支持 `4-Step` / `8-Step` / `16-Step` /
`Normal CFG 4/8/16-Step` / `LCM-Like LoRA`。要用哪个就把上面 PCM 那条的 `--include`
换成对应文件名，或一次全下：

```bash
hf download wangfuyun/PCM_Weights --include "sd15/*" --local-dir models/PCM_Weights
```

`sd15/` 下是 8 个文件，共约 1.1 G。

---

## 2. ProPainter 先验（GitHub Release）

```bash
#   -> models/propainter/{ProPainter.pth, raft-things.pth, recurrent_flow_completion.pth}
mkdir -p models/propainter
base=https://github.com/sczhou/ProPainter/releases/download/v0.1.0
for f in ProPainter.pth raft-things.pth recurrent_flow_completion.pth; do
  curl -L -o "models/propainter/$f" "$base/$f"
done
```

---

## 3. PP-OCRv4 检测模型（PaddleOCR）

```bash
#   -> models/ppocr_det/ch_PP-OCRv4_det_infer/
mkdir -p models/ppocr_det
curl -L -o ch_PP-OCRv4_det_infer.tar \
  https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_infer.tar
tar -xf ch_PP-OCRv4_det_infer.tar -C models/ppocr_det
rm ch_PP-OCRv4_det_infer.tar
```

解出来是 `inference.pdmodel` / `inference.pdiparams` / `inference.pdiparams.info`
三个文件。

---


## 4. 校验

下完跑一次，程序会把缺的模型直接报出来并指出期望路径：

```bash
python run.py -i <一个视频文件> -o outputs
```

程序**只读 `models/`、不会联网下载**：缺哪个就直接报错，不会静默去拉一份。
