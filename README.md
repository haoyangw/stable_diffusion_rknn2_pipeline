# Converting Stable Diffusion model

## Prerequisites

* Linux x64 system with Docker and git installed
* Sufficient disk space to download the source ONNX model locally and temporarily store the generated RKNN model(before it's uploaded to HuggingFace)
* HuggingFace account with a HuggingFace token that has permission to publish a new repository under your account

## Conversion process

To convert a Stable Diffusion ONNX model to the RKNN format, run the following commands:

```
# Clone this repository
git clone https://github.com/haoyangw/stable_diffusion_rknn2_pipeline.git
# Change to the cloned repository
cd stable_diffusion_rknn2_pipeline
# Run the pipeline in docker
##  (might need to run each docker command below with sudo)
docker build -t $(whoami)/rknn-interactive . && docker run -it --rm $(whoami)/rknn-interactive
# Provide the requested values as prompted by the interactive pipeline script
# Converted RKNN model will be uploaded to your HuggingFace account
```

# Credits

@c0zaut for the [ez-er-rkllm-toolkit](https://github.com/c0zaut/ez-er-rkllm-toolkit), which this pipeline is based on
@happyme531 for the [Stable-Diffusion-1.5-LCM-ONNX-RKNN2](https://huggingface.co/happyme531/Stable-Diffusion-1.5-LCM-ONNX-RKNN2) repository, which the model conversion logic in the interactive pipeline script is based on
