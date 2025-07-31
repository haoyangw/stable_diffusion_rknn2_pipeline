from rknn.api import RKNN
from huggingface_hub import login, whoami, snapshot_download, auth_check, ModelCard, HfApi
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
from pathlib import Path
from typing import List
import inquirer
import shutil
import os

"""
Credits to:
-@c0zaut for the ez-er-rkllm-toolkit interactive pipeline script, which this script is based on
-@happyme531 for the convert-onnx-to-rknn.py script for converting Stable Diffusion ONNX model to
  RKNN format, from which this script derives its model conversion logic
"""

class RKNNRemotePipeline:
    def __init__(self, model_id: str = "", component_list: str = "", resolution_list: str = "",
    		platform: str = "rk3588"):
        """
        Initialize primary values for pipeline class.

        :param model_id: HuggingFace repository ID for model (required)
        :param component_list: Comma-separated list of components to convert, e.g. 'text_encoder,unet,vae_decoder'
        :param resolution_list: Comma-separated list of resolution(s) as pairs of (width, height)
         for converted Stable Diffusion model(s)
        :param platform: CPU type of target platform. Must be rk3588 or rk3576
        """
        self.model_id: str = model_id
        self.component_list: str = component_list
        self.resolution_list: str = resolution_list
        self.platform: str = platform

    @staticmethod
    def mkpath(path):
        """
        HuggingFace Hub will just fail if the local_dir you are downloading to does not exist
        RKLLM will also fail to export if the directory does not already exist.

        :param paths: a list of paths (as strings) to check and create
        """
        try:
            if not os.path.exists(path):
                os.makedirs(path)
                print(f"mkdir'd {path}")
            else:
                print(f"Path {path} already exists! Great job!")
        except RuntimeError as e:
            print(f"Can't create paths for importing and exporting model.\n{e}")

    @staticmethod
    def cleanup_models(path=Path("./models")):
        print(f"Cleaning up model directory...")
        shutil.rmtree(path)
    
    # Based on parse_resolution_list() from
    #  https://huggingface.co/happyme531/Stable-Diffusion-1.5-LCM-ONNX-RKNN2/blob/main/convert-onnx-to-rknn.py
    @staticmethod
    def parse_resolution_list(resolutions: str) -> List[List[int]]:
        resolution_pairs = resolutions.split(',')
        parsed_resolutions = []
        for pair in resolution_pairs:
            width, height = map(int, pair.split('x'))
            parsed_resolutions.append([width, height])
        return parsed_resolutions

    def user_inputs(self):
        '''
        Obtain necessary inputs for model conversion
        This remote pipeline downloads the selected model
        It then iterates through and exports a model for each output resolution.
        '''
        self.inputs = [
            inquirer.Text("model_id", 
                          message="HuggingFace Repo ID for Stable Diffusion Model in user/repo format (default is TheyCallMeHex/LCM-Dreamshaper-V7-ONNX)", 
                          default="TheyCallMeHex/LCM-Dreamshaper-V7-ONNX"),
            inquirer.Text("component_list",
                          message="Comma-separated list of the components to convert, e.g. 'text_encoder,unet,vae_decoder'",
                          default="text_encoder, unet, vae_decoder"),
            inquirer.Text("resolution_list",
                          message="Comma-separated list of resolution(s) for the converted model(s), e.g. '256x256,512x512'",
                          default="256x256"),
            inquirer.List("platform", 
                          message="Which platform would you like to build for?", 
                          choices=["rk3562", "rk3566", "rk3568", "rk3576", "rk3588"], 
                          default="rk3588")
        ]
        
        self.config = inquirer.prompt(self.inputs)
        
        self.model_id = self.config["model_id"]
        self.component_list = self.config["component_list"]
        self.resolution_list = self.config["resolution_list"]
        self.platform = self.config["platform"]
        
    def build_vars(self):
        # Parse comma-seperated string of resolutions into list of pairs (width, height)
        self.resolutions: List[List[int]] = self.parse_resolution_list(self.resolution_list)
        # Split comma-seperated string of components into list of component names
        self.components: List[str] = self.component_list.split(',')
        self.model_name: str = self.model_id.split("/", 1)[1]
        self.model_dir: str = f"./models/{self.model_name}/"
        self.name_suffix: str = f"{self.platform}"
        self.export_name: str = f"{self.model_name}-{self.name_suffix}"
        self.export_path: str = f"./models/{self.model_name}-{self.platform}/"
        self.rknn_version: str = "2.3.0"

    # Based on convert_pipeline_component() function from
    #  https://huggingface.co/happyme531/Stable-Diffusion-1.5-LCM-ONNX-RKNN2/blob/main/convert-onnx-to-rknn.py
    def convert_local_model(self, model_path: str, export_path: str, 
            resolution_list: List[List[int]], target_platform: str = "rk3588",
            optimization_level: int = 3, do_quantization: bool = False):
        print(f'Converting {model_path} to RKNN model')
        print(f'    with target platform {target_platform}')
        print(f'    with resolutions:')
        for res in resolution_list:
            print(f'    - {res[0]}x{res[1]}')
        use_dynamic_shape: bool = False
        if(len(resolution_list) > 1):
            print("Warning: RKNN dynamic shape support is probably broken, may throw errors")
            use_dynamic_shape = True

        batch_size: int = 1
        LATENT_RESIZE_FACTOR: int = 8
        # build shape list
        if "text_encoder" in model_path:
            input_size_list = [[[1,77]]]
            inputs=['input_ids']
            use_dynamic_shape = False
        elif "unet" in model_path:
            # batch_size = 2  # for classifier free guidance # broken for rknn python api

            input_size_list = []
            for res in resolution_list:
                input_size_list.append(
                    [[1,4, res[0]//LATENT_RESIZE_FACTOR, res[1]//LATENT_RESIZE_FACTOR],
                     [1],
                     [1, 77, 768],
                     [1, 256]]
                )
            inputs=['sample','timestep','encoder_hidden_states','timestep_cond']
        elif "vae_decoder" in model_path:
            input_size_list = []
            for res in resolution_list:
                input_size_list.append(
                    [[1,4, res[0]//LATENT_RESIZE_FACTOR, res[1]//LATENT_RESIZE_FACTOR]]
                )
            inputs=['latent_sample']
        else:
            print("Unknown component: ", model_path)
            return ret

        # Instantiate RKNN class
        self.rknn = RKNN(verbose=True)

        # pre-process config
        print('--> Config model')
        self.rknn.config(target_platform=target_platform, optimization_level=optimization_level,
                    single_core_mode=True,
                    dynamic_input= input_size_list if use_dynamic_shape else None)
        print('done')

        # Load ONNX model
        print('--> Loading model')
        ret: int = self.rknn.load_onnx(model=model_path,
                             inputs=None if use_dynamic_shape else inputs,
                             input_size_list= None if use_dynamic_shape else input_size_list[0])   
        if ret != 0:
            print('Load model failed!')
            return ret
        print('done')

        # Build model
        print('--> Building model')
        ret = self.rknn.build(do_quantization=do_quantization, rknn_batch_size=batch_size)
        if ret != 0:
            print('Build model failed!')
            return ret
        print('done')

        # Export converted model
        print('--> Export RKNN model')
        ret = self.rknn.export_rknn(export_path)
        if ret != 0:
            print('Export RKNN model failed!')
            return ret
        print('done')

        self.rknn.release()
        print('RKNN model is converted successfully!')

    def remote_pipeline_to_local(self):
        '''
        Full conversion pipeline
        Downloads the chosen model from HuggingFace to a local destination, so no need
        to copy from the local HF cache.
        '''
        print(f"Checking if {self.model_dir} and {self.export_path} exist...")
        self.mkpath(self.model_dir)
        self.mkpath(self.export_path)
        for component in self.components:
            self.mkpath(f'{self.export_path}/{component.strip()}')
        
        if(len(self.resolutions) == 0):
            print("Error: No resolution(s) specified for converted model(s)")
            # Return non-zero code(1) to indicate that an error occurred to caller
            return 1

        print(f"Loading source model {self.model_id} from HuggingFace and downloading to {self.model_dir}")
        self.models_path: str = snapshot_download(repo_id=self.model_id, local_dir=self.model_dir)
        
        for component in self.components:
            onnx_path: str = f'{self.models_path}/{component.strip()}/model.onnx'
            output_model_path: str = f'{self.export_path}/{component.strip()}/model.rknn'
            self.convert_local_model(model_path=onnx_path, export_path=output_model_path,
                                 resolution_list=self.resolutions, target_platform=self.platform)


# Don't trust super().__init__ here    
class HubHelpers:
    def __init__(self, platform, model_id, resolutions, rknn_version):
        """
        Collection of helpers for interacting with HuggingFace.
        Due to some weird memory leak-y behaviors observed, would rather pass down
        parameters from the pipeline class then try to do something with super().__init__

        :param platform: CPU type of target platform. Must be rk3588 or rk3576
        :param model_id: HuggingFace repository ID for source model (required)
        :param resolutions: Comma-seperated list of resolution(s) as pairs of (width, height)
         for converted Stable Diffusion model(s)
        :param rknn_version: Version of RKNN used for conversion.
        """
        self.model_id = model_id
        self.resolutions = resolutions
        self.platform = platform
        self.rknn_version = rknn_version
        self.home_dir = os.environ['HOME']
        # Use Rust implementation of transfer for moar speed
        os.environ['HF_HUB_ENABLE_HF_TRANSFER']='1'

    @staticmethod
    def repo_check(model):
        """
        Checks if a HuggingFace repo exists and is gated
        """
        try:
            auth_check(model)
        except GatedRepoError:
            # Handle gated repository error
            print(f"{model} is a gated repo.\nYou do not have permission to access it.\n \
                  Please authenticate.\n")
        except RepositoryNotFoundError:
            # Handle repository not found error
            print(f"{model} not found.")
        else:
            print(f"Model repo {model} has been validated!")
            return True   
    
    def login_to_hf(self):
        """
        Helper function to authenticate with HuggingFace.
        Necessary for downloading gated repositories, and uploading.
        """
        self.token_path = f"{self.home_dir}/.cache/huggingface/token"
        if os.path.exists(self.token_path):
            self.token_file = open(self.token_path, "r")
            self.hf_token = self.token_file.read()
        else:
            self.hf_input = [inquirer.Text("token", message="Please enter your Hugging Face token", default="")]
            self.hf_token = inquirer.prompt(self.hf_input)["token"]
        try:
            login(token=self.hf_token)
        except Exception as e:
            print(f"Login failed: {e}\nGated models will be inaccessible, and you " + \
                  "will not be able to upload to HuggingFace.")
        else:
            print("Logged into HuggingFace!\n")
            self.hf_username = whoami(self.hf_token)["name"]
            print(self.hf_username + "\n")
            
    def build_card(self, export_path):
        """
        Inserts text into the README.md file of the original model, after the model data. 
        Using the HF built-in functions kept omitting the card's model data,
        so gonna do this old school.
        """
        self.model_name = self.model_id.split("/", 1)[1]
        self.card_in = ModelCard.load(self.model_id)
        self.card_out = export_path + "README.md"
        self.template = f'---\n' + \
            f'{self.card_in.data.to_yaml()}\n' + \
            f'---\n' + \
            f'# {self.model_name}-{self.platform.upper()}-{self.rknn_version}\n\n' + \
            f'This version of {self.model_name} has been converted to run on the {self.platform.upper()} NPU and generate images of resolution(s) {self.resolutions}.\n\n' + \
            f'Compatible with RKNN version: {self.rknn_version}\n\n' + \
            f'## Useful links:\n' + \
            f'[Official RKNN GitHub](https://github.com/airockchip/rknn-toolkit2) \n\n' + \
            f'[RockhipNPU Reddit](https://reddit.com/r/RockchipNPU) \n\n' + \
            f'Pretty much anything by these folks: [marty1885](https://github.com/marty1885) and [happyme531](https://huggingface.co/happyme531) \n\n' + \
            f'# Original Model Card for source model, {self.model_name}, below:\n\n' + \
            f'{self.card_in.text}'
        try:
            ModelCard.save(self.template, self.card_out)
        except RuntimeError as e:
            print(f"Runtime Error: {e}")
        except RuntimeWarning as w:
            print(f"Runtime Warning: {w}")
        else:
            print(f"Model card successfully exported to {self.card_out}!")
            c = open(self.card_out, 'r')
            print(c.read())
            c.close()

    def upload_to_repo(self, model, import_path, export_path):
        self.hf_api = HfApi(token=self.hf_token)
        self.repo_id = f"{self.hf_username}/{model}-{self.platform}-{self.rknn_version}"
        
        print(f"Creating repo if it does not already exist")
        try:
            self.repo_url = self.hf_api.create_repo(exist_ok=True, repo_id=self.repo_id)
        except:
            print(f"Create repo for {model} failed!")
        else:
            print(f"Repo created! URL: {self.repo_url}")

        print(f"Generating model card and copying configs")
        self.build_card(export_path)
        self.import_path = Path(import_path)
        self.export_path = Path(export_path)
        print(f"Uploading files to repo")
        shutil.copytree(self.import_path, self.export_path,
                        ignore=shutil.ignore_patterns('*.onnx*', '*.txt', 'README*'), 
                        copy_function=shutil.copy2, dirs_exist_ok=True)
        self.commit_info = self.hf_api.upload_folder(repo_id=self.repo_id, folder_path=self.export_path)
        print(self.commit_info)

if __name__ == "__main__":

    rk = RKNNRemotePipeline()
    rk.user_inputs()
    rk.build_vars()
    hf = HubHelpers(platform=rk.platform, model_id=rk.model_id, resolutions=rk.resolution_list,
                    rknn_version=rk.rknn_version)
    hf.login_to_hf()
    hf.repo_check(rk.model_id)

    try:
        rk.remote_pipeline_to_local()
    except RuntimeError as e:
        print(f"Model conversion failed: {e}")
    

    hf.upload_to_repo(model=rk.model_name, import_path=rk.model_dir, export_path=rk.export_path)
    print("Okay, these models are really big!")
    rk.cleanup_models("./models")
