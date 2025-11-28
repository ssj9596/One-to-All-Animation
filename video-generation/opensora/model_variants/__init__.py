from .transformer_cogvideo import CogVideoXTransformer3DModel
from .transformer_mochi import MochiTransformer3DModel
from .hunyuan_dit_src import HYVideoDiffusionTransformer, HYVideoDiffusionTransformerControlNet
from .hunyuan_dit_src import HYVideoDiffusionTransformerRef, HYVideoDiffusionTransformerEnd2End, HYVideoDiffusionTransformer_Refextractor, HYVideoDiffusionTransformer_Refextractor_2D, HYVideoDiffusionTransformer_Refextractor_2D_Q
from .hunyuan_dit_src import HYVideoDiffusionTransformer_Tokenmerge
from .hunyuan_dit_i2v_src import HYVideoDiffusionTransformer as HYVideoDiffusionTransformerI2V
from .cogvideo_controlnet import CogVideoXControlnet
from .wanx_diffusers_src import WanTransformer3DModel_Refextractor,WanTransformer3DModel
from .wanx_diffusers_src import WanTransformer3DModel_Refextractor_2D
from .wanx_diffusers_src import WanTransformer3DModel_Refextractor_2D_Pose_Final
from .wanx_diffusers_src import WanTransformer3DModel_Refextractor_2D_Controlnet_prefix
from opensora.flux_modules import PyramidFluxTransformer


def get_dit(model_name, config_path, weight_dtype):
    if model_name == 'cogvideo':
        return CogVideoXTransformer3DModel.from_config(config_path).to(weight_dtype), CogVideoXTransformer3DModel
    elif model_name == 'mochi':
        return MochiTransformer3DModel.from_config(config_path).to(weight_dtype), MochiTransformer3DModel
    elif model_name == 'hunyuan':
        return HYVideoDiffusionTransformer.from_config(config_path).to(weight_dtype), HYVideoDiffusionTransformer
    elif model_name == 'hunyuanref':
        return HYVideoDiffusionTransformerRef.from_config(config_path).to(weight_dtype), HYVideoDiffusionTransformerRef
    elif model_name == 'hunyuanend2end':
        return HYVideoDiffusionTransformerEnd2End.from_config(config_path).to(weight_dtype), HYVideoDiffusionTransformerEnd2End
    elif model_name == 'hunyuan_refextractor':
        return HYVideoDiffusionTransformer_Refextractor.from_config(config_path).to(weight_dtype), HYVideoDiffusionTransformer_Refextractor
    elif model_name == 'hunyuan_tokenmerge':
        return HYVideoDiffusionTransformer_Tokenmerge.from_config(config_path).to(weight_dtype), HYVideoDiffusionTransformer_Tokenmerge
    elif model_name == 'hunyuan_i2v':
        return HYVideoDiffusionTransformerI2V.from_config(config_path).to(weight_dtype), HYVideoDiffusionTransformerI2V

    elif model_name == 'wanx_refextractor':
        return WanTransformer3DModel_Refextractor.from_config(config_path).to(weight_dtype), WanTransformer3DModel_Refextractor
    elif model_name == 'wanx':
        return WanTransformer3DModel.from_config(config_path).to(weight_dtype), WanTransformer3DModel
    elif model_name == 'wanx_refextractor_2d':
        return WanTransformer3DModel_Refextractor_2D.from_config(config_path).to(weight_dtype), WanTransformer3DModel_Refextractor_2D
    
    elif model_name == 'hunyuan_refextractor_2d':
        return HYVideoDiffusionTransformer_Refextractor_2D.from_config(config_path).to(weight_dtype), HYVideoDiffusionTransformer_Refextractor_2D
    elif model_name == 'hunyuan_refextractor_2d_split':
        return HYVideoDiffusionTransformer_Refextractor_2D_Q.from_config(config_path).to(weight_dtype), HYVideoDiffusionTransformer_Refextractor_2D_Q


    elif model_name == 'wanx_refextractor_2d_pose_final':
        return WanTransformer3DModel_Refextractor_2D_Pose_Final.from_config(config_path).to(weight_dtype), WanTransformer3DModel_Refextractor_2D_Pose_Final
    elif model_name == 'wanx_refextractor_2d_controlnet_prefix':
        return WanTransformer3DModel_Refextractor_2D_Controlnet_prefix.from_config(config_path).to(weight_dtype), WanTransformer3DModel_Refextractor_2D_Controlnet_prefix


    elif model_name == 'pyramid_flow':
        #TODO: 临时这么写，后续修改
        model = PyramidFluxTransformer.from_pretrained(
            config_path, use_gradient_checkpointing=True, 
            gradient_checkpointing_ratio=0.75,
            torch_dtype=weight_dtype, 
            use_flash_attn=False, use_temporal_causal=True,
            interp_condition_pos=True, axes_dims_rope=[16, 24, 24],
        )
        return model, PyramidFluxTransformer

def get_controlnet(model_name, config_path, weight_dtype):
    if model_name == 'cogvideo-controlnet':
        return CogVideoXControlnet.from_config(config_path).to(weight_dtype), CogVideoXControlnet
    elif model_name == 'hunyuan-controlnet':
        return HYVideoDiffusionTransformerControlNet.from_config(config_path).to(weight_dtype), HYVideoDiffusionTransformerControlNet