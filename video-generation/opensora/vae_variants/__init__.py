from .cogvideo_vae import get_cogvideo_vae_wrapper
from .pyramid_flow_vae import get_pyramid_flow_vae_wrapper
from .mochi_vae import get_mochi_vae_wrapper
from .hunyuan_vae import get_hunyuan_vae_wrapper
from .wanx_vae import get_wanx_vae_wrapper

def get_vae(vae_name, model_path, weight_dtype):
    if vae_name == 'cogvideo':
        return get_cogvideo_vae_wrapper(model_path, weight_dtype)
    elif vae_name == 'pyramid_flow':
        return get_pyramid_flow_vae_wrapper(model_path, weight_dtype)
    elif vae_name == 'mochi':
        return get_mochi_vae_wrapper(model_path, weight_dtype)
    elif vae_name == 'hunyuan':
        return get_hunyuan_vae_wrapper(model_path, weight_dtype)
    elif vae_name == 'wanx':
        return get_wanx_vae_wrapper(model_path, weight_dtype)
