from .t5 import T5EncoderWrapper
from opensora.flux_modules import FluxTextEncoderWithMask
from .hunyuan import HunyuanEncoderWrapper,HunyuanImageEncoderWrapper
from .hunyuan_i2v import HunyuanEncoderWrapperI2V
from .wanx_t2v import WanxEncoderWrapperT2V
from .wanx_i2v import WanxEncoderWrapperI2V
def get_text_enc(enc_name, model_path, weight_dtype, i2v_type=None):
    if enc_name == 'flux':
        return FluxTextEncoderWithMask(model_path, torch_dtype=weight_dtype).eval()
    elif enc_name == 't5':
        return T5EncoderWrapper(model_path, weight_dtype)
    elif enc_name == 'hunyuan':
        return HunyuanEncoderWrapper(model_path, weight_dtype)
    elif enc_name == 'hunyuan_image':
        return HunyuanImageEncoderWrapper(model_path, weight_dtype)
        
    elif enc_name == 'hunyuan-i2v':
        return HunyuanEncoderWrapperI2V(model_path, weight_dtype, task = 'i2v', i2v_type = i2v_type)
    elif enc_name == 'wanx-i2v':
        return WanxEncoderWrapperI2V(model_path, weight_dtype)
    elif enc_name == 'wanx-t2v':
        return WanxEncoderWrapperT2V(model_path, weight_dtype)
    else:
        raise NotImplementedError(f'{enc_name} is not implemented.')
    