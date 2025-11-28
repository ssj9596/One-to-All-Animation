from .models import HYVideoDiffusionTransformer, HUNYUAN_VIDEO_CONFIG
from .controlnet import HYVideoDiffusionTransformerControlNet
from .models_ref import HYVideoDiffusionTransformerRef
from .models_end2end import HYVideoDiffusionTransformerEnd2End
from .models_refextractor import HYVideoDiffusionTransformer_Refextractor
from .models_reftokenmerge import HYVideoDiffusionTransformer_Tokenmerge  
from .models_refextractor_2d import HYVideoDiffusionTransformer_Refextractor as HYVideoDiffusionTransformer_Refextractor_2D
from .models_refextractor_2d_split_q import HYVideoDiffusionTransformer_Refextractor as HYVideoDiffusionTransformer_Refextractor_2D_Q
def load_model(args, in_channels, out_channels, factor_kwargs):
    """load hunyuan video model

    Args:
        args (dict): model args
        in_channels (int): input channels number
        out_channels (int): output channels number
        factor_kwargs (dict): factor kwargs

    Returns:
        model (nn.Module): The hunyuan video model
    """
    if args.model in HUNYUAN_VIDEO_CONFIG.keys():
        model = HYVideoDiffusionTransformer(
            args,
            in_channels=in_channels,
            out_channels=out_channels,
            **HUNYUAN_VIDEO_CONFIG[args.model],
            **factor_kwargs,
        )
        return model
    else:
        raise NotImplementedError()
