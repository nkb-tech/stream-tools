from torchaudio.utils import ffmpeg_utils

def make_ffmpeg_decoder(decoder: str, gpu: bool) -> str:
    '''Check and make ffmpeg decoder
    Inputs:
        decoder: str
        device: bool if true than gpu, false - cpu 
    
    Returns:
    str
    '''

    if gpu:
        decoder = f'{decoder}_cuvid'

    assert decoder in ffmpeg_utils.get_video_decoders(), \
        f'Decoder {decoder} is not supported in FFmpeg. Please check available decoder.'
    
    return decoder
