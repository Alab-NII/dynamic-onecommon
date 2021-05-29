from models.ctx_encoder import MlpContextEncoder, AttentionContextEncoder
from models.rnn_dialog_model import RnnDialogModel
from models.hierarchical_dialog_model import HierarchicalDialogModel

MODELS = {
    'rnn_model': RnnDialogModel,
    'hierarchical_model': HierarchicalDialogModel,
}

CTX_ENCODERS = {
    'mlp_encoder': MlpContextEncoder,
    'attn_encoder': AttentionContextEncoder,
}

def get_model_names():
    return MODELS.keys()


def get_model_type(name):
    return MODELS[name]

def get_ctx_encoder_names():
    return CTX_ENCODERS.keys()

def get_ctx_encoder_type(name):
    return CTX_ENCODERS[name]