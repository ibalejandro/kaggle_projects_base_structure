from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input


def alejo_first_model(hps, input_shapes):
    """Create the alejo_first_model's architecture."""
    inputs = Input(shape=input_shapes[0])
    outputs = None
    model = Model(inputs=[inputs], outputs=[outputs])
    return model
