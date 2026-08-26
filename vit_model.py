import tensorflow as tf
from vit_keras import vit

def ViT_Model(input_shape=(224, 224, 3), classes=10):
    # Khởi tạo backbone RỖNG hoàn toàn
    vit_base = vit.vit_b16(
    image_size=224, activation='softmax', pretrained=True, 
    include_top=False, pretrained_top=False
)
    vit_base.trainable = False

    model = tf.keras.Sequential([
        vit_base,
        tf.keras.layers.Flatten(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(10, activation='softmax')
        ])
    
    vit_base.trainable = True
    for layer in vit_base.layers[:-4]:
        layer.trainable = False
    
    return model