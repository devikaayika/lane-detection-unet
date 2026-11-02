
import tensorflow as tf
from tensorflow.keras import layers, models


def dice_loss(y_true, y_pred):
    """Dice loss encourages overlap between predicted and actual lane pixels."""
    smooth = 1.0

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])

    intersection = tf.reduce_sum(y_true * y_pred)

    dice_score = (
        (2.0 * intersection + smooth)
        / (
            tf.reduce_sum(y_true)
            + tf.reduce_sum(y_pred)
            + smooth
        )
    )

    return 1.0 - dice_score


def combined_loss(y_true, y_pred):
    """Combine binary cross-entropy and Dice loss."""
    bce = tf.keras.losses.binary_crossentropy(
        y_true, y_pred
    )
    return tf.reduce_mean(bce) + dice_loss(y_true, y_pred)


def build_unet(input_shape=(256, 512, 3)):
    """Build a U-Net model for road lane segmentation."""

    inputs = layers.Input(shape=input_shape)

    # Encoder
    c1 = layers.Conv2D(
        32, 3, activation="relu", padding="same"
    )(inputs)
    c1 = layers.Conv2D(
        32, 3, activation="relu", padding="same"
    )(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(
        64, 3, activation="relu", padding="same"
    )(p1)
    c2 = layers.Conv2D(
        64, 3, activation="relu", padding="same"
    )(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(
        128, 3, activation="relu", padding="same"
    )(p2)
    c3 = layers.Conv2D(
        128, 3, activation="relu", padding="same"
    )(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = layers.Conv2D(
        256, 3, activation="relu", padding="same"
    )(p3)
    c4 = layers.Conv2D(
        256, 3, activation="relu", padding="same"
    )(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    # Bottleneck
    c5 = layers.Conv2D(
        512, 3, activation="relu", padding="same"
    )(p4)
    c5 = layers.Conv2D(
        512, 3, activation="relu", padding="same"
    )(c5)

    # Decoder
    u6 = layers.Conv2DTranspose(
        256, 2, strides=2, padding="same"
    )(c5)
    u6 = layers.Concatenate()([u6, c4])
    c6 = layers.Conv2D(
        256, 3, activation="relu", padding="same"
    )(u6)
    c6 = layers.Conv2D(
        256, 3, activation="relu", padding="same"
    )(c6)

    u7 = layers.Conv2DTranspose(
        128, 2, strides=2, padding="same"
    )(c6)
    u7 = layers.Concatenate()([u7, c3])
    c7 = layers.Conv2D(
        128, 3, activation="relu", padding="same"
    )(u7)
    c7 = layers.Conv2D(
        128, 3, activation="relu", padding="same"
    )(c7)

    u8 = layers.Conv2DTranspose(
        64, 2, strides=2, padding="same"
    )(c7)
    u8 = layers.Concatenate()([u8, c2])
    c8 = layers.Conv2D(
        64, 3, activation="relu", padding="same"
    )(u8)
    c8 = layers.Conv2D(
        64, 3, activation="relu", padding="same"
    )(c8)

    u9 = layers.Conv2DTranspose(
        32, 2, strides=2, padding="same"
    )(c8)
    u9 = layers.Concatenate()([u9, c1])
    c9 = layers.Conv2D(
        32, 3, activation="relu", padding="same"
    )(u9)
    c9 = layers.Conv2D(
        32, 3, activation="relu", padding="same"
    )(c9)

    # One output channel: lane probability for each pixel
    outputs = layers.Conv2D(
        1, 1, activation="sigmoid"
    )(c9)

    model = models.Model(inputs, outputs, name="U_Net_Lane_Detection")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.0001
        ),
        loss=combined_loss,
        metrics=["binary_accuracy"]
    )

    return model


if __name__ == "__main__":
    model = build_unet()
    print("U-Net model created successfully!")
    print("Input shape:", model.input_shape)
    print("Output shape:", model.output_shape)
    print("Loss: Binary Cross-Entropy + Dice Loss")
    model.summary()
