import tensorflow as tf
from prep import train_ds, val_ds
import json

model = tf.keras.models.load_model("data/1_prepared/model_prep.hdf5")

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

model.summary()

# Training

epochs = 16
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

model.save("data/2_model/model.hdf5")

# Get the dictionary containing each metric and the loss for each epoch
history_dict = history.history
# Save it under the form of a json file
json.dump(history_dict, open("data/2_model/history.json", "w"))
