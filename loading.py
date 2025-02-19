from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

# Load your model
model = load_model('Model1.h5')

# Recompile the model with necessary parameters
model.compile(
    optimizer=Adam(),
    loss='sparse_categorical_crossentropy',  # Adjust based on your loss function
    metrics=['accuracy']  # Adjust based on the metrics you want
)
model.save('Model1_compiled.h5')
