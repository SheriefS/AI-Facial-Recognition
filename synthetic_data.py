import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.utils import save_img

# Setup the ImageDataGenerator for augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

def augment_images(source_path, target_path, num_required):
    # Load all images in the source directory
    images = [os.path.join(source_path, fname) for fname in os.listdir(source_path)
              if fname.endswith('.jpg') or fname.endswith('.png')]
    
    num_existing = len(images)
    num_to_generate = num_required - num_existing
    
    # Start generating images if needed
    if num_to_generate > 0:
        os.makedirs(target_path, exist_ok=True)  # Ensure the target directory exists
        for i in range(num_to_generate):
            img_path = np.random.choice(images)
            img = load_img(img_path)  # Load a random image to transform
            x = img_to_array(img)  # Convert image to numpy array
            x = x.reshape((1,) + x.shape)  # Reshape to (1, height, width, channels)

            # Generate batches of randomly transformed images
            for batch in datagen.flow(x, batch_size=1, save_to_dir=target_path, save_prefix='aug', save_format='jpeg'):
                break  # We need only one image per augmentation call

# Paths configuration
base_dir = os.path.join('images_AI_dataset_gender', 'images', 'train')
emotions = ['Focused', 'happy', 'neutral', 'surprise']
genders = ['male', 'female', 'non-binary']

# Number of samples to balance to
target_sample_size = 400  # Adjust based on your requirements

for emotion in emotions:
    for gender in genders:
        source_dir = os.path.join(base_dir, emotion, gender)
        augmented_dir = os.path.join(base_dir, emotion, f"{gender}_augmented")
        augment_images(source_dir, augmented_dir, target_sample_size)