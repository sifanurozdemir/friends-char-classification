import os
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler

def train_cnn_model(train_dir, test_dir, epochs=30, batch_size=32):
    print("Setting up ImageDataGenerators with augmentation...")
    
    # Augmented train and validation generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )
    
    # Test generator (only rescale)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Load directory streams
    train_generator = train_datagen.flow_from_directory(
        directory=train_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        seed=42
    )
    
    val_generator = train_datagen.flow_from_directory(
        directory=train_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        seed=42
    )
    
    test_generator = test_datagen.flow_from_directory(
        directory=test_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    # Class names and indices
    class_indices = train_generator.class_indices
    classes = list(class_indices.keys())
    n_classes = len(classes)
    print(f"Loaded classes: {classes} ({n_classes} classes)")
    
    # MobileNetV2 base model with transfer learning
    print("Building MobileNetV2 Transfer Learning model...")
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Unfreeze the last 15 layers, freeze the rest
    base_model.trainable = True
    for layer in base_model.layers[:-15]:
        layer.trainable = False
        
    # Build classification head on top
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    predictions = Dense(n_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    # Define callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    def lr_decay(epoch, lr):
        if epoch >= 10:
            return lr * 0.95
        return lr
        
    lr_scheduler = LearningRateScheduler(lr_decay)
    
    print("Starting CNN training...")
    # Train using GPU if available
    device_name = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
    with tf.device(device_name):
        history = model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=[early_stopping, lr_scheduler]
        )
        
    # Evaluate model
    print("\nEvaluating CNN model on test dataset...")
    test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
    print(f"\nCNN Test Loss: {test_loss:.4f}")
    print(f"CNN Test Accuracy: {test_accuracy:.4f}")
    
    # Save the model
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "cnn_friends_model.keras")
    model.save(model_path)
    print(f"CNN model saved successfully to: {model_path}")
    
    # Save training curves
    os.makedirs("docs/images", exist_ok=True)
    plt.figure(figsize=(12, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('CNN Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('CNN Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    curves_path = os.path.join("docs", "images", "cnn_training_curves.png")
    plt.savefig(curves_path)
    print(f"CNN Training curves saved to: {curves_path}")
    plt.close()
    
    return test_accuracy

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CNN Transfer Learning Training Script")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs to train.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training.")
    args = parser.parse_args()
    
    train_dir = os.path.join("FriendsDataSet", "train")
    test_dir = os.path.join("FriendsDataSet", "test")
    train_cnn_model(train_dir, test_dir, epochs=args.epochs, batch_size=args.batch_size)
