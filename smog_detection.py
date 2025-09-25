
# Build Pollution Classifier Using Pre-trained MobileNetV2
def build_classifier(input_shape, num_classes):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)

    # Fine-tune the last 20 layers
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Enhanced Smog Detection Using Pixel Intensity
def detect_smog_area(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    black_pixel_percentage = np.sum(gray < 50) / gray.size * 100  # Count black pixels (threshold < 50)
    return black_pixel_percentage

# Predict Smog Intensity
def predict_smog(image_path, classifier, img_size):
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, img_size)
    img_normalized = np.expand_dims(img_resized / 255.0, axis=0)

    # Smog Intensity Detection
    smog_intensity = detect_smog_area(img)

    # Display Smog Intensity Only
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"Smog Intensity: {smog_intensity:.2f}%")
    plt.axis('off')
    plt.show()

# Plot Training and Validation Loss and Accuracy
def plot_training_history(history):
    epochs = range(len(history.history['loss']))

    # Plot Loss
    plt.figure(figsize=(5, 4))
    plt.plot(epochs, history.history['loss'], 'r', label='Training Loss')
    plt.plot(epochs, history.history['val_loss'], 'b', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Plot Accuracy
    plt.figure(figsize=(5, 4))
    plt.plot(epochs, history.history['accuracy'], 'r', label='Training Accuracy')
    plt.plot(epochs, history.history['val_accuracy'], 'b', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# Main Function
def main():
    # Hyperparameters
    img_size = (224, 224)
    batch_size = 32
    epochs = 85

    # Paths for Training and Testing Data
    train_path = "/content/drive/MyDrive/Training Car"
    test_path = "/content/drive/MyDrive/Testing Car"

    # Train Pollution Classifier
    print("Training pollution classifier...")
    train_datagen = ImageDataGenerator(rescale=1.0 / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        train_path, target_size=img_size, batch_size=batch_size, class_mode='categorical'
    )

    test_generator = test_datagen.flow_from_directory(
        test_path, target_size=img_size, batch_size=batch_size, class_mode='categorical'
    )

    input_shape = (img_size[0], img_size[1], 3)
    num_classes = len(train_generator.class_indices)

    classifier = build_classifier(input_shape, num_classes)

    # Compute class weights to handle class imbalance
    class_weights_array = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_generator.classes),
        y=train_generator.classes
    )

    # Convert to dictionary
    class_weights = {i: weight for i, weight in enumerate(class_weights_array)}

    # Add Early Stopping Callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = classifier.fit(
        train_generator,
        epochs=epochs,
        validation_data=test_generator,
        class_weight=class_weights,
        callbacks=[early_stopping]
    )

    classifier.save("pollution_classifier_v2.h5")

    # Plot Training History
    plot_training_history(history)

    # Predict Using Uploaded Image
    image_path = "/content/drive/MyDrive/Vehicle Sm.jpg"
    if image_path:
        print(f"Processing image: {image_path}")
        predict_smog(image_path, classifier, img_size)
    else:
        print("No image selected. Exiting...")

if __name__ == "__main__":
    main()
