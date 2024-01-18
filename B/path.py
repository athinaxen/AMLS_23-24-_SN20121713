import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class PathMNISTClassifier:
    def __init__(self, data_path):
        #Define self values and functions
        self.data_path = data_path
        self.x_train, self.y_train, self.x_val, self.y_val, self.x_test, self.y_test, self.datagen = self.process_data()
        self.model = self.CNNmodel()
        self.history = self.train_model(epochs=50, batch_size=32)
        self.test_accuracy = self.test_model()
        self.plot_results()

    def process_data(self):
        #Load the Pneumonia Dataset through the path and assign the training/validation/test data to values
        path_mnist_data = np.load(self.data_path)

        #Where x are the data images and y the labels
        x_train, y_train = path_mnist_data['train_images'], path_mnist_data['train_labels']
        x_val, y_val = path_mnist_data['val_images'], path_mnist_data['val_labels']
        x_test, y_test = path_mnist_data['test_images'], path_mnist_data['test_labels']

        #Normalise the pixel values of all the images to be between 0-1 
        x_train = x_train.astype('float32')/255.0
        x_val = x_val.astype('float32')/255.0
        x_test = x_test.astype('float32')/255.0

        # Change the class labels with one got categorical coding
        y_train = to_categorical(y_train, num_classes=9)
        y_val = to_categorical(y_val, num_classes=9)
        y_test = to_categorical(y_test, num_classes=9)

        # Test that the shape of training data is correct before proceeding
        #print('x_train shape: ',x_train.shape)
        #print('y train shape', y_train.shape)
        #print('labels', y_train[3])

        # ---------Data Augmentation-----------------------------------
        # Create data generator with augmentation
        datagen = ImageDataGenerator(
            #Flipping, cropping, rotation, and shifting
            rotation_range=20,
            width_shift_range=0.1,  # Randomly shift images horizontally by 10%
            height_shift_range=0.1,  # Randomly shift images vertically by 10%
            zoom_range=0.1,  # Randomly zoom images by 10%
            horizontal_flip=True,  # Randomly flip images horizontally
            vertical_flip=True,  # Randomly flip images vertically
        )


        return x_train, y_train, x_val, y_val, x_test, y_test, datagen

    def CNNmodel(self):
        #Define the optimizer of the model
        optimizer_Adam = Adam(learning_rate=0.001)
        #Create a Sequential CNN model 
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3))) #28x28x3
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(9, activation='softmax'))  # Multi-class classification with softmax activation

        #Compiling the model using the Adam optimizer and the binary crossentropy
        model.compile(optimizer=optimizer_Adam, loss='categorical_crossentropy', metrics=['accuracy'])
        #Printing the model summary to see the trainable parameters
        print(model.summary())

        return model

    def train_model(self, epochs=50, batch_size=32):
        print('Training Starting...\n')

        #ReduceLROnPlateu monitors the validation loss and proceeds with reducing the learning rate if necessary
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-6, verbose=1)

        #Early Stopping will stop the model training if the validation loss does not improve
        early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, mode='auto')

        #Training the model using the specified epochs and batch_size
        history = self.model.fit(self.x_train, self.y_train, epochs=epochs, batch_size=batch_size,
                               validation_data=(self.x_val, self.y_val), callbacks=[reduce_lr,early_stop], verbose=1)
        
        #------------Training with Data Augmentation----------------------

        #history = self.model.fit(
        #    self.datagen.flow(self.x_train, self.y_train, batch_size=batch_size, shuffle=True),
        #    steps_per_epoch=len(self.x_train) // batch_size,
        #    epochs=epochs,
        #    validation_data=(self.x_val, self.y_val),
        #    callbacks=[reduce_lr,early_stop],
        #    verbose=1
        #)

        return history

    def test_model(self):
        #Evaluating the model using the test data
        _, test_accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print(f'Test Accuracy: {test_accuracy:.4f}')
        return test_accuracy

    def plot_results(self):
        # Plot training & validation accuracy values
        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()

        # Plot training & validation loss values
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()

        # Plot confusion matrix
        y_pred = np.argmax(self.model.predict(self.x_test), axis=1)
        y_test_argmax = np.argmax(self.y_test, axis=1)
        cm = confusion_matrix(y_test_argmax, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[str(i) for i in range(9)],
                    yticklabels=[str(i) for i in range(9)])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
