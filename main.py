from A.Pneumonia import PneumoniaMNISTClassifier
from B.Path import PathMNISTClassifier


if __name__ == "__main__":
    print("Please press 1 for the Pleumonia Classifier or 2 for the Path Classifier")
    user_input = input("Enter your choice: ")

    if user_input == "1":
        print("Starting the Pneumonia Classifier...")
        Pneumonia_classifier = PneumoniaMNISTClassifier('Dataset\pneumoniamnist.npz')
    elif user_input == "2":
        print("Starting the Path Classifier...")
        Path_classifier = PathMNISTClassifier('Dataset\pathmnist.npz')
    else:
        print("Invalid choice. Exiting program...")
