"""
Drug Classification using Decision Trees and TF-IDF Vectorization

This script implements a text classification system for drug classification that:
1. Loads training data with drug labels and text descriptions
2. Preprocesses text using tokenization and TF-IDF vectorization
3. Applies feature selection using variance threshold
4. Trains a Decision Tree classifier
5. Evaluates the model on test data and saves predictions

The system uses natural language processing techniques to classify drugs
based on their textual descriptions.
"""

from sklearn import tree
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import VarianceThreshold
from nltk.tokenize import TweetTokenizer


def load_training_data(filename):
    """
    Load and preprocess training data from file.
    
    Args:
        filename (str): Path to the training data file
        
    Returns:
        tuple: (labels, text_descriptions) where labels are drug classifications
               and text_descriptions are the processed text features
    """
    print("Loading and processing training data...")
    
    labels = []
    text_descriptions = []
    tokenizer = TweetTokenizer()
    
    with open(filename, "r") as file:
        for line in file:
            # Extract label from first character
            label = line[0]
            # Extract and tokenize text description (skip first 2 characters)
            text = " ".join(tokenizer.tokenize(line[2:]))
            
            labels.append(label)
            text_descriptions.append(text)
    
    print(f"Loaded {len(labels)} training samples")
    return labels, text_descriptions


def load_test_data(filename):
    """
    Load and preprocess test data from file.
    
    Args:
        filename (str): Path to the test data file
        
    Returns:
        list: Processed text descriptions for testing
    """
    print("Loading and processing test data...")
    
    text_descriptions = []
    tokenizer = TweetTokenizer()
    
    with open(filename, "r") as file:
        for line in file:
            # Tokenize the entire line for test data
            text = " ".join(tokenizer.tokenize(line))
            text_descriptions.append(text)
    
    print(f"Loaded {len(text_descriptions)} test samples")
    return text_descriptions


def create_tfidf_features(train_texts, test_texts=None, max_df=0.15, min_df=0.0013):
    """
    Create TF-IDF features from text data.
    
    Args:
        train_texts (list): Training text descriptions
        test_texts (list, optional): Test text descriptions
        max_df (float): Maximum document frequency for TF-IDF
        min_df (float): Minimum document frequency for TF-IDF
        
    Returns:
        tuple: (train_features, test_features, vectorizer) or (train_features, vectorizer) if no test data
    """
    print("Creating TF-IDF features...")
    
    vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df)
    train_features = vectorizer.fit_transform(train_texts)
    
    print(f"Training data dimensions: {train_features.shape}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    
    if test_texts is not None:
        test_features = vectorizer.transform(test_texts)
        print(f"Test data dimensions: {test_features.shape}")
        return train_features, test_features, vectorizer
    
    return train_features, vectorizer


def apply_feature_selection(train_features, test_features=None, variance_threshold=0.0):
    """
    Apply variance threshold feature selection.
    
    Args:
        train_features: Training feature matrix
        test_features: Test feature matrix (optional)
        variance_threshold (float): Threshold for variance-based feature selection
        
    Returns:
        tuple: (selected_train_features, selected_test_features, selector) or 
               (selected_train_features, selector) if no test features
    """
    print("Applying feature selection...")
    
    selector = VarianceThreshold(variance_threshold)
    selected_train_features = selector.fit_transform(train_features)
    
    print(f"Features after selection: {selected_train_features.shape[1]}")
    print(f"Features removed: {train_features.shape[1] - selected_train_features.shape[1]}")
    
    if test_features is not None:
        selected_test_features = selector.transform(test_features)
        return selected_train_features, selected_test_features, selector
    
    return selected_train_features, selector


def train_decision_tree(features, labels, min_samples_leaf=7):
    """
    Train a Decision Tree classifier.
    
    Args:
        features: Training feature matrix
        labels: Training labels
        min_samples_leaf (int): Minimum samples required at a leaf node
        
    Returns:
        sklearn.tree.DecisionTreeClassifier: Trained classifier
    """
    print("Training Decision Tree classifier...")
    
    classifier = tree.DecisionTreeClassifier(min_samples_leaf=min_samples_leaf)
    classifier.fit(features, labels)
    
    print(f"Decision Tree trained successfully:")
    print(f"  - Number of leaf nodes: {classifier.get_n_leaves()}")
    print(f"  - Tree depth: {classifier.get_depth()}")
    print(f"  - Number of features used: {features.shape[1]}")
    
    return classifier


def make_predictions(classifier, test_features):
    """
    Make predictions on test data.
    
    Args:
        classifier: Trained classifier
        test_features: Test feature matrix
        
    Returns:
        list: Predictions for test data
    """
    print("Making predictions on test data...")
    
    predictions = classifier.predict(test_features)
    print(f"Generated {len(predictions)} predictions")
    
    return predictions


def save_predictions(predictions, filename="results.dat"):
    """
    Save predictions to a file in binary format (0/1).
    
    Args:
        predictions (list): Model predictions
        filename (str): Output filename
    """
    print(f"Saving predictions to {filename}...")
    
    with open(filename, "w") as result_file:
        for prediction in predictions:
            # Convert prediction to binary (0 or 1)
            binary_prediction = "1" if prediction != "0" else "0"
            result_file.write(f"{binary_prediction}\n")
    
    print("Predictions saved successfully!")


def display_model_summary(classifier, train_features, train_labels):
    """
    Display a summary of the trained model.
    
    Args:
        classifier: Trained classifier
        train_features: Training features
        train_labels: Training labels
    """
    print("\nModel Summary:")
    print("=" * 40)
    print(f"Classifier Type: Decision Tree")
    print(f"Training Samples: {len(train_labels)}")
    print(f"Feature Dimensions: {train_features.shape[1]}")
    print(f"Unique Classes: {len(set(train_labels))}")
    print(f"Tree Depth: {classifier.get_depth()}")
    print(f"Number of Leaves: {classifier.get_n_leaves()}")
    
    # Calculate class distribution
    class_counts = {}
    for label in train_labels:
        class_counts[label] = class_counts.get(label, 0) + 1
    
    print(f"Class Distribution:")
    for class_label, count in sorted(class_counts.items()):
        percentage = (count / len(train_labels)) * 100
        print(f"  Class {class_label}: {count} samples ({percentage:.1f}%)")


def main():
    """Main function to orchestrate the drug classification process."""
    print("Drug Classification using Decision Trees")
    print("=" * 50)
    
    # Get input filenames
    train_filename = input("Enter the training file name: ")
    test_filename = input("Enter the test file name: ")
    
    try:
        # Load and preprocess data
        train_labels, train_texts = load_training_data(train_filename)
        test_texts = load_test_data(test_filename)
        
        # Create TF-IDF features
        train_features, test_features, vectorizer = create_tfidf_features(
            train_texts, test_texts
        )
        
        # Apply feature selection
        selected_train_features, selected_test_features, selector = apply_feature_selection(
            train_features, test_features
        )
        
        # Train the classifier
        classifier = train_decision_tree(selected_train_features, train_labels)
        
        # Display model summary
        display_model_summary(classifier, selected_train_features, train_labels)
        
        # Make predictions
        predictions = make_predictions(classifier, selected_test_features)
        
        # Save results
        save_predictions(predictions)
        
        print("\nClassification complete!")
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()