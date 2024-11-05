import numpy as np
from collections import Counter
from treenode import TreeNode

class DecisionTree:
    
    def __init__(self, max_depth, min_samples_leaf, min_information_gain, numb_of_features_splitting):
        """Initialize the decision tree with specified parameters."""
        self.max_depth = max_depth 
        self.min_samples_leaf = min_samples_leaf
        self.min_information_gain = min_information_gain
        self.numb_of_features_splitting = numb_of_features_splitting

    def _entropy(self, probability):
        """Calculate entropy based on a list of probabilities."""
        entropy_sum = 0
        for p in probability:
            if p > 0:
                entropy_sum += -p * np.log2(p)
        return entropy_sum

    def _class_probabilities(self, Y_train):
        """Calculate the probabilities of each category."""
        total_count = len(Y_train)
        category_counts = Counter(Y_train)
        probabilities = []
        for count in category_counts.values():
            probabilities.append(count / total_count)
        return probabilities

    def _data_entropy(self, Y_train: list) -> float:
        """Calculate entropy for a list of categories."""
        class_probs = self._class_probabilities(Y_train)
        data_entropy = self._entropy(class_probs)
        return data_entropy
    
    def _partition_entropy(self, subsets: list) -> float:
        """Calculate the weighted entropy for a list of subsets of target values."""
        total_count = sum(len(subset) for subset in subsets)
        weighted_entropy = 0
        for subset in subsets:
            subset_entropy = self._data_entropy(subset)
            subset_weight = len(subset) / total_count
            weighted_entropy += subset_entropy * subset_weight
        return weighted_entropy

    def _split(self, X_train: np.array, Y_train: np.array, feature_idx: int, feature_val: float) -> tuple:
        """Split the data into two subsets based on a feature threshold."""
        mask = X_train[:, feature_idx] < feature_val
        X_subset1, Y_subset1 = X_train[mask], Y_train[mask]
        X_subset2, Y_subset2 = X_train[~mask], Y_train[~mask]
        return X_subset1, Y_subset1, X_subset2, Y_subset2

    def _select_features_to_use(self, X_train: np.array) -> list:
        """Select features to consider for splitting."""
        total_features = list(range(X_train.shape[1]))
        num_features = len(total_features)

        if self.numb_of_features_splitting == "sqrt":
            num_selected = int(np.sqrt(num_features))
            selected_features = np.random.choice(total_features, size=num_selected, replace=False)
        elif self.numb_of_features_splitting == "log":
            num_selected = int(np.log2(num_features))
            selected_features = np.random.choice(total_features, size=num_selected, replace=False)
        else:
            selected_features = total_features
        return selected_features

    def _find_best_split(self, X_train: np.array, Y_train: np.array) -> tuple:
        """Find the best feature and threshold to split data based on minimum entropy."""
        best_entropy = float('inf')
        best_split = None
        features_to_check = self._select_features_to_use(X_train)

        for feature_idx in features_to_check:
            thresholds = np.percentile(X_train[:, feature_idx], [25, 50, 75])

            for threshold in thresholds:
                X_subset1, Y_subset1, X_subset2, Y_subset2 = self._split(X_train, Y_train, feature_idx, threshold)
                subsets = [Y_subset1, Y_subset2]

                entropy = self._partition_entropy(subsets)
                if entropy < best_entropy:
                    best_entropy = entropy
                    best_split = (X_subset1, Y_subset1, X_subset2, Y_subset2, feature_idx, threshold)

        return (*best_split, best_entropy)

    def _find_category_probabilities(self, Y_train: np.array) -> np.array:
        """Calculate probabilities for each category in the labels."""
        total_labels = len(Y_train)
        category_probabilities = np.zeros(len(self.categories_in_train), dtype=float)

        for i, category in enumerate(self.categories_in_train):
            category_count = np.sum(Y_train == category)
            category_probabilities[i] = category_count / total_labels

        return category_probabilities

    def _create_tree(self, X_train: np.array, Y_train: np.array, depth: int = 0) -> TreeNode:
        """Recursive depth-first tree creation."""
        if depth > self.max_depth:
            return None
        
        X_subset1, Y_subset1, X_subset2, Y_subset2, feature_idx, feature_val, split_entropy = self._find_best_split(X_train, Y_train)
        category_probs = self._find_category_probabilities(Y_train)
        info_gain = self._entropy(category_probs) - split_entropy
        node = TreeNode((X_train, Y_train), feature_idx, feature_val, category_probs, info_gain)

        if min(len(Y_subset1), len(Y_subset2)) < self.min_samples_leaf or info_gain < self.min_information_gain:
            return node

        node.left = self._create_tree(X_subset1, Y_subset1, depth + 1)
        node.right = self._create_tree(X_subset2, Y_subset2, depth + 1)
        return node

    def _predict_one_sample(self, sample: np.array) -> np.array:
        """Predicts probability for a single sample"""
        node = self.tree
        while node:
            pred_probs = node.prediction_probs
            if sample[node.feature_idx] < node.feature_val:
                node = node.left
            else:
                node = node.right
        return pred_probs

    def train(self, X_train: np.array, Y_train: np.array) -> None:
        """Trains the model on X_train and Y_train datasets."""
        self.categories_in_train = np.unique(Y_train)
        self.tree = self._create_tree(X_train, Y_train)
        
        self.feature_importances = {}
        for i in range(X_train.shape[1]):
            self.feature_importances[i] = 0

        self._calculate_feature_importance(self.tree)
        total_importance = sum(self.feature_importances.values())

        for k, v in self.feature_importances.items():
            self.feature_importances[k] = v / total_importance

    def predict_proba(self, X_set: np.array) -> np.array:
        """Returns predicted probabilities for each sample in X_set."""
        pred_probs = np.apply_along_axis(self._predict_one_sample, 1, X_set)
        return pred_probs

    def predict(self, X_set: np.array) -> np.array:
        """Predicts labels for each sample in X_set."""
        pred_probs = self.predict_proba(X_set)
        preds = np.argmax(pred_probs, axis=1)
        return preds

    def _calculate_feature_importance(self, node: TreeNode) -> None:
        """Calculates feature importance by recursively traversing the tree."""
        if node:
            self.feature_importances[node.feature_idx] += node.feature_importance
            self._calculate_feature_importance(node.left)
            self._calculate_feature_importance(node.right)
