class HashTable:
    def __init__(self, size=1000):
        self.size = size
        self.table = [[] for _ in range(self.size)]

    def _get_hash_index(self, key):
        """Generates a hash index for a given key."""
        # Convert key to a string for consistent hashing, especially for tuples
        key_str = str(key)
        hash_obj = hashlib.sha256(key_str.encode('utf-8'))
        return int(hash_obj.hexdigest(), 16) % self.size

    def insert(self, key, value):
        """Inserts a key-value pair into the hash table."""
        index = self._get_hash_index(key)
        # Handle collisions by storing (key, value) pairs in a list at the index
        # First, remove existing key if present to update its value
        for i, (k, v) in enumerate(self.table[index]):
            if k == key:
                self.table[index][i] = (key, value)
                return
        self.table[index].append((key, value))

    def search(self, key):
        """Searches for a key and returns its associated value."""
        index = self._get_hash_index(key)
        for k, v in self.table[index]:
            if k == key:
                return v
        return None  # Key not found


class HashedNaiveBayes:
    def __init__(self, table_size=1000):
        self.table_size = table_size
        # Hash table to store counts: {class: [0] * table_size}
        self.feature_counts = defaultdict(lambda: [0] * self.table_size)
        self.class_counts = defaultdict(int)

    def _get_hash_index(self, word):
        """Map word to a table index using hashlib."""
        hash_obj = hashlib.sha256(word.encode('utf-8'))
        # Convert hex to int and modulo by table size
        return int(hash_obj.hexdigest(), 16) % self.table_size

    def train(self, features, label):
        self.class_counts[label] += 1
        for word in features:
            index = self._get_hash_index(word)
            self.feature_counts[label][index] += 1

    def predict(self, features):
        # Calculate P(Class|Features) using log-sums to avoid underflow
        # Implementation of Bayes Theorem: P(C|F) ∝ P(C) * ̲ P(Fi|C)
        pass


# --- Naive Bayes Classifier Implementations From Scratch ---


class HashTable:
    def __init__(self):
        self._storage = {}

    def insert(self, key, value):
        self._storage[key] = value

    def search(self, key):
        return self._storage.get(key, None)

    def __len__(self):
        return len(self._storage)

    def __str__(self):
        return str(self._storage)
