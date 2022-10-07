from Model import Model
import pandas
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


class Trainer:

    def __init__(self, model_path=None):
        if model_path is not None:
            self.model.load_model(model_path)

    def _load_train_data(self, dataset_path):
        train_dataset = pandas.read_csv(dataset_path)
        train_dataset = train_dataset.drop(columns=['Unnamed: 0'])
        return train_dataset

    def _preprocess_train_data(self, train_data):
        reviews = train_data['review_content'].values
        scores = train_data['score'].values
        reviews_train, reviews_test, scores_train, scores_test = train_test_split(
            reviews, scores, test_size=0.33, random_state=1000)
        vectorizer = CountVectorizer()
        vectorizer.fit(reviews_train)
        self.vectorizer = vectorizer
        X_train = vectorizer.transform(reviews_train)
        X_test = vectorizer.transform(reviews_test)
        self.test_data = X_test, scores_test
        return X_train, scores_train

    def train(self, data_path, save_path):
        data = self._load_train_data(data_path)
        x, y = self._preprocess_train_data(data)
        self.model = Model()
        self.model.new(x.shape[1])
        self.model.vectorizer = self.vectorizer
        print(x.shape[0])
        history = self.model.fit(x, y, epochs=64, verbose=True, validation_data=self.test_data, batch_size=1000)
        self.model.save_model(save_path)
        train_loss, train_accuracy = self.model.evaluate(x, y, verbose=True)
        test_loss, test_accuracy = self.model.evaluate(self.test_data[0], self.test_data[1], verbose=True)
        return {'train_loss': train_loss, 'train_accuracy': train_accuracy,
            'test_loss': test_loss, 'test_accuracy': test_accuracy}
