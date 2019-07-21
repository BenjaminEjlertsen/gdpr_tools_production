import os, sys
from gdpr_tools_production import preprocessor, word_embedder
from gdpr_tools_production.leipzig_corpora_preprocessor import get_preprocessed_corpora
from gdpr_tools_production.classifiers import DNN
from gdpr_tools_production.nlp_tools import tokenizer
import numpy as np
from sklearn.externals import joblib
from sklearn.feature_extraction import DictVectorizer
from keras.models import load_model
import datetime
from imblearn.under_sampling import RandomUnderSampler
import shutil
import json

class NameRecognizer():

    def __init__(self):
        self.par_dir = os.getcwd() + '/'
        self.config = self.set_configurations()
        self.model_dir = self.par_dir+self.config['paths']['model_folder']
        self.data_dir = self.par_dir+self.config['paths']['data_folder']
        self.clf = None
        self.we_model = None
        self.dv = None

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir+'/temp')
        elif not os.path.exists(self.model_dir+'/temp'):
            os.makedirs(self.model_dir+'/temp')

    def train(self, clf_data_path, clf_model_path = None, we_data_path = None, we_model_path = None, dv_model_path = None,
              epochs = None, batch_size = None, max_tokens = None):

        if batch_size is None:
            batch_size = self.config['training_params']['clf_params']['batch_size']
        if epochs is None:
            epochs = self.config['training_params']['clf_params']['epochs']

        if we_model_path is None:
            self.we_model = self.build_train_we(5)
        else:
            self.we_model = word_embedder.WordEmbedder()
            self.we_model.loadModel(self.model_dir+we_model_path)

        if dv_model_path is None:
            self.dv = self.fit_dict_vectorizer()
            joblib.dump(self.dv, self.model_dir + 'dict_vec')
        else:
            self.dv = joblib.load(self.model_dir + dv_model_path)

        if clf_model_path is None:
            self.clf = DNN((len(self.dv.get_feature_names()) + self.we_model.getModelSize()*3,))
        else:
            self.clf = load_model(self.model_dir+clf_model_path)

        temp_folder = 'temp/'

        for epoch in range(epochs):

            print("Epoch {} of {} epochs".format(epoch + 1, epochs))
            end_of_data = False

            iterations = 0
            start_index = 0

            while not end_of_data:

                #start = datetime.datetime.now()
                print("Training from sample {} to sample {}".format(start_index, start_index + batch_size))
                x, x_w_embeddings, y, end_of_data, end_index = preprocessor.get_dataset(self.data_dir+clf_data_path,
                                                                                        start_index=start_index,
                                                                                        batch_size=batch_size,
                                                                                        we_model=self.we_model)

                if end_of_data:
                    continue

                start_index += end_index-start_index
                x_transformed = self.dv.transform(x)
                # xTrain, xTest, yTrain, yTest = train_test_split(x_transformed, y, test_size=0.2, shuffle=False)

                # maxSamplesToTrainOn = len(xTrain)

                x_w_embeddings = np.array(x_w_embeddings)
                x_w_embeddings = x_w_embeddings.reshape((x_w_embeddings.shape[0], -1))

                x_transformed = np.array(x_transformed)
                x_transformed = np.append(x_transformed, x_w_embeddings, axis=1)

                for idx in range(len(y)):
                    if y[idx] == "PER":
                        y[idx] = 1
                    else:
                        y[idx] = 0

                try:
                    print("Resampling")
                    print(len(y))
                    rus = RandomUnderSampler(0.05)
                    x_transformed, y = rus.fit_resample(x_transformed,y)
                    print(len(y))

                except:

                    continue

                loss = self.clf.model.train_on_batch(x_transformed, y)
                if iterations % 10 == 0:
                    model_path = temp_folder + 'clf_epoch_{}_iter_{}_loss_{}'.format(epoch, iterations, loss[0])
                    self.clf.model.save(self.model_dir+model_path)
                    self.limit_temp_models(self.model_dir+temp_folder, 2)
                    self.predict()
                print(loss)
                iterations += 1

    @staticmethod
    def build_train_we(epochs, min_count = 5, neurons=300, algorithm=0, window_size=10):
        we_model = word_embedder.WordEmbedder(minCount=min_count, neurons=neurons, algorithm=algorithm,
                                              windowSize=window_size)

        corpus_files = []

        for res in os.listdir('name_recognizer/data/LeipzigCorpora'):
            if res.startswith('.'):
                continue
            corpus_files.append(res)

        resource_path = 'name_recognizer/data/LeipzigCorpora/'

        total_sentences = 0
        sentences = None

        for file_num, file in enumerate(corpus_files):
            file = resource_path + file
            print("Loading file: ", file)
            sentences = get_preprocessed_corpora(file)
            total_sentences += len(sentences)
            if file_num < 1:
                update = False
            else:
                update = True

            we_model.buildVocabulary(sentences, updateVocab=update)
            print("Vocabulary Size: ", we_model.getVocabularySize())

        del sentences

        for epoch in range(epochs):
            for file in corpus_files:
                file = resource_path + file
                sentences = get_preprocessed_corpora(file)
                print("Training on file: {}...".format(file), end='')
                sys.stdout.flush()
                we_model.train(sentences, totalExamples=total_sentences, epochs=1)
                print("DONE\n")

        del sentences

        return we_model

    def fit_dict_vectorizer(self):
        start = datetime.datetime.now()
        dv = DictVectorizer(sparse=False)

        start_index = 10000000
        batch_size = 5000000

        print("Fitting vectorizer...")
        x, x_w_embeddings, y, end_of_data, _ = preprocessor.get_dataset(
            self.data_dir+'labeled_data_copy_negative_list_kommune_fix.txt', start_index, batch_size, self.we_model)
        dv.fit(x, y)
        x_transformed = dv.transform(x[:10])

        x_w_embeddings = np.array(x_w_embeddings[:10])
        x_w_embeddings = x_w_embeddings.reshape((x_w_embeddings.shape[0], -1))

        x_transformed = np.array(x_transformed)
        x_transformed = np.append(x_transformed, x_w_embeddings, axis=1)
        features = np.shape(x_transformed)[1]
        print("Features: ", features)
        del x, y
        t_delta = datetime.datetime.now() - start
        print("Fitting done. Fitting time: " + str(t_delta.seconds) + " seconds")

        return dv

    def predict(self, clf_model_path = None, we_model_path = None, dv_model_path = None):

        if we_model_path is not None:
            self.we_model = word_embedder.WordEmbedder()
            self.we_model.loadModel(self.model_dir+we_model_path)

        if dv_model_path is not None:
            self.dv = joblib.load(self.model_dir + dv_model_path)

        if clf_model_path is not None:
            self.clf = load_model(self.model_dir+clf_model_path)

        data_folder = self.data_dir+'/predictions/data_to_predict'

        files_to_predict = os.listdir(data_folder)

        for file in files_to_predict:

            files_to_save_prediction = []
            x = []
            # sentences = []

            if file.startswith('.'):
                continue

            files_to_save_prediction.append(file.split('.')[0] + '_prediction')
            x_w_embeddings = []
            all_words_to_predict = []

            with open(data_folder + '/' + file) as f:

                for line in f.readlines():
                    #print(repr(line))
                    line = line.strip()
                    words = tokenizer(line)

                    if words:
                        for i in range(len(words)):
                            w_embeddings, vectorized_features = preprocessor.to_features(words, i, self.we_model)
                            x_w_embeddings.append(w_embeddings)
                            x.append(vectorized_features)
                            all_words_to_predict.append(words[i])
                            # sentences.append(words)

            # dv = joblib.load('dict_vectorizer.pkl')
            x_trans = self.dv.transform(x)
            x_w_embeddings = np.array(x_w_embeddings)
            x_w_embeddings = x_w_embeddings.reshape((x_w_embeddings.shape[0], -1))
            print(np.shape(x_trans))

            # print(all_words_to_predict)

            x_trans = np.array(x_trans)
            x_trans = np.append(x_trans, x_w_embeddings, axis=1)
            preds = self.clf.model.predict(x_trans)

            predicted_folder = self.data_dir+'predictions/data_predicted'
            if not os.path.exists(predicted_folder):
                os.makedirs(predicted_folder)

            for file_predict in files_to_save_prediction:

                file_save_name = self.save_iterator(predicted_folder,file_predict)

                with open(predicted_folder + '/' + file_save_name + '.txt', 'w') as f:

                    for index, word in enumerate(x_w_embeddings):
                        # print(all_words_to_predict[index])
                        if preds[index] > 0.1:
                            f.write(all_words_to_predict[index] + ' ' + str(preds[index]) + '\n')
                            #f.write(WE.model.wv.most_similar(positive=[x_w_embeddings[index][:300],],topn=1)[0][0]+' '+str(preds[index])+'\n')
                        else:
                            f.write(all_words_to_predict[index] + '\n')
                            #f.write(WE.model.wv.most_similar(positive=[x_w_embeddings[index][:300],],topn=1)[0][0]+'\n')

    def clean_folder(self, path):

        for the_file in os.listdir(path):
            file_path = os.path.join(path, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(e)

    def limit_temp_models(self, path, n_temp_models):

        folder_entries = os.listdir(path)
        folder_entries.sort()
        current_models = 0

        for entry in folder_entries:
            if entry.startswith('clf'):
                current_models += 1

        while current_models > n_temp_models:
            os.remove(path+folder_entries[0])
            folder_entries.pop(0)
            current_models -= 1

    def set_configurations(self):

        if os.path.exists(self.par_dir + 'config.json'):
            with open(self.par_dir + 'config.json', 'r') as fp:
                config = json.load(fp)
        else:
            print("Creating default config file")
            config = {

                "paths": {
                    "model_folder": "models/",
                    "data_folder": "data/",
                    "clf_model_path": "clf_model",
                    "we_model_path": "we_model",
                    "dv_model_path": "dv_model",
                    "clf_data_path": "labeled_data_copy_negative_list_kommune_fix.txt",
                    "we_data_path": "name_recognizer/data/LeipzigCorpora",
                    "dv_data_path": "labeled_data_copy_negative_list_kommune_fix.txt"
                },

                "training_params": {

                    "clf_params": {
                        "epochs": 5,
                        "batch_size": 100000,
                        "use_we": True,
                        "resampling": 0.05,
                        "temp_model_save_interval": 10,
                        "verbose": 1
                    },

                    "we_params": {
                        "hidden_size": 300,
                        "algorithm": 0,
                        "window_size": 10,
                        "min_word_count": 5,
                        "epochs": 1
                    },

                    "dv_params": {
                        "start_index": 10000000,
                        "batch_size": 5000000
                    }
                },

                "prediction_params": {
                    "predict_threshold": 0.5,
                    "file_suffix": "_prediction",
                    "data_to_predict_path": "predictions/data_to_predict/",
                    "predicted_data_path": "predictions/data_predicted/"
                }
            }

            with open(((self.par_dir)) + 'config.json', 'w') as fp:
                json.dump(config, fp, indent=4, sort_keys=True)

        return config

    def save_iterator(self, folder, file_prefix):

        files_content = os.listdir(folder)
        iter_list = []

        for file in files_content:
            if file.startswith(file_prefix):
                iter_list.append(''.join(c for c in file if c.isdigit()))

        iter_list = sorted(iter_list)

        if iter_list:
            return file_prefix+str(int(iter_list[-1])+1)
        else:
            return file_prefix+'1'










