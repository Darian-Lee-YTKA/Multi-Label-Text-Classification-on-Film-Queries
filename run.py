# This is a sample Python script.
import os
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import spacy
import re
import unicodedata
import gensim.downloader as api
from nltk import word_tokenize
from gensim.models import KeyedVectors
import nltk
nltk.download('punkt')
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, multilabel_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
nltk.download('punkt')
nltk.download('punkt_tab')

from nltk import word_tokenize

# using a fnn with 4 hidden layers and 1500 units each.
# using leaky relu
class Relation_extraction(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 1500)
        self.fc2 = nn.Linear(1500, 1500)
        self.fc3 = nn.Linear(1500, 1500)
        self.fc4 = nn.Linear(1500, 1000)

        self.relu = nn.LeakyReLU(negative_slope=0.01)
        self.final = nn.Linear(1000, output_dim)
        self.batch_norm1 = nn.BatchNorm1d(1000)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):


      x = self.relu(self.fc1(x)) # note: although its called relu, it reffers to leaky relu. I wrote it this way to have an easier time changing between the 2
      #x = self.batch_norm1(x)
      x = self.dropout(x)

      x = self.relu(self.fc2(x))

      #x = self.batch_norm1(x)
      x = self.dropout(x)

      x = self.relu(self.fc3(x))

      #x = self.batch_norm1(x)
      x = self.dropout(x)

      x = self.fc4(x)

      #x = self.batch_norm1(x)
      x = self.dropout(x)


      x = self.final(x)


      return x
# the main function that does everything basically 
def everything(train, test, output_file):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import CountVectorizer
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    from torch.autograd import Variable
    import torch.nn.functional as F
    import spacy
    import re
    import unicodedata
    import gensim.downloader as api
    from nltk import word_tokenize
    from gensim.models import KeyedVectors
    import nltk
    nltk.download('punkt')
    from sklearn.preprocessing import LabelEncoder
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    from sklearn.model_selection import KFold
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, multilabel_confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns

    # deal with inital preprocessing and simplification of utterances
    train['CORE RELATIONS'] = train['CORE RELATIONS'].apply(lambda x: 'none' if pd.isna(x) else x)

    def process(x):
        x = str(x)
        return x.split(" ")

    train["categories"] = train["CORE RELATIONS"].apply(lambda x: process(x))

    nlp = spacy.load("en_core_web_sm")
# lemmatize 
    def lemmatize_sentence(sentence):
        doc = nlp(sentence)
        return " ".join([token.lemma_ for token in doc])

    train["UTTERANCES"] = train["UTTERANCES"].apply(lemmatize_sentence)
    test["UTTERANCES"] = test["UTTERANCES"].apply(lemmatize_sentence)

    # create one-hot y encodings
    all_classes = set()
    for line in train["categories"]:
        for element in line:
            all_classes.add(element)
    all_classes = list(all_classes)

    for q_class in all_classes:
        train[q_class] = train["categories"].apply(lambda x: 1 if q_class in x else 0)

    # removing extra stop words and accents (some stop words arent included in count vectorizor but still do not add anything in this dataset

    stop_words = ["i", "i'd", "please", "be", "show", "see", "can", "want", "need", "me", "my", "myself", "we", "our",
                  "ours",
                  "ourselves", "you", "your", "yours", "yourself", "yourselves", "it", "to",
                  "am", "is", "are", "was", "were", "been", "being", "have", "has", "had", "having", "do", "does",
                  "they",
                  "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while",
                  "of", "by", "for", "against", "d", "between", "into", "through", "during", "this", "the", "that",
                  "would", "like"]

    def remove_accents(text):

        return ''.join(
            c for c in unicodedata.normalize('NFD', text)
            if unicodedata.category(c) != 'Mn'
        )

    def remove_stop_words(text):

        text = remove_accents(text)

        text = text.replace("'s", "")

        stop_words_pattern = r'\b(?:' + '|'.join(re.escape(word) for word in stop_words) + r')\b'

        cleaned_text = re.sub(stop_words_pattern, ' ', text, flags=re.IGNORECASE)

        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

        return cleaned_text

    train["UTTERANCES"] = train["UTTERANCES"].apply(remove_stop_words)
    test["UTTERANCES"] = test["UTTERANCES"].apply(remove_stop_words)
    train = train.sample(frac=1, random_state=12).reset_index(drop=True)

    # split up the x and y
    x_test = test["UTTERANCES"]
    x_train = train["UTTERANCES"]
    y_train = train[list(all_classes)]

    # character trigrams

    vectorizer = CountVectorizer(stop_words='english', max_features=1000, ngram_range=(1, 3), analyzer='char_wb')

    vectorizer.fit(x_train)

    x_train_char3 = vectorizer.transform(x_train).toarray()

    x_test_char3 = vectorizer.transform(x_test).toarray()

    wv1 = api.load('glove-twitter-200')

    # normalizing the trigrams

    x_train_char3 = torch.FloatTensor(x_train_char3)

    x_test_char3 = torch.FloatTensor(x_test_char3)

    mean, std = x_train_char3.mean(), x_train_char3.std()
    x_train_char3 = (x_train_char3 - mean) / std
    x_test_char3 = (x_test_char3 - mean) / std

    # do the embeddings
    text_embeddings = []
    print("I advanced to embeddings")
    for text in train["UTTERANCES"]:

        token_embeddings = []
        for token in word_tokenize(text.lower()):
            if token in wv1:
                token_embeddings.append(wv1[token])
            else:
                continue #ignore OOV tokens 

        text_embedding = np.array(token_embeddings).mean(axis=0) #mean pooling
        text_embeddings.append(text_embedding)
    x_train_embed = np.array(text_embeddings)

    text_embeddings = []
    for text in test["UTTERANCES"]:

        token_embeddings = []
        for token in word_tokenize(text.lower()):
            if token in wv1:
                token_embeddings.append(wv1[token])
            else:
                continue
        if len(token_embeddings) != 0:
            text_embedding = np.array(token_embeddings).mean(axis=0)

            text_embeddings.append(text_embedding)
        else:
            text_embedding = np.zeros(200) # there is one utterance in the test set which consists entirely of OOVs. This line of code exists to handle that edge case
            text_embeddings.append(text_embedding)
    x_test_embed = np.array(text_embeddings)

    # make y a tensor
    train_y_np = y_train.values

    y_train = torch.FloatTensor(train_y_np)

    # concat both tensors
    x_train_embed_tensor = torch.tensor(x_train_embed)
    x_test_embed_tensor = torch.tensor(x_test_embed)

    x_train = torch.cat((x_train_embed_tensor, x_train_char3), dim=1)

    x_test = torch.cat((x_test_embed_tensor, x_test_char3), dim=1)

    # create model
    print("I advanced to model building")
    model = Relation_extraction(x_train.size(1), output_dim=19)
#using 5 kfolds and 25 epochs 
    k_folds = 5
    num_epoch = 25
    kfold = KFold(n_splits=k_folds, shuffle=True)
    train_losses = []
    val_losses = []
    val_accuracies = []
    val_precisions = []
    val_recalls = []
    val_f1_scores = []
# saving the best models based on f1, acc, and loss
# F1 gives best results, so although 3 are saved, the final eval function will run on just F1
    best_val_loss = float('inf')
    best_val_acc = 0
    best_val_F1 = 0
    best_model_state_loss = None
    best_model_state_acc = None
    best_model_state_F1 = None

    train_dataset = TensorDataset(x_train, y_train)

    train_loader = DataLoader(train_dataset, batch_size=64, pin_memory=True, shuffle=True)

    for fold, (train_i, val_i) in enumerate(kfold.split(train_dataset)):
        print(f'Fold {fold + 1}')

        train_subset = torch.utils.data.Subset(train_dataset, train_i)
        val_subset = torch.utils.data.Subset(train_dataset, val_i)

        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=32)

        model = Relation_extraction(x_train.size(1), output_dim=19)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0006)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        train_loss = []
        val_loss = []
        val_acc = []

        for epoch in range(num_epoch):
            running_loss = torch.tensor(0.)
            etrain_loss = []
            eval_loss = []

            model.train()
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.float()
                y_batch = y_batch.float()

                optimizer.zero_grad()
                outputs = model(x_batch)
                loss = loss_fn(outputs, y_batch)
                loss.backward()
                optimizer.step()

                running_loss += loss
                etrain_loss.append(loss.item())

            model.eval()
            with torch.no_grad():
                total_correct = 0
                total_samples = 0
                all_preds = []
                all_labels = []

                for x_batch, y_batch in val_loader:
                    x_batch = x_batch.float()
                    y_batch = y_batch.float()

                    outputs = model(x_batch)
                    predicted = (outputs > 0.5).int()
                    loss = loss_fn(outputs, y_batch)
                    eval_loss.append(loss.item())
# choose every output that is over .5
                    for i in range(outputs.shape[0]):
                        if (outputs[i] > 0.5).any():
                            continue
                        else:
                            # if none are over .5, choose the highest one
                            max_index = outputs[i].argmax().item()
                            predicted[i, max_index] = 1

                    total_correct += (predicted == y_batch).all(dim=1).sum()
                    total_samples += y_batch.size(0)

                    all_preds.append(predicted)
                    all_labels.append(y_batch)

                all_preds = torch.cat(all_preds).cpu().numpy()
                all_labels = torch.cat(all_labels).cpu().numpy()

                acc = total_correct.float() / total_samples

            current_val_loss = np.mean(eval_loss)

            precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
            recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
            f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

            val_precisions.append(precision)
            val_recalls.append(recall)
            val_f1_scores.append(f1)

            if current_val_loss < best_val_loss:

                best_val_loss = current_val_loss
                best_model_state_loss = model.state_dict()

            if acc > best_val_acc:

                best_val_acc = acc
                best_model_state_acc = model.state_dict()


            if f1 > best_val_F1:

                best_val_F1 = f1
                best_model_state_F1 = model.state_dict()


            print(
                f'Epoch {epoch + 1}, Loss: {running_loss.item() / len(train_loader)}, Val Acc: {acc.item()}, Val Loss: {current_val_loss}, Val Precision: {precision}, Val Recall: {recall}, Val F1: {f1}')

        train_losses.append(np.mean(etrain_loss))
        val_losses.append(np.mean(eval_loss))
        val_accuracies.append(acc)
# save best models at end 
    model_dir = '../PycharmProjects/run.py/models'
    os.makedirs(model_dir, exist_ok=True)


    if best_model_state_acc is not None:
        torch.save(best_model_state_acc, os.path.join(model_dir, '../PycharmProjects/run.py/best_model_acc.pth'))

    if best_model_state_F1 is not None:
        torch.save(best_model_state_F1, os.path.join(model_dir, '../PycharmProjects/run.py/best_model_f1.pth'))

    if best_model_state_loss is not None:
        torch.save(best_model_state_loss, os.path.join(model_dir, '../PycharmProjects/run.py/best_model_loss.pth'))

    print(f'Average Train Loss: {np.mean(train_losses)}')
    print(f'Average Val Loss: {np.mean(val_losses)}')
    print(f'Average Val Accuracy: {np.mean(val_accuracies)}')
    print(f'Average Val Precision: {np.mean(val_precisions)}')
    print(f'Average Val Recall: {np.mean(val_recalls)}')
    print(f'Average Val F1 Score: {np.mean(val_f1_scores)}')

    model_dir = '../PycharmProjects/run.py/models'

# load best models 

    model_f1 = Relation_extraction(x_train.size(1), output_dim=19)
    model_f1.load_state_dict(torch.load(os.path.join(model_dir, '../PycharmProjects/run.py/best_model_f1.pth')))

    model_acc = Relation_extraction(x_train.size(1), output_dim=19)
    model_acc.load_state_dict(torch.load(os.path.join(model_dir, '../PycharmProjects/run.py/best_model_acc.pth')))

    model_loss = Relation_extraction(x_train.size(1), output_dim=19)
    model_loss.load_state_dict(torch.load(os.path.join(model_dir, '../PycharmProjects/run.py/best_model_loss.pth')))

    count = 0
# this function ensures the csv file is sorted alphabetically 
    def sort_words(relations):

        new_words = sorted(relations.split())
        return ' '.join(new_words)

    def eval_model(model, mode):
        print("\n===========================", mode, "===========================\n")
        count = 0
        model.eval()
        with torch.no_grad():

            outputs = model(x_test.float())

            predicted = (outputs > 0.5).int()

            for i in range(outputs.shape[0]):
                if (outputs[i] > 0.5).any():
                    continue
                else:
                    # if no outputs are > .5, we use arg max 
                    print("we are using an arg max")
                    max_index = outputs[i].argmax().item()
                    max_value = outputs[i].max().item()
                    print(max_value)
                    predicted[i, max_index] = 1
                    count += 1
            none_index = 0
            for index in range(len(all_classes)):
                if all_classes[index] == "none":
                    none_index = index
            for i in range(predicted.shape[0]):
                # dont let none cooccur with another prediction
                if predicted[i, none_index] == 1:
                    for j in range(predicted.shape[1]):
                        if j != none_index and predicted[i, j] == 1:
                            print("it happened")
                            print(i)
                            predicted[i, none_index] = 0

                            break
        # just formatting stuff 
        print("🙋‍♀️🙋‍♀️🙋‍♀️ percent that was argmax: ", count / outputs.shape[0])
        predicted_np = predicted.numpy()

        results_df = pd.DataFrame()

        for index, class_name in enumerate(all_classes):
            results_df[class_name] = predicted_np[:, index]

        results_df["categories"] = ""

        for category in all_classes:
            results_df["categories"] = results_df["categories"] + np.where(results_df[category] == 1, category + " ",
                                                                           "")
        results_df["categories"] = results_df["categories"].apply(lambda x: x[:-1])
        final_df = pd.DataFrame()
        final_df["ID"] = range(0, 981)
        final_df["Core Relations"] = results_df["categories"]
        final_df["Core Relations"] = final_df["Core Relations"].apply(sort_words)

        print(final_df)
        title = mode
        final_df.to_csv(title, index=False)

    eval_model(model_f1, output_file)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python run.py <train_data> <test_data> <output_file>")
        sys.exit(1)

    train_data = sys.argv[1]
    test_data = sys.argv[2]
    output_file = sys.argv[3]

    train = pd.read_csv(train_data)
    test = pd.read_csv(test_data)

    everything(train, test, output_file)





