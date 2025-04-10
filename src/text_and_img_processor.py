import dill
from navec import Navec
from slovnet import NER

from natasha import (
    Segmenter,
    MorphVocab,

    NewsEmbedding,
    NewsMorphTagger,

    Doc
)

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

import numpy as np

import os


def text_and_img_processor(FULL_DF_PATH, DATASET_PATH, NAVEC_PATH, NER_PATH):
    with open(FULL_DF_PATH, "rb") as f:
        try:
            df = dill.load(f)
        except Exception as e:
            print(f"Ошибка загрузки существующих данных: {e}")
    dsc_list = df.desc.tolist()

    navec = Navec.load(NAVEC_PATH)
    ner = NER.load(NER_PATH)
    ner.navec(navec)

    dsc_ner = []

    for elem in dsc_list:
        dsc_ner.append(ner(elem))

    segmenter = Segmenter()
    morph_vocab = MorphVocab()

    emb = NewsEmbedding()
    morph_tagger = NewsMorphTagger(emb)

    def extract_loc_tags(span_markup):
        loc_tags = []
        for span in span_markup.spans:
            if span.type == "LOC":
                loc_word = span_markup.text[span.start:span.stop]

                doc = Doc(loc_word)

                doc.segment(segmenter)

                doc.tag_morph(morph_tagger)

                for token in doc.tokens:
                    token.lemmatize(morph_vocab)
                loc_tags.append(" ".join(token.lemma for token in doc.tokens))
        return {"text": span_markup.text, "tags": loc_tags}

    dsc_tags = [extract_loc_tags(item) for item in dsc_ner]

    docs_str = [' '.join(tags['tags']) for tags in dsc_tags]

    vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split(), preprocessor=lambda x: x)
    tfidf_matrix = vectorizer.fit_transform(docs_str)

    feature_names = vectorizer.get_feature_names_out()
    sum_tfidf = tfidf_matrix.sum(axis=1).A1

    scaler_text = MinMaxScaler()
    text_scores_norm = scaler_text.fit_transform(sum_tfidf.reshape(-1, 1)).flatten()

    result_df = pd.DataFrame({'desc': [tags["text"] for tags in dsc_tags], 'tag': [tags["tags"] for tags in dsc_tags],
                              'text_scores_norm': text_scores_norm})

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # средние значения для ImageNet
            std=[0.229, 0.224, 0.225]  # стандартные отклонения для ImageNet
        )
    ])

    # удаляем классификационную голову, последние слои и переводим в режим .eval()
    model = models.mobilenet_v2(pretrained=True)
    model.classifier = nn.Identity()
    model.eval()

    def extract_image_feature(image_path):
        base_dir = os.path.dirname(__file__)
        normalized_path = os.path.normpath(os.path.join(base_dir, image_path))
        try:
            image = Image.open(normalized_path).convert('RGB')
        except Exception as e:
            print(f"Ошибка загрузки {normalized_path}: {e}")
            return np.zeros(1280)  # размер эмбеддинга MobileNet_v2
        input_tensor = preprocess(image).unsqueeze(0)  # добавляем batch размер 1
        with torch.no_grad():
            features = model(input_tensor)
        return features.squeeze().numpy()

    image_embeddings = np.array([extract_image_feature(path) for path in df['img']])

    pca = PCA(n_components=1)
    image_scores = pca.fit_transform(image_embeddings).flatten()

    scaler_img = MinMaxScaler()
    image_scores_norm = scaler_img.fit_transform(image_scores.reshape(-1, 1)).flatten()
    result_df['image_score'] = image_scores_norm

    alpha = 0.5
    combined_scores = alpha * text_scores_norm + (1 - alpha) * image_scores_norm
    result_df['combined_score'] = combined_scores

    quantiles = np.quantile(combined_scores, [0.2, 0.4, 0.6, 0.8])

    def assign_rank(score):
        if score <= quantiles[0]:
            return 1
        elif score <= quantiles[1]:
            return 2
        elif score <= quantiles[2]:
            return 3
        elif score <= quantiles[3]:
            return 4
        else:
            return 5

    result_df['rank'] = [assign_rank(s) for s in combined_scores]

    with open(DATASET_PATH, "wb") as f:
        try:
            dill.dump(result_df, f)
        except Exception as e:
            print(f"Ошибка загрузки существующих данных: {e}")