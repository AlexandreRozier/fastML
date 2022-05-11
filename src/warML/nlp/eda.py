#%%
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
from matplotlib import pyplot as plt
import pandas as pd
from nltk.corpus import stopwords
import numpy as np


#%%


class TextAnalyzer():
    """
    Init with texts, language
    1. ta.analyze()
    2. ta.print_summary()
    """
    def __init__(self, texts:list[str], language:str) -> None:
        self.language = language
        self.stopwords = stopwords.words(language)
        self.texts = texts
        self.vectorizer = CountVectorizer(strip_accents='unicode',stop_words=self.stopwords)
        self.analyze_df:pd.DataFrame 
    
    
    def analyze(self):
        # Result df
        df = pd.DataFrame(dict(text=self.texts))
        df['approx_word_length'] = df.text.str.split(' ').apply(len)
        self.analyze_df = df
        
        # Wordcloud
        X = self.vectorizer.fit_transform(df.text)
        words = self.vectorizer.get_feature_names()
        word_freqs = np.array(X.sum(axis=0)).squeeze()        
        self.wc = WordCloud().fit_words(dict(zip(words, word_freqs)))
        
        
    def show_wc(self):
        plt.figure()
        plt.imshow(self.wc, interpolation="bilinear")
        plt.axis("off")
        plt.show()
        
        
    def get_longest_text(self):
        return self.analyze_df.sort_values('approx_word_length',ascending=False).text.values[0]
        
    def get_shortest_text(self):
        return self.analyze_df.sort_values('approx_word_length').text.values[0]
    
    def print_h_separator(self):
        print("-"*100)
    def print_summary(self):
        self.print_h_separator()

        self.show_wc()
        self.print_h_separator()

        print(self.get_longest_text())
        self.print_h_separator()
        print(self.get_shortest_text())
        self.print_h_separator()

        self.analyze_df.boxplot(column='approx_word_length')

#%%
def main():
    df = pd.read_json('./contraintes_v10_53pct_annotations.jsonlines',lines=True)
    texts=df['item'].apply(lambda it: it['data']['text']).values
    ta = TextAnalyzer(texts, 'french')
    ta.analyze()
    ta.print_summary()


if __name__ == "__main__":
    main()

#%%
# %%
