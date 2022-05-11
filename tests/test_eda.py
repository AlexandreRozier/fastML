
import pandas as pd
from src.warML.nlp.eda import TextAnalyzer


def test_text_analyzer():
    texts = ["J'aime les pommes.","Les pommes c'est délicieux !","Je préfère les poires..."]
    ta = TextAnalyzer(texts, 'french')
    ta.analyze()

    ta.print_summary()
    assert True