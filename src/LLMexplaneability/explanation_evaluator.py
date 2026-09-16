import math
from typing import Iterable, List, Tuple, Dict
from collections import Counter
import spacy
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


class ExplanationEvaluator:
    def __init__(self, text):
        self.text = text
        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])
        self.doc = self.nlp(self.text)
        self.tokens = self._spacy_tokens_en()

    def score_text(self):
        pass

    def _spacy_tokens_en(self, text = None, remove_stopwords = True, keep_negations = False, only_alpha = True) -> List[str]:
        """
        Tokenize using spaCy:
        - lowercase (token.lower_)
        - remove stopwords (spaCy) when remove_stopwords=True
        - optional keeps 'not', 'no'
        - filteralphabetically when only_alpha=True
        """
        doc = self.doc if text == None else self.nlp(text)
        out = []
        for tok in doc:
            if tok.is_space:
                continue
            if only_alpha and not tok.is_alpha:
                continue
            t = tok.lower_
            if remove_stopwords:
                if keep_negations and t in {"no", "not", "n't"}:
                    pass  
                elif tok.is_stop:
                    continue
            out.append(t)
        return out

class DistinctN(ExplanationEvaluator):
    def __init__(self, text, n):
        super().__init__(text)
        self.n = n

    def _make_ngrams(self, tokens: List[str]) -> List[Tuple[str, ...]]:
        if self.n <= 0:
            raise ValueError("n most be >= 1")
        return [tuple(tokens[i:i+self.n]) for i in range(len(tokens) - self.n + 1)]

    def distinct_n(self) -> float:
        ngrams = self._make_ngrams(self.tokens)
        if not ngrams:
            return 0.0
        return len(set(ngrams)) / len(ngrams)

    def sentence_level_distinct(self) -> float:
        """
        Mean sentence distinct-n
        """
        vals = []
        for sent in self.doc.sents:
            tokens = self._spacy_tokens_en(text = sent)
            vals.append(self.distinct_n_from_tokens(tokens))
        
        vals = [v for v in vals if not math.isnan(v)]
        return sum(vals)/len(vals) if vals else 0.0
    
    def score_text(self):
        return (self.distinct_n(), self.sentence_level_distinct())


class SelfBLEU(ExplanationEvaluator):
    """
    Self-BLEU: para cada oración (hipótesis), computa BLEU contra el conjunto
    de referencias formado por TODAS las otras oraciones del mismo texto.
    Devuelve el promedio (cuanto más ALTO, menos diversidad; cuanto más BAJO, más diversidad).
    """
    def __init__(self, text: str, n: int = 4):
        super().__init__(text)
        if n <= 0:
            raise ValueError("n debe ser >= 1")
        self.n = n
        self._smooth = SmoothingFunction().method1  # método de smoothing estándar


    def _sent_tokens(self) -> List[List[str]]:
        return [self._spacy_tokens_en(sent) for sent in self.doc.sents]

    def self_bleu(self) -> float:
        """
        Promedio de sentence-BLEU por oración vs. el resto.
        Si hay <2 oraciones válidas, devuelve 0.0
        """
        sent_tok = self._sent_tokens()
        # quitamos oraciones vacías tras el preprocesado
        sent_tok = [s for s in sent_tok if len(s) > 0]
        if len(sent_tok) < 2:
            return 0.0

        # pesos uniformes para BLEU-n
        weights = tuple([1.0 / self.n] * self.n)

        scores = []
        for i, hyp in enumerate(sent_tok):
            # referencias = todas las otras oraciones no vacías
            refs = [ref for j, ref in enumerate(sent_tok) if j != i and len(ref) > 0]
            if not refs:
                continue
            # BLEU por oración con smoothing
            score = sentence_bleu(refs, hyp, weights=weights, smoothing_function=self._smooth)
            scores.append(score)

        return sum(scores) / len(scores) if scores else 0.0

    def score_text(self):
        # Conveniencia: devolver solo el Self-BLEU
        return self.self_bleu()


import math
from typing import List, Tuple, Optional
import spacy

class MTLD(ExplanationEvaluator):
    """
    MTLD (McCarthy & Jarvis, 2010)
    - Calcula MTLD en dirección forward, backward y su promedio (bidireccional).
    - Preprocesa con spaCy: lowercase, stopwords, filtros alfabéticos, etc.
    """
    def __init__(
        self,
        text: str,
        threshold: float = 0.72,
        min_factor_len: int = 10,
        use_lemmas: bool = False,
        nlp: Optional["spacy.language.Language"] = None,
    ):
        super().__init__(text)
        if threshold <= 0 or threshold >= 1:
            raise ValueError("threshold debe estar entre 0 y 1")
        if min_factor_len < 1:
            raise ValueError("min_factor_len debe ser >= 1")
        self.threshold = threshold
        self.min_factor_len = min_factor_len
        self.use_lemmas = use_lemmas

        # Carga spaCy (permite inyectar un `nlp` externo para evitar recargas)
        if nlp is None:
            disable = ["parser", "ner"]
            # habilita lematizador si se pide
            if not use_lemmas:
                disable.append("lemmatizer")
            self.nlp = spacy.load("en_core_web_sm", disable=disable)
            if use_lemmas:
                # asegura el pipeline del lemmatizer si el modelo lo requiere
                try:
                    self.nlp.enable_pipe("lemmatizer")
                except Exception:
                    pass
        else:
            self.nlp = nlp

        self.doc = self.nlp(self.text)

    def _tokens(self, text=None) -> List[str]:
        doc = self.doc if text is None else (text if hasattr(text, "ents") else self.nlp(text))
        out = []
        for tok in doc:
            if tok.is_space:
                continue
            if self.only_alpha and not tok.is_alpha:
                continue
            t = tok.lemma_.lower() if self.use_lemmas else tok.lower_
            if self.remove_stopwords:
                if self.keep_negations and t in {"no", "not", "n't"}:
                    pass
                elif tok.is_stop:
                    continue
            out.append(t)
        return out

    @staticmethod
    def _mtld_one_direction(tokens: List[str], threshold: float, min_factor_len: int) -> float:
        """
        Devuelve MTLD en una sola dirección: len(tokens) / factor_count
        (incluye factor parcial al final).
        """
        if len(tokens) == 0:
            return 0.0

        types = set()
        tok_count = 0
        factors = 0.0

        for w in tokens:
            tok_count += 1
            types.add(w)
            ttr = len(types) / tok_count

            # cerramos factor solo si TTR <= umbral y se alcanza la longitud mínima
            if ttr <= threshold and tok_count >= min_factor_len:
                factors += 1.0
                types.clear()
                tok_count = 0

        # factor parcial si quedan tokens
        if tok_count > 0:
            ttr = len(types) / tok_count
            # convierte el TTR restante en fracción de factor: (1 - TTR)/(1 - threshold)
            partial = (1.0 - ttr) / (1.0 - threshold)
            # acota por si ruido numérico
            partial = max(0.0, min(1.0, partial))
            factors += partial

        # Evita división por cero si, por alguna razón, no se acumuló ningún factor
        if factors == 0:
            return float(len(tokens))
        return len(tokens) / factors

    def mtld_forward(self) -> float:
        tokens = self._tokens()
        return self._mtld_one_direction(tokens, self.threshold, self.min_factor_len)

    def mtld_backward(self) -> float:
        tokens = list(reversed(self._tokens()))
        return self._mtld_one_direction(tokens, self.threshold, self.min_factor_len)

    def mtld_bidirectional(self) -> float:
        fwd = self.mtld_forward()
        bwd = self.mtld_backward()
        # si una dirección es 0.0 (texto vacío), devuelve la otra
        if fwd == 0.0:
            return bwd
        if bwd == 0.0:
            return fwd
        return (fwd + bwd) / 2.0

    def score_text(self) -> Tuple[float, float, float]:
        """
        Devuelve (forward, backward, bidirectional)
        """
        f = self.mtld_forward()
        b = self.mtld_backward()
        return (f, b, (f + b) / 2.0 if f and b else (f or b))
