class SimpleTokenizer:
    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_size = 0
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

    def build_vocab(self, texts: list[str]) -> None:
        """
        Builds the vocabulary in place.
        """
        self.word_to_id[self.pad_token] = 0
        self.word_to_id[self.unk_token]=1
        self.word_to_id[self.bos_token]=2
        self.word_to_id[self.eos_token]=3
        all_words=set()
        for text in texts:
            all_words.update(text.lower().split())
        for i,word in enumerate(sorted(all_words),start=4):
            self.word_to_id[word]=i
        # Create the reverse mapping
        self.id_to_word={v:k for k,v in self.word_to_id.items()}
        self.vocab_size=len(self.word_to_id)


    def encode(self, text: str) -> list[int]:
        """
        Returns token IDs for the input text.
        """
        tokens=text.lower().split()
        ids=[self.word_to_id.get(token,self.word_to_id[self.unk_token]) for token in tokens]
        return ids


    def decode(self, ids: list[int]) -> str:
        """
        Returns the decoded, space-separated text.
        """
        tokens=[self.id_to_word.get(id,self.unk_token) for id in ids]
        return ' '.join(tokens)