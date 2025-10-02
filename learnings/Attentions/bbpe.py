import re
from collections import defaultdict


# ----------------- GPT-2 风格的 bytes_to_unicode -----------------
def bytes_to_unicode():
    """
    GPT-2 中用于把字节（0-255）映射成可见的 Unicode 字符。
    避免直接用不可见/控制字符。
    """
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(c) for c in cs]
    return dict(zip(bs, cs))


# ----------------- Byte-level BPE -----------------
class BBPE:
    def __init__(self, vocab_size=100):
        self.vocab_size = vocab_size
        self.vocab = {}  # word -> frequency
        self.merges = []  # list of merges
        self.bpe_ranks = {}  # pair -> rank
        self.token2id = {}  # token(str) -> id
        self.id2token = {}  # id -> token(str)
        self.next_id = 0  # 下一个分配的id
        self.byte_encoder = bytes_to_unicode()  # 字节到可见符号的映射

    # ---------- 构建初始词表 ----------
    def build_vocab(self, corpus):
        vocab = defaultdict(int)
        for line in corpus:
            words = line.strip().split()
            for word in words:
                print(f"word:{word}")
                byte_seq = list(word.encode("utf-8"))  # 直接字节，不加 </w>
                print(f"byte_seq:{byte_seq}")
                vocab[tuple(byte_seq)] += 1
        self.vocab = dict(vocab)
        print(f"初始vocab大小={len(self.vocab)}")
        print(f"初始vocab {self.vocab}")

        # 初始化 token 映射：0-255 字节
        print(f"self.byte_encoder:{self.byte_encoder}")
        for b in range(256):
            s = self.byte_encoder.get(b, chr(b))
            self.token2id[s] = self.next_id
            self.id2token[self.next_id] = s
            self.next_id += 1

    # ---------- 统计 pair ----------
    def get_stats(self):
        pairs = defaultdict(int)
        for word, freq in self.vocab.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += freq
        return pairs

    # ---------- 合并 ----------
    def merge_vocab(self, pair):
        a, b = pair
        token_a = self._id_to_token(a)
        token_b = self._id_to_token(b)
        new_token = token_a + token_b

        if new_token not in self.token2id:
            self.token2id[new_token] = self.next_id
            self.id2token[self.next_id] = new_token
            self.next_id += 1

        new_id = self.token2id[new_token]

        new_vocab = {}
        for word, freq in self.vocab.items():
            print(f"word, freq:{word, freq}")
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                    new_word.append(new_id)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_vocab[tuple(new_word)] = freq
        print(f"old dself.vocab{self.vocab}")
        self.vocab = new_vocab
        print(f"new dself.vocab{self.vocab}")
        return (token_a, token_b)

    # ---------- 训练 ----------
    def train(self, save_merges="merges.txt", save_vocab="vocab.txt"):
        assert self.byte_encoder == self.id2token

        alphabet = set(ch for word in self.vocab for ch in word)
        num_merges = self.vocab_size - len(alphabet)
        print(f"初始alphabet大小={len(alphabet)}，目标vocab_size={self.vocab_size}，合并次数≈{num_merges}")
        for i in range(num_merges):
            pairs = self.get_stats()
            print(f"pairs:{pairs}")
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            print(f"best:{best}")
            merged = self.merge_vocab(best)
            self.merges.append(merged)

        self.bpe_ranks = dict(zip(self.merges, range(len(self.merges))))
        print("训练完成 ✅")

        # 保存 merges
        with open(save_merges, "w", encoding="utf-8") as f:
            for a, b in self.merges:
                f.write(f"{a} {b}\n")

        # 保存 vocab
        with open(save_vocab, "w", encoding="utf-8") as f:
            for token, idx in self.token2id.items():
                f.write(f"{token} {idx}\n")

        print(f"✅ merges 保存到 {save_merges}, vocab 保存到 {save_vocab}")

    # ---------- 编码 ----------
    def encode_word(self, word):
        word = list(word.encode("utf-8"))  # 不加 </w>
        word = [self.token2id[self._id_to_token(b)] for b in word]

        pairs = self.get_pairs(word)
        while pairs:
            bigram = min(pairs, key=lambda p: self.bpe_ranks.get(
                (self._id_to_token(p[0]), self._id_to_token(p[1])), float("inf")))
            if (self._id_to_token(bigram[0]), self._id_to_token(bigram[1])) not in self.bpe_ranks:
                break
            new_word = []
            i = 0
            new_token = self._id_to_token(bigram[0]) + self._id_to_token(bigram[1])
            new_id = self.token2id[new_token]
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == bigram:
                    new_word.append(new_id)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
            if len(word) == 1:
                break
            pairs = self.get_pairs(word)
        return word

    def get_pairs(self, word):
        return set((word[i], word[i + 1]) for i in range(len(word) - 1))

    # ---------- 解码 ----------
    def decode_word(self, tokens):
        text = "".join([self.id2token[t] for t in tokens])
        # 替换 bytes_to_unicode 回字节
        byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        byte_seq = [byte_decoder.get(ch, ord(ch)) for ch in text]
        return bytes(byte_seq).decode("utf-8", errors="ignore")

    # ---------- 工具 ----------
    def _id_to_token(self, idx):
        if idx in self.id2token:
            return self.id2token[idx]
        else:
            return chr(idx)


# ================== 示例 ==================
if __name__ == "__main__":
    corpus = [
        "deep learning is the future of ai",
        "see my eyes first",
        "see my dogs",
        "you are the best",
        "you are the fast",
        "machine learning can be applied to natural language processing",
        "深度学习是人工智能的未来",
        "机器学习可以应用于自然语言处理",
        "人工智能改变世界",
        "学习深度神经网络在图像识别中表现优秀"
    ]

    bbpe = BBPE(vocab_size=100)
    bbpe.build_vocab(corpus)
    bbpe.train("merges.txt", "vocab.txt")

    print("\n=== 单词测试 ===")
    for w in ["lowest", "人工智能", "深度学习"]:
        tokens = bbpe.encode_word(w)
        print(f"{w} -> {tokens} -> {bbpe.decode_word(tokens)}")

    print("\n=== 句子测试 ===")
    sentence = "lowest人工智能深度学习"
    tokens_list = [bbpe.encode_word(w) for w in sentence.split()]
    print(tokens_list)
    print("解码:", " ".join(bbpe.decode_word(toks) for toks in tokens_list))
