import paddle
import paddle.nn as nn
from paddlenlp.seq2vec import CNNEncoder


class TextCNNModel(nn.Layer):
    def __init__(self, vocab_size, num_classes, emb_dim=128, padding_idx=0, num_filter=128,
                 ngram_filter_sizes=(1, 2, 3),
                 fc_hidden_size=96):
        super().__init__()
        self.embedder = nn.Embedding(vocab_size, emb_dim, padding_idx=padding_idx)
        self.encoder = CNNEncoder(emb_dim=emb_dim,
                                  num_filter=num_filter,
                                  ngram_filter_sizes=ngram_filter_sizes)
        self.fc = nn.Linear(self.encoder.get_input_dim(), fc_hidden_size)
        self.output_layer = nn.Linear(fc_hidden_size, num_classes)

    def forward(self, text):
        embedded_text = self.embedder(text)  # [batch_size,seq_len,embedding_dim]
        encoder_out = paddle.tanh(self.encoder(embedded_text))  # [batch_size,len(ngram_filter_sizes) *num_filter]
        fc_out = paddle.tanh(self.fc(encoder_out))  # [batch_size,fc_hidden_size]
        logits = self.output_layer(fc_out)  # [batch_size,num_classes]
        return logits
