from absl.testing import absltest
import torch
from transformers import AutoModel

from model import TextClassificationModel


class ModelTest(absltest.TestCase):
    def test_tohoku_bert_v2(self):
        batch_size = 5
        seq_length = 32
        n_label = 10
        name_or_path = "cl-tohoku/bert-base-japanese-v2"
        pretrained_model = AutoModel.from_pretrained(name_or_path)
        hidden_dim = pretrained_model.pooler.dense.out_features
        model = TextClassificationModel(pretrained_model, hidden_dim, n_label)
        input = torch.randint(0, 100, (batch_size, seq_length))  # B x S
        output = model(input)  # get sentence embeddings
        self.assertEqual((batch_size, n_label), tuple(output.size()))


if __name__ == "__main__":
    absltest.main()
