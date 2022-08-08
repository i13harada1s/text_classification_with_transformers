from absl.testing import absltest
from transformers import AutoTokenizer

from dataset import load_csv_dataset


class DatasetTest(absltest.TestCase):
    data_path = "examples/sample.csv"
    name_or_path = "cl-tohoku/bert-base-japanese-v2"
    tokenizer = AutoTokenizer.from_pretrained(name_or_path)

    def test_load_csv_dataset(self):
        dataset = load_csv_dataset(self.data_path, self.tokenizer)
        self.assertEqual(2, len(dataset))


if __name__ == "__main__":
    absltest.main()
