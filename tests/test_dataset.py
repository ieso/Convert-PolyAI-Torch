import pytest
from sentencepiece import SentencePieceProcessor


from src.config import ConveRTTrainConfig
from src.dataset import load_instances_from_reddit_json, load_instances_from_ieso_json, max_context_length


@pytest.fixture
def config():
    return ConveRTTrainConfig()


@pytest.fixture
def tokenizer() -> SentencePieceProcessor:
    tokenizer = SentencePieceProcessor()
    tokenizer.Load(config.sp_model_path)
    return tokenizer


def test_load_instances_from_reddit_json():
    instances = load_instances_from_reddit_json("../data/sample-dataset.json")
    assert len(instances) == 1000
    assert max([len(i.context) for i in instances]) == max_context_length


def test_load_instances_from_ieso_json():
    instances = load_instances_from_ieso_json("../data/transcripts/02aa9cb4-3875-434e-aea0-7d036ab30dc2.json")
    assert len(instances) == 78
    assert max([len(i.context) for i in instances]) == max_context_length


if __name__ == "__main__":
    pytest.main()
