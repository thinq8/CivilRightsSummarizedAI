from pathlib import Path

from clearinghouse.clients import MockClearinghouseClient
from clearinghouse.ingest import IngestionPipeline
from clearinghouse.processing import HeuristicSummarizer
from clearinghouse.storage import create_session_factory, init_db


def test_mock_ingestion(tmp_path):
    db_path = tmp_path / "test.db"
    session_factory, engine = create_session_factory(f"sqlite:///{db_path}")
    init_db(engine)

    fixture = Path("data/fixtures/mock_dataset.json")
    client = MockClearinghouseClient(fixture)
    pipeline = IngestionPipeline(client, session_factory, HeuristicSummarizer(max_sentences=2))

    stats = pipeline.run()

    assert stats.cases == 2
    assert stats.dockets == 2
    assert stats.documents == 4
