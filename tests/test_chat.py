import pytest

from app.chat import ChatService, REFUSAL
from app.models import ContextChunk, FileRecord, FileState, SourceLocation
from app.retrieval import SearchHit


class MemoryRepository:
    def __init__(self):
        self.messages = []

    def conversation_history(self, _conversation_id):
        return []

    def save_message(self, conversation_id, role, content, citations=None):
        self.messages.append((conversation_id, role, content, citations or []))


class FakeRetriever:
    def __init__(self, hits):
        self.hits = hits
        self.repository = MemoryRepository()

    def search(self, _question, _limit):
        return self.hits


class FakeLLM:
    def __init__(self, response):
        self.response = response

    async def answer(self, question, context, history):
        return self.response


def hit() -> SearchHit:
    file = FileRecord(1, "facts.pdf", "a" * 64, 1, 1, "pdf", FileState.READY)
    chunk = ContextChunk(
        "chunk", "a" * 64, 0, "The launch date is 12 May.", 7, "text", (), SourceLocation(page=4)
    )
    return SearchHit(file, chunk, 0.03, 0.8, True)


@pytest.mark.asyncio
async def test_grounded_answer_keeps_only_valid_citations(settings) -> None:
    retriever = FakeRetriever([hit()])
    service = ChatService(settings, retriever, FakeLLM({"answer": "It launches 12 May.", "citations": ["S1", "S99"]}))
    result = await service.ask("When does it launch?")
    assert result.answer == "It launches 12 May."
    assert [citation.id for citation in result.citations] == ["S1"]
    assert result.citations[0].page == 4


@pytest.mark.asyncio
async def test_missing_evidence_refuses_without_calling_model(settings) -> None:
    retriever = FakeRetriever([])
    service = ChatService(settings, retriever, FakeLLM({"answer": "should not run", "citations": []}))
    result = await service.ask("What is outside the files?")
    assert result.refused is True
    assert result.answer == REFUSAL


@pytest.mark.asyncio
async def test_uncited_model_answer_is_rejected(settings) -> None:
    service = ChatService(settings, FakeRetriever([hit()]), FakeLLM({"answer": "Unsupported", "citations": []}))
    result = await service.ask("Tell me")
    assert result.refused is True
    assert result.citations == []

