"""Regression tests with real storage and deterministic embeddings (no downloads)."""
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from typer.testing import CliRunner
from watchdog.events import DirDeletedEvent, DirMovedEvent, FileMovedEvent

from rag_kit.cli import main as cli
from rag_kit.config import Config
from rag_kit.ingest import ingest_file
from rag_kit.store import VectorStore, _make_schema


class Engine:
    def embed_texts(self, texts):
        return np.tile(np.eye(8, dtype=np.float32)[0], (len(texts), 1))

    def embed_query(self, text):
        return self.embed_texts([text])[0]


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = Config.from_dict({'db_path': str(tmp_path / 'db'), 'vlm_enabled': False,
                            'watch_folder': str(tmp_path)})
    monkeypatch.setattr(cli, '_ensure_config', lambda: cfg)
    monkeypatch.setattr(cli, 'get_config', lambda: cfg)
    monkeypatch.setattr(cli, 'create_engine_from_config', lambda _: Engine())
    return cfg


def test_explicit_new_config_never_falls_back_to_home(tmp_path, monkeypatch):
    from rag_kit import config
    home = tmp_path / 'home'
    home.mkdir()
    (home / '.rag-kit.yaml').write_text('watch_folder: old-folder\n')
    explicit = tmp_path / 'new' / 'config.yml'
    monkeypatch.setenv('RAG_KIT_CONFIG', str(explicit))
    monkeypatch.setattr(config, '_HOME', home)
    assert config.get_config_path() == explicit
    monkeypatch.setattr(cli, 'get_config_path', lambda: explicit)
    cli.config_set('chunk_size', '256')
    assert explicit.exists()
    assert (home / '.rag-kit.yaml').read_text() == 'watch_folder: old-folder\n'


def import_doc(path):
    cli.ingest(str(path), ocr=False, vlm=False, json_output=False)


def rows(cfg):
    return VectorStore(cfg.db_path)._get_table().to_arrow().to_pylist()


def test_reimport_replaces_only_its_source(setup, tmp_path):
    p = tmp_path / "author's note.txt"
    other = tmp_path / 'other.txt'
    p.write_text('Old obsolete instruction', encoding='utf-8')
    other.write_text('Keep this document', encoding='utf-8')
    import_doc(p)
    import_doc(other)
    p.write_text('New replacement instruction', encoding='utf-8')
    import_doc(p)
    import_doc(p)
    assert {r['text'] for r in rows(setup)} == {'Keep this document', 'New replacement instruction'}
    assert len(rows(setup)) == 2


def test_relative_import_can_be_deleted(setup, tmp_path, capsys):
    p = Path('note.txt')
    p.write_text('Document content', encoding='utf-8')
    import_doc(p)
    assert rows(setup)[0]['source'] == str(p.resolve())
    capsys.readouterr()
    cli.delete(str(p), json_output=True)
    assert json.loads(capsys.readouterr().out)['removed_chunks'] == 1
    assert VectorStore(setup.db_path).count_rows() == 0


def test_move_removes_old_source(setup, tmp_path):
    p = tmp_path / 'before.txt'
    p.write_text('Watched document', encoding='utf-8')
    handler = cli._WatchHandler(setup)
    handler._ingest_one(str(p))
    q = tmp_path / 'after.txt'
    p.rename(q)
    handler.handler.dispatch(FileMovedEvent(str(p), str(q)))
    handler.flush_and_stop()
    assert VectorStore(setup.db_path).list_sources() == [str(q)]


def test_directory_move_and_delete_remove_children(setup, tmp_path):
    before = tmp_path / 'before'
    before.mkdir()
    p = before / 'note.txt'
    p.write_text('Document content', encoding='utf-8')
    handler = cli._WatchHandler(setup)
    handler._ingest_one(str(p))
    after = tmp_path / 'after'
    before.rename(after)
    handler.handler.dispatch(DirMovedEvent(str(before), str(after)))
    handler.flush_and_stop()
    assert VectorStore(setup.db_path).list_sources() == [str(after / p.name)]
    another = cli._WatchHandler(setup)
    another.handler.dispatch(DirDeletedEvent(str(after)))
    another.flush_and_stop()
    assert VectorStore(setup.db_path).count_rows() == 0


@pytest.mark.parametrize('watch', [False, True])
def test_empty_edit_removes_old_content(setup, tmp_path, watch):
    p = tmp_path / 'note.txt'
    p.write_text('Text removed by the user', encoding='utf-8')
    ingest = cli._WatchHandler(setup)._ingest_one if watch else import_doc
    ingest(str(p))
    p.write_text('', encoding='utf-8')
    ingest(str(p))
    assert VectorStore(setup.db_path).count_rows() == 0


def test_startup_reconciles_offline_changes_and_keeps_other_roots(setup, tmp_path):
    watched = tmp_path / 'watched'
    watched.mkdir()
    deleted = watched / 'deleted.txt'
    changed = watched / 'changed.txt'
    other = tmp_path / 'other.txt'
    for p in (deleted, changed, other):
        p.write_text('Old ' + p.name, encoding='utf-8')
        import_doc(p)
    deleted.unlink()
    changed.write_text('New offline edit', encoding='utf-8')
    created = watched / 'created.txt'
    created.write_text('Created offline', encoding='utf-8')
    handler = cli._WatchHandler(setup)
    handler.reconcile(watched)
    handler.flush_and_stop()
    assert {r['text'] for r in rows(setup)} == {'New offline edit', 'Created offline', 'Old other.txt'}


def test_unsupported_extension_is_removed_on_reconcile(setup, tmp_path):
    p = tmp_path / 'note.md'
    p.write_text('Indexed before configuration changed', encoding='utf-8')
    import_doc(p)
    setup.supported_extensions = ['.txt']
    handler = cli._WatchHandler(setup)
    handler.reconcile(tmp_path)
    handler.flush_and_stop()
    assert VectorStore(setup.db_path).count_rows() == 0


def test_extraction_and_embedding_failures_keep_previous_document(setup, tmp_path):
    p = tmp_path / 'note.txt'
    p.write_text('Previous document', encoding='utf-8')
    import_doc(p)
    handler = cli._WatchHandler(setup)
    with patch.object(cli, 'ingest_file', side_effect=ValueError('Incomplete file')):
        handler._ingest_one(str(p))
    p.write_text('New document', encoding='utf-8')
    with patch.object(Engine, 'embed_texts', side_effect=RuntimeError('Embedding failed')):
        handler._ingest_one(str(p))
    assert [r['text'] for r in rows(setup)] == ['Previous document']


def test_wrong_embedding_dimension_preserves_rows(setup, tmp_path):
    p = tmp_path / 'note.txt'
    p.write_text('Previous document', encoding='utf-8')
    import_doc(p)
    store = VectorStore(setup.db_path)
    with pytest.raises(ValueError, match='dimension'):
        store.replace_source(str(p), ingest_file(p), np.ones((1, 3), dtype=np.float32))
    assert [r['text'] for r in rows(setup)] == ['Previous document']


def test_open_reader_sees_changes_from_another_store(tmp_path):
    db_path = str(tmp_path / 'db')
    reader = VectorStore(db_path, dim=8)
    reader._ensure_table()
    assert reader.count_rows() == 0
    writer = VectorStore(db_path)
    writer.replace_source('a.txt', [{'id': 'one', 'text': 'New content', 'source': 'a.txt'}], Engine().embed_texts(['x']))
    assert reader.count_rows() == 1
    assert reader.search(Engine().embed_query('x'))[0]['text'] == 'New content'


def test_visual_caption_shares_source_and_page(setup, tmp_path):
    import fitz
    p = tmp_path / 'visual.pdf'
    with fitz.open() as doc:
        page = doc.new_page()
        page.insert_text((72, 72), 'A sufficiently long textual introduction to a chart.')
        doc.save(p)
    caption = {'text': 'A visual chart', 'page': 0, 'vlm_generated': True, 'source_type': 'image'}
    with patch('rag_kit.ingest.pipeline._run_vlm_captioning', return_value=[caption]):
        chunks = ingest_file(p, use_ocr=False, use_vlm=True)
    assert {c['source'] for c in chunks} == {str(p)}
    assert {c['page'] for c in chunks} == {1}
    with patch.object(cli, 'ingest_file', return_value=chunks):
        import_doc(p)
    assert rows(setup)[-1]['vlm_generated']
    cli._WatchHandler(setup)._delete_one(str(p))
    assert VectorStore(setup.db_path).count_rows() == 0


def test_page_vlm_fallback_is_marked_generated(tmp_path):
    from rag_kit.ingest import pipeline
    p = tmp_path / 'image.pdf'
    p.touch()
    with patch.object(pipeline, '_extract_pages', return_value=[pipeline._Page('Caption', 1, extraction_method='vlm')]):
        chunks = ingest_file(p)
    assert chunks[0]['vlm_generated'] is True
    assert chunks[0]['source_type'] == 'image'


@pytest.mark.parametrize('content', ['Document to ingest', ''])
def test_json_ingest_is_single_parseable_object(setup, tmp_path, content):
    p = tmp_path / 'note.txt'
    p.write_text(content, encoding='utf-8')
    result = CliRunner().invoke(cli.app, ['ingest', str(p), '--no-ocr', '--no-vlm', '--json'])
    assert result.exit_code == 0, result.output
    output = json.loads(result.stdout)
    assert output['status'] == 'ok'
    assert output['chunks_stored'] == bool(content)


@pytest.mark.skipif(__import__('os').name != 'nt', reason='Windows ANSI pipe regression')
def test_windows_cli_emits_utf8_through_ansi_environment(tmp_path):
    import os
    import subprocess
    import sys
    config = tmp_path / 'config.yml'
    result = subprocess.run([sys.executable, '-m', 'rag_kit.cli.main', 'config', 'set', 'watch_folder', '中文资料'],
                            env={**os.environ, 'PYTHONIOENCODING': 'cp1252', 'PYTHONUTF8': '0',
                                 'RAG_KIT_CONFIG': str(config)}, capture_output=True,
                            text=True, encoding='utf-8', timeout=15,
                            cwd=Path(__file__).resolve().parents[1])
    assert result.returncode == 0, result.stderr
    assert '中文资料' in result.stdout


def test_failed_folder_file_does_not_replace_its_previous_rows(setup, tmp_path):
    good = tmp_path / 'good.txt'
    bad = tmp_path / 'bad.xlsx'
    good.write_text('Working document', encoding='utf-8')
    bad.write_text('Not a workbook', encoding='utf-8')
    store = VectorStore(setup.db_path)
    store.add_chunks([{'id': 'saved', 'text': 'Previous workbook', 'source': str(bad)}], Engine().embed_texts(['x']))
    result = CliRunner().invoke(cli.app, ['ingest', str(tmp_path), '--no-ocr', '--no-vlm', '--json'])
    assert result.exit_code == 1
    assert json.loads(result.stdout)['errors'][0]['file'] == str(bad)
    assert {r['text'] for r in rows(setup)} == {'Working document', 'Previous workbook'}


def test_zero_alpha_passes_through_cli(setup, tmp_path):
    setup.search_alpha = 0.0
    p = tmp_path / 'note.txt'
    p.write_text('Searchable text', encoding='utf-8')
    import_doc(p)
    with patch.object(VectorStore, 'search_hybrid', return_value=[]) as mock:
        cli.query_cmd('anything', top_k=5, source='', json_output=True)
    assert mock.call_args.kwargs['alpha'] == 0.0


def test_source_filter_resolves_relative_path_and_keeps_keyword_weight(setup, tmp_path):
    setup.search_alpha = 0.0
    p = Path('note.txt')
    p.write_text('Keyword alpha', encoding='utf-8')
    import_doc(p)
    with patch.object(VectorStore, 'search_hybrid', return_value=[]) as mock:
        cli.query_cmd('alpha', top_k=5, source=str(p), json_output=True)
    assert mock.call_args.kwargs['alpha'] == 0.0
    assert str(p.resolve()) in mock.call_args.kwargs['filter_sql']


def test_move_out_of_watched_root_does_not_ingest_destination(setup, tmp_path):
    watched = tmp_path / 'watched'
    watched.mkdir()
    setup.watch_folder = str(watched)
    p = watched / 'note.txt'
    p.write_text('Content', encoding='utf-8')
    handler = cli._WatchHandler(setup)
    handler._ingest_one(str(p))
    q = tmp_path / 'outside.txt'
    p.rename(q)
    handler.handler.dispatch(FileMovedEvent(str(p), str(q)))
    handler.flush_and_stop()
    assert VectorStore(setup.db_path).count_rows() == 0


def test_failed_ocr_of_scanned_pdf_is_an_error(tmp_path):
    import fitz
    import io
    from PIL import Image
    p = tmp_path / 'scanned.pdf'
    png = io.BytesIO()
    Image.new('RGB', (40, 40), 'white').save(png, format='PNG')
    with fitz.open() as doc:
        page = doc.new_page()
        page.insert_image(page.rect, stream=png.getvalue())
        doc.save(p)
    with patch('rag_kit.ingest.pipeline.ocr_image', side_effect=ImportError('No OCR engine')):
        with pytest.raises(RuntimeError, match='PDF page 1'):
            ingest_file(p, use_ocr=True, use_vlm=False)


def test_native_observer_handles_create_edit_move_and_delete(setup, tmp_path):
    import time
    from watchdog.observers import Observer
    watched = tmp_path / 'watched'
    watched.mkdir()
    setup.watch_folder = str(watched)
    handler = cli._WatchHandler(setup, debounce_seconds=0.1)
    observer = Observer()
    observer.schedule(handler.handler, str(watched), recursive=True)
    observer.start()
    handler.reconcile(watched)
    handler.start_worker()
    store = VectorStore(setup.db_path)

    def wait_for(predicate):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.1)
        pytest.fail('Watcher did not process the filesystem change')

    try:
        p = watched / 'note.txt'
        p.write_text('Initial content', encoding='utf-8')
        wait_for(lambda: store.count_rows() == 1)
        p.write_text('Updated content', encoding='utf-8')
        wait_for(lambda: rows(setup)[0]['text'] == 'Updated content')
        q = watched / 'renamed.txt'
        p.rename(q)
        wait_for(lambda: VectorStore(setup.db_path).list_sources() == [str(q)])
        q.unlink()
        wait_for(lambda: VectorStore(setup.db_path).count_rows() == 0)
    finally:
        observer.stop()
        observer.join()
        handler.flush_and_stop()


def test_hybrid_weights_change_ranking_and_index_refreshes(tmp_path):
    store = VectorStore(str(tmp_path / 'db'))
    chunks = [{'id': str(i), 'text': t, 'source': 'a.txt'} for i, t in enumerate(['alpha keyword', 'beta other', 'gamma unrelated'])]
    vec = np.eye(3, dtype=np.float32)
    store.add_chunks(chunks, vec)
    assert store.search_hybrid('alpha', vec[2], alpha=0.01)[0]['id'] == '0'
    assert store.search_hybrid('alpha', vec[2], alpha=0.99)[0]['id'] == '2'
    assert store.search_hybrid('alpha', vec[2], alpha=0)[0]['id'] == '0'
    replacement = [{'id': 'new', 'text': 'delta keyword', 'source': 'a.txt'}]
    store.replace_source('a.txt', replacement, vec[:1])
    assert store.search_hybrid('delta', vec[0], alpha=0)[0]['id'] == 'new'
    assert store.search_hybrid('alpha', vec[0], alpha=0) == []


def test_legacy_metadata_migration_keeps_existing_rows(tmp_path):
    import lancedb
    schema = _make_schema(8)
    schema = schema.remove(schema.get_field_index('vlm_generated'))
    schema = schema.remove(schema.get_field_index('source_type'))
    db_path = str(tmp_path / 'db')
    table = lancedb.connect(db_path).create_table('documents', schema=schema)
    table.add([{'id': 'old', 'text': 'Existing document', 'vector': [1.] * 8, 'source': 'old.txt', 'page': 1, 'chunk_idx': 0}])
    store = VectorStore(db_path)
    assert store.count_rows() == 1  # Also exercises an already-cached legacy table.
    store.replace_source('new.txt', [{'id': 'new', 'text': 'New document', 'source': 'new.txt'}], Engine().embed_texts(['x']))
    assert set(store.list_sources()) == {'old.txt', 'new.txt'}
    old = next(r for r in store._get_table().to_arrow().to_pylist() if r['id'] == 'old')
    assert old['vlm_generated'] is False
    assert old['source_type'] == 'text'


def test_cached_model_uses_complete_namespaced_snapshot(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from rag_kit.embed import EmbeddingEngine, _resolve_cached_model
    name = 'paraphrase-multilingual-MiniLM-L12-v2'
    hub = tmp_path / f'models--sentence-transformers--{name}'
    cache = hub / 'snapshots' / 'pinned'
    cache.mkdir(parents=True)
    (cache / 'config.json').write_text('{}')
    assert _resolve_cached_model(name, str(tmp_path)) is None
    for file in ('modules.json', 'tokenizer.json', 'model.safetensors'):
        (cache / file).write_text('{}')
    from unittest.mock import Mock
    constructor = Mock()
    monkeypatch.setitem(sys.modules, 'sentence_transformers', SimpleNamespace(SentenceTransformer=constructor))
    engine = EmbeddingEngine(model_name=name, model_dir=str(tmp_path))
    engine._load_model()
    assert constructor.call_args.args == (str(cache.resolve()),)
    assert constructor.call_args.kwargs['local_files_only'] is True


def test_poll_watcher_replaces_and_clears_empty_file(setup, tmp_path, monkeypatch):
    from rag_kit import watcher, config, embed
    monkeypatch.setattr(config, 'get_config', lambda: setup)
    monkeypatch.setattr(embed, 'create_engine_from_config', lambda _: Engine())
    monkeypatch.setattr(watcher, '_state_path', lambda: tmp_path / 'state.json')
    p = tmp_path / 'note.txt'
    p.write_text('Old content', encoding='utf-8')
    assert not watcher.scan_and_ingest(once=True)['errors']
    p.write_text('New content', encoding='utf-8')
    assert not watcher.scan_and_ingest(once=True)['errors']
    assert [r['text'] for r in rows(setup)] == ['New content']
    p.write_text('', encoding='utf-8')
    assert not watcher.scan_and_ingest(once=True)['errors']
    assert VectorStore(setup.db_path).count_rows() == 0
