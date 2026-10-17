from graph_extract.graphiti_client import document_context


def test_document_context_formats_the_line():
    ctx = document_context("Veeam", "Veeam Backup & Replication", "VBR User Guide")
    assert ctx == (
        "\n\nDOCUMENT CONTEXT: This chunk comes from Veeam documentation for "
        "Veeam Backup & Replication (document set: VBR User Guide). The "
        'documented product is "Veeam Backup & Replication".')


def test_document_context_empty_without_a_product():
    assert document_context("Veeam", "", "VBR User Guide") == ""
    assert document_context("", "", "") == ""
