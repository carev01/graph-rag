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


def test_document_context_omits_empty_vendor_cleanly():
    # No double space from an empty vendor ("from  documentation").
    ctx = document_context("", "AWS Backup", "AWS Backup Developer Guide")
    assert "from  documentation" not in ctx
    assert "from documentation for AWS Backup" in ctx
    assert "(document set: AWS Backup Developer Guide)" in ctx


def test_document_context_omits_empty_source_cleanly():
    # No empty parens from an empty source ("(document set: )").
    ctx = document_context("AWS", "AWS Backup", "")
    assert "(document set: )" not in ctx
    assert "document set" not in ctx
    assert "from AWS documentation for AWS Backup." in ctx


def test_document_context_omits_both_empty_parts_cleanly():
    ctx = document_context("", "AWS Backup", "")
    assert "from  documentation" not in ctx
    assert "(document set: )" not in ctx
    assert ctx == (
        "\n\nDOCUMENT CONTEXT: This chunk comes from documentation for "
        'AWS Backup. The documented product is "AWS Backup".')
