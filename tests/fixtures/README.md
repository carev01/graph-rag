# Test Fixtures

These files are live API responses captured from the DocExtractor cluster on 2026-07-12 for use as test inputs in unit tests.

## Source Data

- **Source ID**: `21632f3b-5a4c-4c93-9f00-6701d0e9f677` (AWS Backup Developer Guide)
- **Captured from**: `https://docextractor.k3s.home.lan`
- **API version**: Current production

## Files

- `aws_delta.ndjson` - Delta stream bootstrap payload (148 lines: 146 content records + 2 control records)
- `aws_toc.json` - Table of contents tree (41 KB)
- `vendors.json` - Vendor catalog (7.1 KB, 23 vendors)
- `products.json` - Product catalog (20 KB, ~80 products)
- `sources.json` - Source catalog (106 KB, full corpus)

## Important

The `content_markdown` fields in the delta stream contain real, unmodified text excerpts from the AWS Backup Developer Guide. These are used as test input only and are not included in any deliverable.

All fixtures are read-only test data and should not be modified.
