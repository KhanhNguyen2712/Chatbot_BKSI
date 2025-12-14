"""Pytest configuration."""

import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Create test data directory."""
    return tmp_path_factory.mktemp("data")


@pytest.fixture(scope="session")
def sample_markdown_content():
    """Sample markdown content for testing."""
    return """
# Quy định đăng ký môn học

## 1. Thời gian đăng ký

Sinh viên đăng ký môn học từ ngày 01/08 đến 15/08.

## 2. Quy trình đăng ký

1. Đăng nhập vào hệ thống
2. Chọn môn học
3. Xác nhận đăng ký

## 3. Lưu ý

- Kiểm tra lịch học trước khi đăng ký
- Không đăng ký trùng lịch
"""


@pytest.fixture
def sample_document(sample_markdown_content, tmp_path):
    """Create sample document file."""
    doc_path = tmp_path / "sample.md"
    doc_path.write_text(sample_markdown_content, encoding="utf-8")
    return doc_path
