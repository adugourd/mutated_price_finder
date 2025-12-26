"""
Unit tests for EVE Ref contract data fetching.
"""

import pytest
import responses
import tarfile
import io
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime

import pandas as pd

from src.data.everef import (
    download_contract_archive,
    extract_csv_from_archive,
    get_contracts_with_items,
    get_dogma_attributes,
    get_cache_dir,
    EVEREF_CONTRACTS_URL,
    CACHE_DURATION,
)


@pytest.fixture
def sample_archive(tmp_path):
    """Create a sample tar.bz2 archive for testing."""
    archive_path = tmp_path / "test_archive.tar.bz2"

    # Create mock CSV content
    contracts_csv = b"contract_id,price,type,region_id\n1,1000000,item_exchange,10000002\n2,2000000,item_exchange,10000002"
    dynamic_items_csv = b"contract_id,item_id,type_id,source_type_id\n1,100,47749,12058\n2,101,47749,13937"
    dogma_attrs_csv = b"item_id,attribute_id,value\n100,64,1.15\n100,204,0.92\n101,64,1.20\n101,204,0.88"

    with tarfile.open(archive_path, 'w:bz2') as tar:
        # Add contracts.csv
        info = tarfile.TarInfo(name='contracts.csv')
        info.size = len(contracts_csv)
        tar.addfile(info, io.BytesIO(contracts_csv))

        # Add contract_dynamic_items.csv
        info = tarfile.TarInfo(name='contract_dynamic_items.csv')
        info.size = len(dynamic_items_csv)
        tar.addfile(info, io.BytesIO(dynamic_items_csv))

        # Add contract_dynamic_items_dogma_attributes.csv
        info = tarfile.TarInfo(name='contract_dynamic_items_dogma_attributes.csv')
        info.size = len(dogma_attrs_csv)
        tar.addfile(info, io.BytesIO(dogma_attrs_csv))

    return archive_path


class TestGetCacheDir:
    """Tests for cache directory management."""

    def test_returns_path_object(self):
        """Returns a Path object."""
        cache_dir = get_cache_dir()
        assert isinstance(cache_dir, Path)

    def test_cache_dir_is_absolute(self):
        """Cache directory is an absolute path."""
        cache_dir = get_cache_dir()
        assert cache_dir.is_absolute()

    def test_cache_dir_exists_or_created(self):
        """Cache directory exists (mkdir is called with exist_ok=True)."""
        cache_dir = get_cache_dir()
        assert cache_dir.exists()


class TestExtractCsvFromArchive:
    """Tests for CSV extraction from archives."""

    def test_extracts_contracts_csv(self, sample_archive):
        """Extracts contracts.csv from archive."""
        df = extract_csv_from_archive(sample_archive, 'contracts.csv')

        assert not df.empty
        assert 'contract_id' in df.columns
        assert 'price' in df.columns
        assert len(df) == 2

    def test_extracts_dynamic_items_csv(self, sample_archive):
        """Extracts contract_dynamic_items.csv from archive."""
        df = extract_csv_from_archive(sample_archive, 'contract_dynamic_items.csv')

        assert not df.empty
        assert 'contract_id' in df.columns
        assert 'item_id' in df.columns
        assert 'source_type_id' in df.columns

    def test_extracts_dogma_attributes_csv(self, sample_archive):
        """Extracts dogma attributes CSV from archive."""
        df = extract_csv_from_archive(sample_archive, 'contract_dynamic_items_dogma_attributes.csv')

        assert not df.empty
        assert 'item_id' in df.columns
        assert 'attribute_id' in df.columns
        assert 'value' in df.columns

    def test_raises_for_missing_file(self, sample_archive):
        """Raises FileNotFoundError for missing CSV file."""
        with pytest.raises(FileNotFoundError) as exc_info:
            extract_csv_from_archive(sample_archive, 'nonexistent.csv')

        assert 'nonexistent.csv' in str(exc_info.value)

    def test_handles_nested_path(self, tmp_path):
        """Handles CSV files in nested paths within archive."""
        archive_path = tmp_path / "nested.tar.bz2"
        csv_content = b"col1,col2\n1,2\n3,4"

        with tarfile.open(archive_path, 'w:bz2') as tar:
            info = tarfile.TarInfo(name='data/subfolder/test.csv')
            info.size = len(csv_content)
            tar.addfile(info, io.BytesIO(csv_content))

        # Should find the file even with nested path
        df = extract_csv_from_archive(archive_path, 'test.csv')
        assert len(df) == 2


class TestGetContractsWithItems:
    """Tests for getting contracts and items together."""

    def test_returns_both_dataframes(self, sample_archive):
        """Returns tuple of (contracts, dynamic_items) DataFrames."""
        contracts, dynamic_items = get_contracts_with_items(sample_archive)

        assert isinstance(contracts, pd.DataFrame)
        assert isinstance(dynamic_items, pd.DataFrame)
        assert 'contract_id' in contracts.columns
        assert 'contract_id' in dynamic_items.columns

    def test_dataframes_have_matching_contracts(self, sample_archive):
        """Contracts can be joined with dynamic_items."""
        contracts, dynamic_items = get_contracts_with_items(sample_archive)

        # All contract_ids in dynamic_items should be in contracts
        contract_ids = set(contracts['contract_id'])
        item_contract_ids = set(dynamic_items['contract_id'])

        assert item_contract_ids.issubset(contract_ids)


class TestGetDogmaAttributes:
    """Tests for getting dogma attributes."""

    def test_returns_attributes_dataframe(self, sample_archive):
        """Returns DataFrame with dogma attributes."""
        df = get_dogma_attributes(sample_archive)

        assert isinstance(df, pd.DataFrame)
        assert 'item_id' in df.columns
        assert 'attribute_id' in df.columns
        assert 'value' in df.columns

    def test_contains_expected_attribute_ids(self, sample_archive):
        """Contains expected EVE attribute IDs."""
        df = get_dogma_attributes(sample_archive)

        # Our test data has attribute_id 64 (damage) and 204 (rof)
        assert 64 in df['attribute_id'].values
        assert 204 in df['attribute_id'].values


class TestDownloadContractArchive:
    """Tests for contract archive downloading."""

    @responses.activate
    def test_downloads_archive_when_not_cached(self, tmp_path):
        """Downloads archive when not cached."""
        mock_content = b"fake tar content"

        responses.add(
            responses.GET,
            EVEREF_CONTRACTS_URL,
            body=mock_content,
            status=200,
            stream=True
        )

        result = download_contract_archive(cache_dir=tmp_path)

        assert result.exists()
        assert result.read_bytes() == mock_content
        assert len(responses.calls) == 1

    def test_uses_cached_file_when_valid(self, tmp_path):
        """Uses cached file if recent enough."""
        cache_file = tmp_path / "public-contracts-latest.v2.tar.bz2"
        cache_file.write_bytes(b"cached content")

        # Touch file to make it recent
        cache_file.touch()

        # Should use cache and not make HTTP request
        with responses.RequestsMock() as rsps:
            result = download_contract_archive(cache_dir=tmp_path)

            assert result == cache_file
            assert len(rsps.calls) == 0

    @responses.activate
    def test_redownloads_when_cache_expired(self, tmp_path):
        """Re-downloads archive when cache is expired."""
        import os
        import time

        cache_file = tmp_path / "public-contracts-latest.v2.tar.bz2"
        cache_file.write_bytes(b"old cached content")

        # Set modification time to past (older than CACHE_DURATION)
        old_time = datetime.now().timestamp() - CACHE_DURATION - 100
        os.utime(cache_file, (old_time, old_time))

        new_content = b"new content"
        responses.add(
            responses.GET,
            EVEREF_CONTRACTS_URL,
            body=new_content,
            status=200,
            stream=True
        )

        result = download_contract_archive(cache_dir=tmp_path)

        assert result.read_bytes() == new_content
        assert len(responses.calls) == 1


class TestArchiveIntegration:
    """Integration tests for archive operations."""

    def test_full_workflow_extracts_all_data(self, sample_archive):
        """Full workflow extracts contracts, items, and attributes."""
        contracts, dynamic_items = get_contracts_with_items(sample_archive)
        dogma = get_dogma_attributes(sample_archive)

        # Verify we can join the data
        items_with_attrs = dynamic_items.merge(
            dogma,
            on='item_id',
            how='left'
        )

        assert len(items_with_attrs) > 0
        assert 'attribute_id' in items_with_attrs.columns

    def test_contracts_have_required_columns(self, sample_archive):
        """Contracts DataFrame has all required columns."""
        contracts = extract_csv_from_archive(sample_archive, 'contracts.csv')

        required_cols = ['contract_id', 'price']
        for col in required_cols:
            assert col in contracts.columns, f"Missing column: {col}"

    def test_dogma_values_are_numeric(self, sample_archive):
        """Dogma attribute values are numeric."""
        df = get_dogma_attributes(sample_archive)

        # Should be able to convert to numeric
        assert pd.api.types.is_numeric_dtype(df['value'])
        assert pd.api.types.is_numeric_dtype(df['attribute_id'])
