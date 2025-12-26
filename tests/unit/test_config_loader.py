"""
Unit tests for configuration loading.
"""

import pytest
from pathlib import Path

from src.config.loader import (
    load_attributes,
    load_constants,
    load_all_targets,
    get_target,
    clear_cache,
    RollTarget,
    MutaplasmidRange,
    _resolve_attr_id,
    _cache,
)


class TestLoadAttributes:
    """Tests for attribute loading."""

    def test_loads_valid_attributes(self):
        """Loads attributes from config file."""
        clear_cache()
        attrs = load_attributes()

        assert isinstance(attrs, dict)
        assert len(attrs) > 0

    def test_contains_expected_attributes(self):
        """Contains expected attribute mappings."""
        clear_cache()
        attrs = load_attributes()

        # These are common EVE attributes that should exist
        expected_attrs = ['damage', 'rof', 'cpu']
        for attr in expected_attrs:
            assert attr in attrs, f"Missing attribute: {attr}"
            assert isinstance(attrs[attr], int), f"{attr} should map to int ID"

    def test_caches_attributes(self):
        """Caches attributes after first load."""
        clear_cache()
        attrs1 = load_attributes()
        attrs2 = load_attributes()

        # Should return cached reference
        assert attrs1 is attrs2

    def test_attribute_ids_are_positive(self):
        """All attribute IDs are positive integers."""
        clear_cache()
        attrs = load_attributes()

        for name, attr_id in attrs.items():
            assert attr_id > 0, f"Invalid ID for {name}: {attr_id}"


class TestLoadConstants:
    """Tests for constants loading."""

    def test_loads_valid_constants(self):
        """Loads constants from config file."""
        clear_cache()
        constants = load_constants()

        assert isinstance(constants, dict)
        assert len(constants) > 0

    def test_has_required_sections(self):
        """Constants have all required sections."""
        clear_cache()
        constants = load_constants()

        required_sections = ['simulation', 'analysis', 'api', 'cache']
        for section in required_sections:
            assert section in constants, f"Missing section: {section}"

    def test_simulation_has_num_samples(self):
        """Simulation section has num_samples."""
        clear_cache()
        constants = load_constants()

        assert 'num_samples' in constants['simulation']
        assert constants['simulation']['num_samples'] > 0

    def test_api_has_required_urls(self):
        """API section has required URLs."""
        clear_cache()
        constants = load_constants()

        assert 'fuzzwork_market_url' in constants['api']
        assert constants['api']['fuzzwork_market_url'].startswith('http')


class TestLoadAllTargets:
    """Tests for target loading."""

    def test_loads_all_target_files(self):
        """Loads targets from all YAML files."""
        clear_cache()
        targets = load_all_targets()

        assert isinstance(targets, dict)
        assert len(targets) > 0

    def test_targets_are_roll_target_instances(self):
        """All targets are RollTarget instances."""
        clear_cache()
        targets = load_all_targets()

        for key, target in targets.items():
            assert isinstance(target, RollTarget), f"{key} is not RollTarget"
            assert target.key == key

    def test_target_has_required_fields(self):
        """Each target has all required fields."""
        clear_cache()
        targets = load_all_targets()

        for key, target in targets.items():
            assert target.base_type_id > 0, f"{key} missing base_type_id"
            assert target.muta_type_id > 0, f"{key} missing muta_type_id"
            assert len(target.muta_ranges) > 0, f"{key} missing muta_ranges"
            assert target.module_type, f"{key} missing module_type"

    def test_muta_ranges_are_valid(self):
        """Muta ranges have valid min/max values."""
        clear_cache()
        targets = load_all_targets()

        for key, target in targets.items():
            for i, muta_range in enumerate(target.muta_ranges):
                assert isinstance(muta_range, MutaplasmidRange), \
                    f"{key} muta_range[{i}] is not MutaplasmidRange"
                assert muta_range.min_mult <= muta_range.max_mult, \
                    f"{key} muta_range[{i}] has min > max"
                assert muta_range.attr_id > 0, \
                    f"{key} muta_range[{i}] has invalid attr_id"

    def test_caches_targets(self):
        """Caches targets after first load."""
        clear_cache()
        targets1 = load_all_targets()
        targets2 = load_all_targets()

        assert targets1 is targets2


class TestGetTarget:
    """Tests for getting specific targets."""

    def test_returns_existing_target(self):
        """Returns target for valid key."""
        clear_cache()
        targets = load_all_targets()
        first_key = list(targets.keys())[0]

        target = get_target(first_key)

        assert isinstance(target, RollTarget)
        assert target.key == first_key

    def test_raises_for_invalid_key(self):
        """Raises KeyError for invalid target key."""
        clear_cache()

        with pytest.raises(KeyError) as exc_info:
            get_target('nonexistent_target_key_xyz')

        assert 'nonexistent_target_key_xyz' in str(exc_info.value)


class TestResolveAttrId:
    """Tests for attribute ID resolution."""

    def test_resolves_valid_name(self):
        """Resolves valid attribute name to ID."""
        attrs = {'damage': 64, 'rof': 204, 'cpu': 50}

        result = _resolve_attr_id('damage', attrs)

        assert result == 64

    def test_raises_for_invalid_name(self):
        """Raises ValueError for unknown attribute name."""
        attrs = {'damage': 64}

        with pytest.raises(ValueError) as exc_info:
            _resolve_attr_id('unknown_attr', attrs)

        assert 'unknown_attr' in str(exc_info.value)


class TestClearCache:
    """Tests for cache clearing."""

    def test_clears_all_caches(self):
        """Clears all cached configuration data."""
        # Load everything first
        load_attributes()
        load_constants()
        load_all_targets()

        # Verify cache is populated
        assert _cache.get_attributes() is not None
        assert _cache.get_constants() is not None
        assert _cache.get_targets() is not None

        # Clear cache
        clear_cache()

        # Verify cache is cleared
        assert _cache.get_attributes() is None
        assert _cache.get_constants() is None
        assert _cache.get_targets() is None

    def test_clear_allows_reload(self):
        """Clearing cache allows fresh reload."""
        clear_cache()
        attrs1 = load_attributes()

        clear_cache()
        attrs2 = load_attributes()

        # Different object references after clear
        assert attrs1 is not attrs2
        # But same content
        assert attrs1 == attrs2


class TestThreadSafety:
    """Tests for thread-safe cache operations."""

    def test_cache_uses_lock(self):
        """ConfigCache uses a lock for thread safety."""
        assert hasattr(_cache, '_lock')
        assert _cache._lock is not None

    def test_concurrent_loads_are_safe(self):
        """Concurrent loads don't cause race conditions."""
        import threading

        clear_cache()
        results = []
        errors = []

        def load_attrs():
            try:
                attrs = load_attributes()
                results.append(attrs)
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = [threading.Thread(target=load_attrs) for _ in range(10)]

        # Start all threads
        for t in threads:
            t.start()

        # Wait for completion
        for t in threads:
            t.join()

        # No errors should occur
        assert len(errors) == 0
        # All results should have the same content
        assert len(results) == 10
        assert all(r == results[0] for r in results)
