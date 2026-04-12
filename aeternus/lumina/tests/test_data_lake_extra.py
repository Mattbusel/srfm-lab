"""Tests for data_lake.py extended components."""
import pytest
import numpy as np
import torch

class TestDataSchemaField:
    def test_valid_float(self):
        from data_lake import DataSchemaField
        field = DataSchemaField('price', 'float32', min_val=0.0, max_val=1000.0)
        ok, msg = field.validate(50.0)
        assert ok

    def test_below_min(self):
        from data_lake import DataSchemaField
        field = DataSchemaField('price', 'float32', min_val=0.0)
        ok, msg = field.validate(-1.0)
        assert not ok

    def test_nullable_none(self):
        from data_lake import DataSchemaField
        field = DataSchemaField('optional', 'float32', nullable=True)
        ok, msg = field.validate(None)
        assert ok

    def test_non_nullable_none(self):
        from data_lake import DataSchemaField
        field = DataSchemaField('required', 'float32', nullable=False)
        ok, msg = field.validate(None)
        assert not ok

    def test_enum_validation(self):
        from data_lake import DataSchemaField
        field = DataSchemaField('side', 'string', enum_values=['buy', 'sell'])
        ok1, _ = field.validate('buy')
        ok2, _ = field.validate('hold')
        assert ok1
        assert not ok2


class TestDataSchema:
    def _make_schema(self):
        from data_lake import DataSchema, DataSchemaField
        fields = [
            DataSchemaField('price', 'float32', min_val=0.0),
            DataSchemaField('volume', 'float32', min_val=0.0, nullable=True),
        ]
        return DataSchema('ohlcv', fields)

    def test_validate_valid_record(self):
        schema = self._make_schema()
        record = {'price': 100.0, 'volume': 1000.0}
        ok, errors = schema.validate_record(record)
        assert ok
        assert len(errors) == 0

    def test_validate_invalid_record(self):
        schema = self._make_schema()
        record = {'price': -10.0, 'volume': 1000.0}
        ok, errors = schema.validate_record(record)
        assert not ok
        assert len(errors) > 0

    def test_validate_batch(self):
        schema = self._make_schema()
        records = [
            {'price': 100.0, 'volume': 1000.0},
            {'price': -1.0, 'volume': 500.0},
            {'price': 200.0, 'volume': None},
        ]
        result = schema.validate_batch(records)
        assert result['total'] == 3
        assert result['valid'] >= 2


class TestStreamingDataPipeline:
    def test_add_transformation(self):
        from data_lake import StreamingDataPipeline
        pipeline = StreamingDataPipeline(None)
        pipeline.add_transformation(lambda x: x * 2)
        result = pipeline.process_record(5)
        assert result == 10

    def test_add_filter(self):
        from data_lake import StreamingDataPipeline
        pipeline = StreamingDataPipeline(None)
        pipeline.add_filter(lambda x: x > 0)
        result = pipeline.process_record(-1)
        assert result is None

    def test_stats_tracking(self):
        from data_lake import StreamingDataPipeline
        pipeline = StreamingDataPipeline(None)
        pipeline.add_filter(lambda x: x > 0)
        for v in [1, -1, 2, -2, 3]:
            pipeline.process_record(v)
        stats = pipeline.get_stats()
        assert stats['records_in'] == 5
        assert stats['records_out'] == 3
        assert stats['records_filtered'] == 2


class TestFinancialCalendar:
    def test_weekday_is_trading(self):
        from data_lake import FinancialCalendar
        cal = FinancialCalendar()
        # Wednesday 2024-01-03
        assert cal.is_trading_day(2024, 1, 3)

    def test_weekend_not_trading(self):
        from data_lake import FinancialCalendar
        cal = FinancialCalendar()
        # Saturday 2024-01-06
        assert not cal.is_trading_day(2024, 1, 6)

    def test_holiday_not_trading(self):
        from data_lake import FinancialCalendar
        cal = FinancialCalendar()
        # New Year's Day (Jan 1) - assume not weekend
        # 2024-01-01 is Monday
        assert not cal.is_trading_day(2024, 1, 1)

    def test_custom_holiday(self):
        from data_lake import FinancialCalendar
        cal = FinancialCalendar()
        cal.add_holiday(3, 15)  # March 15
        assert not cal.is_trading_day(2024, 3, 15)


class TestMarketMicrostructureProcessor:
    def test_process_quote(self):
        from data_lake import MarketMicrostructureProcessor
        proc = MarketMicrostructureProcessor()
        result = proc.process_quote(99.9, 100.1, 1000, 800, time=0.0)
        assert 'mid' in result
        assert result['mid'] == pytest.approx(100.0)
        assert result['spread'] == pytest.approx(0.2, abs=1e-6)

    def test_kyle_lambda(self):
        from data_lake import MarketMicrostructureProcessor
        proc = MarketMicrostructureProcessor()
        trades = [
            {'price': 100.0, 'size': 100, 'side': 1},
            {'price': 100.1, 'size': 100, 'side': 1},
            {'price': 100.2, 'size': 100, 'side': 1},
        ]
        lam = proc.kyle_lambda(trades)
        assert lam >= 0

    def test_amihud_illiquidity(self):
        from data_lake import MarketMicrostructureProcessor
        proc = MarketMicrostructureProcessor()
        prices = [100.0, 100.5, 99.5, 101.0]
        volumes = [1000, 2000, 500, 3000]
        ill = proc.amihud_illiquidity(prices, volumes)
        assert ill >= 0


class TestSyntheticDataGenerator:
    def test_gbm(self):
        from data_lake import SyntheticDataGenerator
        gen = SyntheticDataGenerator(seed=42)
        prices = gen.generate_gbm(252)
        assert len(prices) == 253  # S0 + 252
        assert (prices > 0).all()

    def test_heston(self):
        from data_lake import SyntheticDataGenerator
        gen = SyntheticDataGenerator(seed=0)
        prices, variances = gen.generate_heston(100)
        assert len(prices) == 101
        assert (variances >= 0).all()

    def test_multivariate_returns(self):
        from data_lake import SyntheticDataGenerator
        gen = SyntheticDataGenerator(seed=42)
        returns = gen.generate_multivariate_returns(252, n_assets=10)
        assert returns.shape == (252, 10)

    def test_regime_switching(self):
        from data_lake import SyntheticDataGenerator
        gen = SyntheticDataGenerator(seed=1)
        returns, labels = gen.generate_regime_switching(200)
        assert len(returns) == 200
        assert len(labels) == 200
        assert labels.min() >= 0

    def test_order_flow(self):
        from data_lake import SyntheticDataGenerator
        gen = SyntheticDataGenerator(seed=7)
        data = gen.generate_order_flow(100)
        assert 'prices' in data
        assert len(data['prices']) == 101
        assert (data['buy_volumes'] >= 0).all()


@pytest.mark.parametrize('n,n_assets', [
    (100, 5), (252, 10), (504, 20), (63, 3),
])
def test_multivariate_returns_shapes(n, n_assets):
    from data_lake import SyntheticDataGenerator
    gen = SyntheticDataGenerator()
    returns = gen.generate_multivariate_returns(n, n_assets=n_assets)
    assert returns.shape == (n, n_assets)
    assert not np.isnan(returns).any()
