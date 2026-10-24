"""
High-Level Backtest Execution Orchestrator and Run Persistence Manager.
Loads datasets, enriches technical indicators, runs simulation, produces analytics,
and persists immutable run artifacts for full historical reproducibility.
"""

from datetime import datetime
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd

from core.backtesting.analytics import BacktestAnalytics, BacktestResult
from core.backtesting.costs import IndianCostModel
from core.backtesting.data_provider import YahooHistoricalDataProvider, LocalHistoricalDataProvider
from core.backtesting.engine import BacktestEngine
from core.backtesting.strategy import BuiltinStrategies, StrategyConfig, Strategy
from core.indicators import enrich_with_indicators


class BacktestRunner:
    """Orchestrator for managing and persisting backtest workflows."""

    STORAGE_DIR = Path("data/backtests")
    ENGINE_VERSION = "2.0.0-PROD"

    @classmethod
    def run_backtest(
        cls,
        strategy_id: str,
        symbol: str = "NIFTY",
        start_date: str = "2023-01-01",
        end_date: str = "2024-01-01",
        interval: str = "1d",
        segment: str = "OPTIONS",
        initial_capital: float = 200000.0,
        stop_loss_pct: float = 0.03,
        target_pct: float = 0.06,
        max_holding_bars: int = 10,
        brokerage_per_order: float = 20.0,
        slippage_pct: float = 0.0005,
        enable_statutory_costs: bool = True,
        custom_rules: Optional[Dict[str, Any]] = None,
        use_local_csv_first: bool = True,
    ) -> BacktestResult:
        """
        Executes a complete backtest cycle end-to-end.
        """
        cls.STORAGE_DIR.mkdir(parents=True, exist_ok=True)

        # 1. Historical Data Retrieval
        data_provider = None
        df_bars = pd.DataFrame()
        dq_report = None

        if use_local_csv_first:
            local_prov = LocalHistoricalDataProvider()
            df_bars, dq_report = local_prov.get_historical_bars(symbol, start_date, end_date, interval)

        if df_bars.empty:
            y_prov = YahooHistoricalDataProvider()
            df_bars, dq_report = y_prov.get_historical_bars(symbol, start_date, end_date, interval)
            data_provider = y_prov
        else:
            data_provider = local_prov

        if df_bars.empty:
            return BacktestAnalytics._empty_result(strategy_id, cls.ENGINE_VERSION, symbol, initial_capital)

        # 2. Indicator Enrichment
        df_bars = enrich_with_indicators(df_bars.copy())

        # 3. Contract Specifications
        specs = data_provider.get_contract_specifications(symbol)
        lot_size = specs.get("lot_size", 50)
        strike_step = float(specs.get("strike_step", 50.0))

        # 4. Strategy Instantiation
        strat_cfg = StrategyConfig(
            name=BuiltinStrategies.list_strategies().get(strategy_id, strategy_id),
            version=cls.ENGINE_VERSION,
            segment=segment,
            stop_loss_pct=stop_loss_pct,
            target_pct=target_pct,
            max_holding_bars=max_holding_bars,
        )

        strategy = BuiltinStrategies.create(
            strategy_id,
            config=strat_cfg,
            rules=custom_rules or {},
        )

        # 5. Cost Model
        cost_model = IndianCostModel(
            brokerage_per_order=brokerage_per_order,
            slippage_pct=slippage_pct,
            enable_statutory=enable_statutory_costs,
        )

        # 6. Engine Execution
        engine = BacktestEngine(
            strategy=strategy,
            initial_capital=initial_capital,
            cost_model=cost_model,
        )

        engine_out = engine.run(
            market_data=df_bars,
            symbol=symbol,
            lot_size=lot_size,
            strike_step=strike_step,
        )

        # 7. Analytics & Benchmark Attribution
        result = BacktestAnalytics.analyze(
            engine_output=engine_out,
            benchmark_data=None,  # spot_close is embedded inside engine_output equity_curve
            data_quality_report=dq_report.to_dict() if dq_report else {},
        )

        # 8. Persist Run Artifact for Reproducibility
        cls._save_run(result, {
            "strategy_id": strategy_id,
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "interval": interval,
            "segment": segment,
            "initial_capital": initial_capital,
            "stop_loss_pct": stop_loss_pct,
            "target_pct": target_pct,
            "max_holding_bars": max_holding_bars,
            "brokerage": brokerage_per_order,
            "slippage_pct": slippage_pct,
            "statutory": enable_statutory_costs,
        })

        return result

    @classmethod
    def _save_run(cls, result: BacktestResult, config: Dict[str, Any]):
        """Persists run metadata and performance summary."""
        try:
            run_id = f"RUN_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{result.symbol}_{result.strategy_name.replace(' ', '_')[:10]}"
            filepath = cls.STORAGE_DIR / f"{run_id}.json"

            payload = {
                "run_id": run_id,
                "engine_version": cls.ENGINE_VERSION,
                "created_at": datetime.now().isoformat(),
                "config": config,
                "summary": result.to_summary_dict(),
                "data_quality": result.data_quality_report,
                "trades_count": len(result.trades),
                "trades_sample": [t.to_dict() for t in result.trades[:30]],
            }

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception:
            pass

    @classmethod
    def list_saved_runs(cls) -> List[Dict[str, Any]]:
        """Retrieves list of past backtest runs sorted chronologically descending."""
        runs = []
        if not cls.STORAGE_DIR.exists():
            return []
        for file in cls.STORAGE_DIR.glob("RUN_*.json"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    runs.append(data)
            except Exception:
                continue
        runs.sort(key=lambda r: r.get("created_at", ""), reverse=True)
        return runs
