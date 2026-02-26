"""
Comprehensive EDGAR Sentinel Backtest Script
============================================
Runs a full backtest pipeline for ~20 S&P 500 companies, 2024-present.

Pipeline:
1. Run similarity analyzer on all ingested filings
2. Generate composite signals for each quarterly rebalance date (Q1 2024 - Q1 2026)
3. Run backtest simulation (2024-01-01 to 2026-02-26)
4. Save detailed results report
"""

import asyncio
import json
import logging
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("backtest_runner")

# Suppress noisy yfinance logs
logging.getLogger("yfinance").setLevel(logging.WARNING)
logging.getLogger("peewee").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


async def run_similarity_analysis(store, config) -> int:
    """Run similarity analyzer on all ingested filings.

    For each ticker, pairs consecutive same-type filings and computes
    TF-IDF cosine similarity.

    Returns count of new similarity results saved.
    """
    from edgar_sentinel.analyzers import SimilarityAnalyzer
    from edgar_sentinel.core import FormType

    analyzer = SimilarityAnalyzer(config.analyzers.similarity)
    total_saved = 0
    total_skipped = 0

    # Get all tickers
    all_filings = await store.list_filings()
    tickers = sorted(set(f.ticker for f in all_filings if f.ticker))
    logger.info("Running similarity analysis for %d tickers", len(tickers))

    for ticker in tickers:
        filings = await store.list_filings(ticker=ticker)
        # Sort by date ascending
        filings.sort(key=lambda f: f.filed_date)

        # Group by form_type
        by_type: dict[str, list] = {}
        for f in filings:
            ft = f.form_type.value if hasattr(f.form_type, 'value') else str(f.form_type)
            by_type.setdefault(ft, []).append(f)

        for form_type_str, type_filings in by_type.items():
            if len(type_filings) < 2:
                continue  # Need at least 2 for similarity

            for i in range(1, len(type_filings)):
                current_meta = type_filings[i]
                prior_meta = type_filings[i - 1]

                # Check if already analyzed
                existing = await store.get_similarity(current_meta.accession_number)
                if existing:
                    total_skipped += 1
                    continue

                # Load full filings
                current_filing = await store.get_filing(current_meta.accession_number)
                prior_filing = await store.get_filing(prior_meta.accession_number)

                if current_filing is None or prior_filing is None:
                    continue

                # Analyze each section
                for section_name, current_section in current_filing.sections.items():
                    prior_section = prior_filing.sections.get(section_name)
                    if prior_section is None:
                        continue

                    try:
                        result = analyzer.analyze(current_section, prior_section)
                        await store.save_similarity(result)
                        total_saved += 1
                    except Exception as e:
                        logger.debug("Similarity failed for %s/%s: %s", ticker, section_name, e)

    logger.info(
        "Similarity analysis complete: %d new results, %d already existed",
        total_saved, total_skipped,
    )
    return total_saved


async def generate_quarterly_signals(store, config, quarter_end_dates: list[date]) -> dict[date, int]:
    """Generate composite signals for each quarterly rebalance date.

    For each date, builds signals using only filings available as of that date,
    then saves composite signals with signal_date = quarter_end_date.

    Returns dict of date -> count of composites generated.
    """
    from edgar_sentinel.analyzers.base import AnalysisResults
    from edgar_sentinel.core import CompositeMethod
    from edgar_sentinel.signals.builder import FilingDateMapping, SignalBuilder
    from edgar_sentinel.signals.composite import SignalComposite

    builder = SignalBuilder(config.signals)
    composite = SignalComposite(method=CompositeMethod.EQUAL)

    results_by_date: dict[date, int] = {}

    # Pre-load all filings and sentiments/similarities once
    all_filings_meta = await store.list_filings()
    logger.info("Loaded %d filing metadata records", len(all_filings_meta))

    # Load all sentiments
    all_sentiments = []
    all_similarities = []
    filing_dates: dict[str, FilingDateMapping] = {}

    for meta in all_filings_meta:
        if meta.ticker is None:
            continue
        sents = await store.get_sentiments(meta.accession_number)
        all_sentiments.extend(sents)
        sims = await store.get_similarity(meta.accession_number)
        all_similarities.extend(sims)
        filing_dates[meta.accession_number] = FilingDateMapping(
            ticker=meta.ticker,
            filing_id=meta.accession_number,
            filed_date=meta.filed_date,
            signal_date=meta.filed_date,
        )

    logger.info(
        "Loaded %d sentiments, %d similarities from %d filings",
        len(all_sentiments), len(all_similarities), len(filing_dates),
    )

    analysis_results = AnalysisResults(
        sentiment_results=all_sentiments,
        similarity_results=all_similarities,
    )

    for as_of in quarter_end_dates:
        logger.info("Generating signals for %s...", as_of)

        try:
            raw_signals = builder.build(analysis_results, filing_dates, as_of)

            if not raw_signals:
                logger.warning("No raw signals for %s", as_of)
                results_by_date[as_of] = 0
                continue

            composites = composite.combine(raw_signals, as_of_date=as_of)

            for c in composites:
                await store.save_composite(c)

            logger.info(
                "  %s: %d composites across %d tickers",
                as_of, len(composites), len({c.ticker for c in composites}),
            )
            results_by_date[as_of] = len(composites)

        except Exception as e:
            logger.error("Signal generation failed for %s: %s", as_of, e)
            results_by_date[as_of] = 0

    return results_by_date


async def run_full_backtest(store, config, start: date, end: date) -> dict:
    """Run the full backtest simulation.

    Returns detailed results dict including metrics and monthly returns.
    """
    from edgar_sentinel.backtest import run_backtest
    from edgar_sentinel.backtest.metrics import MetricsCalculator
    from edgar_sentinel.backtest.returns import YFinanceProvider
    from edgar_sentinel.core import BacktestConfig, RebalanceFrequency

    # Load all composite signals
    composites = await store.get_composites()

    if not composites:
        raise ValueError("No composite signals found")

    # Filter to signals within backtest window + some buffer
    universe = sorted({c.ticker for c in composites})
    logger.info(
        "Backtest universe: %d tickers, %d total signals",
        len(universe), len(composites),
    )

    # Check signals at each rebalance date
    for signal in composites[:5]:
        logger.info("  Sample signal: %s on %s = %.4f", signal.ticker, signal.signal_date, signal.composite_score)

    # Run quarterly backtest (long-only for simplicity with ~20 stocks)
    # 5 quantiles requires >= 5 tickers, we have 20, so that's fine
    # Long quantile 1 = highest composite scores
    bt_config = BacktestConfig(
        start_date=start,
        end_date=end,
        universe=universe,
        rebalance_frequency=RebalanceFrequency.QUARTERLY,
        num_quantiles=5,
        long_quantile=1,
        short_quantile=5,  # Long-short strategy
        transaction_cost_bps=10,
    )

    # Use yfinance for returns
    returns_provider = YFinanceProvider(
        cache_db_path=config.storage.sqlite_path,
        request_buffer_days=5,
    )

    result = run_backtest(
        config=bt_config,
        signals=composites,
        returns_provider=returns_provider,
    )

    logger.info("Backtest complete:")
    logger.info("  Total return: %.2f%%", result.total_return * 100)
    logger.info("  Annualized return: %.2f%%", result.annualized_return * 100)
    logger.info("  Sharpe ratio: %.3f", result.sharpe_ratio)
    logger.info("  Max drawdown: %.2f%%", result.max_drawdown * 100)

    # Also run long-only for comparison
    bt_config_long = BacktestConfig(
        start_date=start,
        end_date=end,
        universe=universe,
        rebalance_frequency=RebalanceFrequency.QUARTERLY,
        num_quantiles=5,
        long_quantile=1,
        short_quantile=None,
        transaction_cost_bps=10,
    )

    result_long = run_backtest(
        config=bt_config_long,
        signals=composites,
        returns_provider=returns_provider,
    )

    logger.info("Long-only backtest:")
    logger.info("  Total return: %.2f%%", result_long.total_return * 100)
    logger.info("  Annualized return: %.2f%%", result_long.annualized_return * 100)

    # Get universe returns for IC calculation
    universe_returns = returns_provider.get_returns(
        tickers=universe, start=start, end=end, frequency="monthly"
    )

    calc = MetricsCalculator()
    metrics = calc.compute_all(
        returns=pd.Series(
            [mr.long_return for mr in result.monthly_returns],
            index=pd.DatetimeIndex([mr.period_end for mr in result.monthly_returns]),
        ),
        signals=composites,
        universe_returns=universe_returns,
    )

    return {
        "long_short": result,
        "long_only": result_long,
        "metrics": metrics,
        "universe": universe,
        "universe_returns": universe_returns,
    }


def format_report(results: dict, signals_by_quarter: dict, start: date, end: date) -> str:
    """Format comprehensive backtest report as markdown."""
    ls = results["long_short"]
    lo = results["long_only"]
    metrics = results["metrics"]
    universe = results["universe"]

    lines = [
        "# EDGAR Sentinel Backtest Report",
        f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}",
        f"**Period:** {start} → {end}",
        f"**Universe:** {len(universe)} S&P 500 companies",
        "",
        "## Overview",
        "",
        "This backtest evaluates the EDGAR Sentinel signal system using SEC filing",
        "sentiment analysis as a return predictor. Signals are derived from:",
        "- **Loughran-McDonald Dictionary**: Word-count sentiment scores from 10-K/10-Q filings",
        "- **TF-IDF Cosine Similarity**: Change detection between consecutive same-type filings",
        "",
        "**Rebalancing:** Quarterly (end of Q1, Q2, Q3, Q4)",
        "**Signal delay:** 2 days after filing date (to account for processing time)",
        "**Signal decay:** 90-day half-life",
        "**Transaction costs:** 10 bps per turnover",
        "",
        "## Universe",
        "",
        "| Ticker | " + " | ".join([""] * 2) + "|",
        "|--------|---------|",
    ]

    # Universe table (3 columns)
    for i in range(0, len(universe), 3):
        row = universe[i:i+3]
        while len(row) < 3:
            row.append("")
        lines.append(f"| {' | '.join(row)} |")

    lines += [
        "",
        "## Strategy Performance",
        "",
        "### Long-Short Strategy (Q1 Long / Q5 Short)",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total Return | {ls.total_return:.1%} |",
        f"| Annualized Return | {ls.annualized_return:.1%} |",
        f"| Sharpe Ratio | {ls.sharpe_ratio:.3f} |",
        f"| Max Drawdown | {ls.max_drawdown:.1%} |",
        f"| Avg Turnover | {ls.turnover:.1%} |",
    ]

    if metrics.get("sortino_ratio") is not None:
        lines.append(f"| Sortino Ratio | {metrics['sortino_ratio']:.3f} |")
    if metrics.get("win_rate") is not None:
        lines.append(f"| Win Rate | {metrics['win_rate']:.1%} |")
    if metrics.get("information_coefficient") is not None:
        lines.append(f"| Mean IC (Rank Corr) | {metrics['information_coefficient']:.4f} |")
    if ls.information_ratio is not None:
        lines.append(f"| Information Ratio (ICIR) | {ls.information_ratio:.3f} |")

    lines += [
        "",
        "### Long-Only Strategy (Q1 Long)",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total Return | {lo.total_return:.1%} |",
        f"| Annualized Return | {lo.annualized_return:.1%} |",
        f"| Sharpe Ratio | {lo.sharpe_ratio:.3f} |",
        f"| Max Drawdown | {lo.max_drawdown:.1%} |",
        f"| Avg Turnover | {lo.turnover:.1%} |",
    ]

    lines += [
        "",
        "## Monthly Returns (Long-Short)",
        "",
        "| Period End | Long Return | Short Return | Net Return |",
        "|-----------|------------|-------------|-----------|",
    ]

    for mr in ls.monthly_returns:
        short_str = f"{mr.short_return:.2%}" if mr.short_return is not None else "N/A"
        ls_str = f"{mr.long_short_return:.2%}" if mr.long_short_return is not None else "N/A"
        lines.append(
            f"| {mr.period_end} | {mr.long_return:.2%} | {short_str} | {ls_str} |"
        )

    lines += [
        "",
        "## Signal Generation by Quarter",
        "",
        "| Quarter End | Composite Signals |",
        "|------------|------------------|",
    ]

    for qdate, count in sorted(signals_by_quarter.items()):
        lines.append(f"| {qdate} | {count} |")

    if metrics.get("factor_exposures"):
        lines += [
            "",
            "## Factor Exposures (Fama-French)",
            "",
            "| Factor | Loading |",
            "|--------|---------|",
        ]
        for factor, loading in metrics["factor_exposures"].items():
            lines.append(f"| {factor} | {loading:.4f} |")
        if metrics.get("alpha") is not None:
            lines.append(f"| **Alpha** | **{metrics['alpha']:.4f}** |")

    lines += [
        "",
        "## Methodology Notes",
        "",
        "### Signal Construction",
        "1. **Filing ingestion**: 10-K and 10-Q filings downloaded from EDGAR for all tickers",
        "2. **Dictionary analysis**: Loughran-McDonald positive/negative word counts applied to each section",
        "3. **Similarity analysis**: TF-IDF cosine similarity computed between consecutive same-type filings",
        "4. **Signal normalization**: Cross-sectional z-score within each signal type at each rebalance date",
        "5. **Decay weighting**: Exponential decay with 90-day half-life applied before compositing",
        "6. **Composite score**: Equal-weighted average of normalized components",
        "",
        "### Backtest Construction",
        "- At each quarterly rebalance date, tickers ranked by composite score",
        "- Top quintile (Q1 = highest sentiment) → Long positions (equal-weighted)",
        "- Bottom quintile (Q5 = lowest/most negative sentiment) → Short positions (equal-weighted)",
        "- Only signals available on or before the rebalance date are used (no look-ahead bias)",
        "",
        "### Limitations",
        "- LLM analyzer disabled (would require API calls per invocation)",
        "- Historical filings only available from early 2024 (short backtest window)",
        "- Universe limited to 20 large-cap stocks (S&P 500 subset)",
        "- Survivorship bias possible (all selected companies are current S&P constituents)",
        "",
        "## Conclusion",
    ]

    # Conclusion based on results
    sharpe = ls.sharpe_ratio
    if sharpe > 0.5:
        conclusion = (
            f"The strategy generated a Sharpe ratio of {sharpe:.2f} over the backtest period, "
            "suggesting the EDGAR filing sentiment signal has meaningful return-predictive content. "
            "However, the 2-year window is short for definitive conclusions."
        )
    elif sharpe > 0:
        conclusion = (
            f"The strategy generated a positive but modest Sharpe ratio of {sharpe:.2f}. "
            "The signal shows some predictive power, though a longer history would strengthen "
            "statistical confidence."
        )
    else:
        conclusion = (
            f"The strategy did not outperform on a risk-adjusted basis (Sharpe: {sharpe:.2f}). "
            "This may reflect the short backtest window, or that the current signal construction "
            "needs refinement. The LLM analyzer (disabled) and longer history would likely improve results."
        )

    lines += [conclusion, ""]

    return "\n".join(lines)


async def main():
    """Main backtest orchestration."""
    from edgar_sentinel.core import load_config
    from edgar_sentinel.ingestion import create_store

    # Load config from the edgar-sentinel directory
    config = load_config(config_path="edgar-sentinel.yml")
    store = await create_store(config.storage)

    try:
        # Step 1: Similarity analysis
        logger.info("=" * 60)
        logger.info("STEP 1: Running similarity analysis")
        logger.info("=" * 60)
        similarity_count = await run_similarity_analysis(store, config)
        logger.info("Similarity analysis done: %d results", similarity_count)

        # Step 2: Generate quarterly composite signals
        logger.info("=" * 60)
        logger.info("STEP 2: Generating quarterly composite signals")
        logger.info("=" * 60)

        # Quarter-end dates from Q1 2024 through Q1 2026
        quarter_ends = [
            date(2024, 3, 29),   # Q1 2024 (last business day of Mar)
            date(2024, 6, 28),   # Q2 2024
            date(2024, 9, 30),   # Q3 2024
            date(2024, 12, 31),  # Q4 2024
            date(2025, 3, 31),   # Q1 2025
            date(2025, 6, 30),   # Q2 2025
            date(2025, 9, 30),   # Q3 2025
            date(2025, 12, 31),  # Q4 2025
            date(2026, 2, 26),   # Current date (for reference)
        ]

        signals_by_quarter = await generate_quarterly_signals(store, config, quarter_ends)
        total_signals = sum(signals_by_quarter.values())
        logger.info("Signal generation done: %d total composites across %d quarters",
                    total_signals, len(quarter_ends))

        # Step 3: Run backtest
        logger.info("=" * 60)
        logger.info("STEP 3: Running backtest (2024-01-01 to 2026-02-26)")
        logger.info("=" * 60)

        backtest_start = date(2024, 1, 1)
        backtest_end = date(2026, 2, 26)

        results = await run_full_backtest(store, config, backtest_start, backtest_end)

        # Step 4: Save report
        logger.info("=" * 60)
        logger.info("STEP 4: Saving backtest report")
        logger.info("=" * 60)

        report = format_report(results, signals_by_quarter, backtest_start, backtest_end)

        output_path = Path("/output/research/edgar-sentiment/backtest-results.md")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report)
        logger.info("Report saved to %s", output_path)

        # Also save JSON for programmatic access
        json_path = Path("/output/research/edgar-sentiment/backtest-results.json")
        ls = results["long_short"]
        lo = results["long_only"]
        metrics = results["metrics"]

        json_output = {
            "generated_at": datetime.now().isoformat(),
            "period": {"start": str(backtest_start), "end": str(backtest_end)},
            "universe": results["universe"],
            "long_short": {
                "total_return": ls.total_return,
                "annualized_return": ls.annualized_return,
                "sharpe_ratio": ls.sharpe_ratio,
                "max_drawdown": ls.max_drawdown,
                "turnover": ls.turnover,
                "information_ratio": ls.information_ratio,
            },
            "long_only": {
                "total_return": lo.total_return,
                "annualized_return": lo.annualized_return,
                "sharpe_ratio": lo.sharpe_ratio,
                "max_drawdown": lo.max_drawdown,
                "turnover": lo.turnover,
            },
            "metrics": {
                k: v for k, v in metrics.items()
                if v is not None and not isinstance(v, dict)
            },
            "signals_by_quarter": {str(k): v for k, v in signals_by_quarter.items()},
            "monthly_returns": [
                {
                    "period_end": str(mr.period_end),
                    "long_return": mr.long_return,
                    "short_return": mr.short_return,
                    "net_return": mr.long_short_return,
                }
                for mr in ls.monthly_returns
            ],
        }

        json_path.write_text(json.dumps(json_output, indent=2, default=str))
        logger.info("JSON saved to %s", json_path)

        # Print summary
        print("\n" + "=" * 60)
        print("BACKTEST COMPLETE")
        print("=" * 60)
        print(f"Universe: {len(results['universe'])} tickers")
        print(f"Period: {backtest_start} to {backtest_end}")
        print()
        print("Long-Short Strategy:")
        print(f"  Total Return:       {ls.total_return:+.2%}")
        print(f"  Annualized Return:  {ls.annualized_return:+.2%}")
        print(f"  Sharpe Ratio:       {ls.sharpe_ratio:.3f}")
        print(f"  Max Drawdown:       {ls.max_drawdown:.2%}")
        print()
        print("Long-Only Strategy (Q1):")
        print(f"  Total Return:       {lo.total_return:+.2%}")
        print(f"  Annualized Return:  {lo.annualized_return:+.2%}")
        print(f"  Sharpe Ratio:       {lo.sharpe_ratio:.3f}")
        print()
        if metrics.get("information_coefficient") is not None:
            print(f"Mean IC (Rank Corr): {metrics['information_coefficient']:.4f}")
        print()
        print(f"Report: {output_path}")
        print()

    finally:
        await store.close()


if __name__ == "__main__":
    asyncio.run(main())
