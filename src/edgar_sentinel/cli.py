"""Click-based CLI for edgar-sentinel.

Thin wrapper around library modules. Zero business logic — every operation
delegates to ingestion, analyzers, signals, or backtest modules.
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import date, datetime

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console(stderr=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_async(coro):
    """Run an async coroutine from synchronous Click code."""
    return asyncio.run(coro)


def _load_config(ctx: click.Context):
    """Load config lazily, caching on first call."""
    if "config" not in ctx.obj:
        from edgar_sentinel.core import load_config

        ctx.obj["config"] = load_config(config_path=ctx.obj.get("config_path"))
    return ctx.obj["config"]


async def _create_store_async(config):
    """Create and initialize storage from config."""
    from edgar_sentinel.ingestion import create_store

    return await create_store(config.storage)


def _resolve_form_types(form_type: str) -> list:
    """Convert CLI form-type string to FormType list."""
    from edgar_sentinel.core import FormType

    if form_type == "both":
        return [FormType.FORM_10K, FormType.FORM_10Q]
    elif form_type == "10-K":
        return [FormType.FORM_10K]
    elif form_type == "10-Q":
        return [FormType.FORM_10Q]
    raise click.UsageError(f"Unknown form type: {form_type}")


def _resolve_tickers(universe: str, tickers: str | None) -> list[str]:
    """Resolve ticker list from universe name or comma-separated string."""
    if universe == "custom":
        if not tickers:
            raise click.UsageError("--tickers is required when universe=custom")
        return [t.strip().upper() for t in tickers.split(",") if t.strip()]
    raise click.UsageError(
        f"Universe '{universe}' not yet supported. Use --universe custom --tickers AAPL,MSFT"
    )


def _resolve_analyzers(names: tuple[str, ...], config):
    """Instantiate requested analyzer objects."""
    from edgar_sentinel.analyzers import DictionaryAnalyzer, LLMAnalyzer, SimilarityAnalyzer

    if "all" in names:
        names = ("dictionary", "similarity", "llm")

    analyzers = []
    for name in names:
        if name == "dictionary":
            if config.analyzers.dictionary.enabled:
                analyzers.append(DictionaryAnalyzer(config.analyzers.dictionary))
        elif name == "similarity":
            if config.analyzers.similarity.enabled:
                analyzers.append(SimilarityAnalyzer(config.analyzers.similarity))
        elif name == "llm":
            if not config.analyzers.llm.enabled:
                console.print("[yellow]LLM analyzer disabled in config. Skipping.[/yellow]")
                continue
            analyzers.append(LLMAnalyzer(config.analyzers.llm))
    return analyzers


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    envvar="EDGAR_SENTINEL_CONFIG",
    default=None,
    help="Path to edgar-sentinel.yml config file.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Enable verbose output.",
)
@click.version_option(package_name="edgar-sentinel")
@click.pass_context
def cli(ctx: click.Context, config: str | None, verbose: bool) -> None:
    """EDGAR Sentinel: SEC filing sentiment signal generator."""
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config
    ctx.obj["verbose"] = verbose
    ctx.obj["console"] = console


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------


@cli.command()
@click.option(
    "--universe",
    "-u",
    type=click.Choice(["custom"], case_sensitive=False),
    default="custom",
    help="Ticker universe (currently: 'custom').",
)
@click.option(
    "--tickers",
    "-t",
    type=str,
    default=None,
    help="Comma-separated list of tickers (required).",
)
@click.option(
    "--form-type",
    type=click.Choice(["10-K", "10-Q", "both"], case_sensitive=False),
    default="both",
    help="Filing types to download.",
)
@click.option("--start", "-s", type=int, required=True, help="Start year (e.g. 2020).")
@click.option("--end", "-e", type=int, required=True, help="End year (e.g. 2025).")
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Re-download even if filings already exist.",
)
@click.pass_context
def ingest(
    ctx: click.Context,
    universe: str,
    tickers: str | None,
    form_type: str,
    start: int,
    end: int,
    force: bool,
) -> None:
    """Download and parse SEC filings from EDGAR."""
    config = _load_config(ctx)
    ticker_list = _resolve_tickers(universe, tickers)
    form_types = _resolve_form_types(form_type)

    async def _run():
        from edgar_sentinel.ingestion import EdgarClient, FilingParser

        store = await _create_store_async(config)
        try:
            client = EdgarClient(config.edgar)
            parser = FilingParser()

            total_new = 0
            total_errors = 0

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Ingesting filings...", total=len(ticker_list))
                for ticker in ticker_list:
                    try:
                        from datetime import date as _date
                        filings_meta = await client.get_filings_for_ticker(
                            ticker=ticker,
                            form_types=form_types,
                            start_date=_date(start, 1, 1),
                            end_date=_date(end, 12, 31),
                        )
                        for meta in filings_meta:
                            if not force and await store.filing_exists(
                                meta.accession_number
                            ):
                                continue
                            url = await client.get_filing_url(meta)
                            raw = await client.get_filing_document(url)
                            sections = parser.parse(raw, meta.form_type, meta.accession_number)
                            from edgar_sentinel.core import Filing

                            filing = Filing(metadata=meta, sections=sections)
                            await store.save_filing(filing)
                            total_new += 1
                    except Exception as exc:
                        total_errors += 1
                        if ctx.obj["verbose"]:
                            console.print(f"[red]Error for {ticker}: {exc}[/red]")
                    progress.advance(task)

            console.print(
                f"[green]\u2713[/green] Ingested {total_new} new filings "
                f"for {len(ticker_list)} tickers"
                + (f" ({total_errors} errors)" if total_errors else "")
            )
        finally:
            await store.close()

    _run_async(_run())


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------


@cli.command()
@click.option(
    "--analyzer",
    "-a",
    type=click.Choice(["dictionary", "similarity", "llm", "all"], case_sensitive=False),
    multiple=True,
    default=("all",),
    help="Analyzers to run. Can specify multiple: -a dictionary -a llm",
)
@click.option(
    "--tickers",
    "-t",
    type=str,
    default=None,
    help="Comma-separated tickers. Default: all ingested.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Re-analyze even if results exist.",
)
@click.pass_context
def analyze(
    ctx: click.Context,
    analyzer: tuple[str, ...],
    tickers: str | None,
    force: bool,
) -> None:
    """Run sentiment analyzers on ingested filings."""
    config = _load_config(ctx)
    analyzers = _resolve_analyzers(analyzer, config)

    if not analyzers:
        console.print("[yellow]No analyzers enabled. Check config.[/yellow]")
        raise SystemExit(1)

    async def _run():
        store = await _create_store_async(config)
        try:
            ticker_list = (
                [t.strip().upper() for t in tickers.split(",")]
                if tickers
                else None
            )

            filings = await store.list_filings(ticker=ticker_list[0] if ticker_list and len(ticker_list) == 1 else None)

            if not filings:
                console.print("[yellow]No filings found. Run 'ingest' first.[/yellow]")
                raise SystemExit(1)

            if ticker_list:
                filings = [f for f in filings if f.ticker in ticker_list]

            total_results = 0
            for az in analyzers:
                az_name = getattr(az, "name", type(az).__name__)
                console.print(f"Running [bold]{az_name}[/bold] analyzer...")
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    task = progress.add_task(f"{az_name}...", total=len(filings))
                    for meta in filings:
                        try:
                            full_filing = await store.get_filing(meta.accession_number)
                            if full_filing is None:
                                progress.advance(task)
                                continue
                            for section in full_filing.sections.values():
                                try:
                                    result = az.analyze(section)
                                    if hasattr(result, "sentiment_score"):
                                        from edgar_sentinel.core import SimilarityResult
                                        if isinstance(result, SimilarityResult):
                                            await store.save_similarity(result)
                                        else:
                                            await store.save_sentiment(result)
                                    else:
                                        await store.save_similarity(result)
                                    total_results += 1
                                except Exception as exc:
                                    if ctx.obj["verbose"]:
                                        console.print(
                                            f"[red]Error analyzing section {section.section_name}: {exc}[/red]"
                                        )
                        except Exception as exc:
                            if ctx.obj["verbose"]:
                                console.print(
                                    f"[red]Error analyzing {meta.accession_number}: {exc}[/red]"
                                )
                        progress.advance(task)

            console.print(
                f"[green]\u2713[/green] Generated {total_results} analysis results"
            )
        finally:
            await store.close()

    _run_async(_run())


# ---------------------------------------------------------------------------
# signals
# ---------------------------------------------------------------------------


@cli.command()
@click.option(
    "--ticker",
    "-t",
    type=str,
    default=None,
    help="Single ticker to show signals for. Default: all.",
)
@click.option(
    "--date",
    "-d",
    "signal_date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=None,
    help="Signal date (YYYY-MM-DD). Default: today.",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json", "csv"], case_sensitive=False),
    default="table",
    help="Output format.",
)
@click.pass_context
def signals(
    ctx: click.Context,
    ticker: str | None,
    signal_date: datetime | None,
    output_format: str,
) -> None:
    """Build and display composite signals."""
    async def _run():
        from datetime import date as date_type

        from edgar_sentinel.analyzers.base import AnalysisResults
        from edgar_sentinel.core import CompositeMethod
        from edgar_sentinel.signals.builder import FilingDateMapping, SignalBuilder
        from edgar_sentinel.signals.composite import SignalComposite

        config = _load_config(ctx)
        store = await _create_store_async(config)
        try:
            as_of = signal_date.date() if signal_date else date_type.today()

            # Build composites from stored sentiment/similarity results
            filings_meta = await store.list_filings(
                ticker=ticker.upper() if ticker else None,
            )
            if not filings_meta:
                console.print("[yellow]No filings found. Run 'ingest' first.[/yellow]")
                raise SystemExit(1)

            all_sentiments = []
            all_similarities = []
            filing_dates: dict = {}
            for meta in filings_meta:
                sents = await store.get_sentiments(meta.accession_number)
                all_sentiments.extend(sents)
                filing_dates[meta.accession_number] = FilingDateMapping(
                    ticker=meta.ticker or "UNKNOWN",
                    filing_id=meta.accession_number,
                    filed_date=meta.filed_date,
                    signal_date=meta.filed_date,
                )

            if not all_sentiments and not all_similarities:
                console.print(
                    "[yellow]No analysis results found. Run 'analyze' first.[/yellow]"
                )
                raise SystemExit(1)

            analysis_results = AnalysisResults(
                sentiment_results=all_sentiments,
                similarity_results=all_similarities,
            )

            builder = SignalBuilder(config.signals)
            raw_signals = builder.build(analysis_results, filing_dates, as_of)

            if not raw_signals:
                console.print("[yellow]No signals could be built from analysis results.[/yellow]")
                raise SystemExit(1)

            composite = SignalComposite(method=CompositeMethod.EQUAL)
            composites = composite.combine(raw_signals, as_of_date=as_of)

            # Persist composites
            for c in composites:
                await store.save_composite(c)

            if not composites:
                console.print("[yellow]No composite signals produced.[/yellow]")
                raise SystemExit(1)

            # Sort by rank (or composite_score descending)
            composites.sort(key=lambda c: c.composite_score, reverse=True)

            if output_format == "json":
                _output_signals_json(composites)
            elif output_format == "csv":
                _output_signals_csv(composites)
            else:
                _output_signals_table(composites)
        finally:
            await store.close()

    _run_async(_run())


def _output_signals_table(composites) -> None:
    """Render composite signals as a Rich table."""
    table = Table(title="Composite Signals")
    table.add_column("Ticker", style="bold")
    table.add_column("Date")
    table.add_column("Composite", justify="right")
    table.add_column("Rank", justify="right")
    table.add_column("Components")

    for c in composites:
        components_str = " ".join(
            f"{k}={v:.3f}" for k, v in sorted(c.components.items())
        )
        table.add_row(
            c.ticker,
            str(c.signal_date),
            f"{c.composite_score:.4f}",
            str(c.rank or ""),
            components_str,
        )

    console.print(table)


def _output_signals_json(composites) -> None:
    """Write composite signals as JSON to stdout."""
    output = [c.model_dump(mode="json") for c in composites]
    click.echo(json.dumps(output, indent=2, default=str))


def _output_signals_csv(composites) -> None:
    """Write composite signals as CSV to stdout."""
    import csv
    import io

    if not composites:
        return

    all_components = sorted({name for c in composites for name in c.components})
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["ticker", "signal_date", "composite_score", "rank"] + all_components)
    for c in composites:
        row = [c.ticker, str(c.signal_date), f"{c.composite_score:.6f}", c.rank or ""]
        row.extend(f"{c.components.get(name, '')}" for name in all_components)
        writer.writerow(row)
    click.echo(buf.getvalue(), nl=False)


# ---------------------------------------------------------------------------
# backtest
# ---------------------------------------------------------------------------


@cli.command()
@click.option(
    "--start",
    "-s",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    required=True,
    help="Backtest start date (YYYY-MM-DD).",
)
@click.option(
    "--end",
    "-e",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    required=True,
    help="Backtest end date (YYYY-MM-DD).",
)
@click.option(
    "--rebalance",
    type=click.Choice(["monthly", "quarterly"], case_sensitive=False),
    default="quarterly",
    help="Rebalance frequency.",
)
@click.option("--quantiles", type=int, default=5, help="Number of quantile bins.")
@click.option(
    "--long-only",
    is_flag=True,
    default=False,
    help="Long-only portfolio (no short leg).",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json"], case_sensitive=False),
    default="table",
    help="Output format.",
)
@click.pass_context
def backtest(
    ctx: click.Context,
    start: datetime,
    end: datetime,
    rebalance: str,
    quantiles: int,
    long_only: bool,
    output_format: str,
) -> None:
    """Run a backtest simulation using composite signals."""
    async def _run():
        from edgar_sentinel.backtest import MetricsCalculator, run_backtest
        from edgar_sentinel.core import BacktestConfig, RebalanceFrequency

        config = _load_config(ctx)
        store = await _create_store_async(config)
        try:
            # Load all composite signals from storage
            composites = await store.get_composites()

            if not composites:
                console.print(
                    "[yellow]No composite signals found. Run 'signals' first.[/yellow]"
                )
                raise SystemExit(1)

            # Determine universe from signals
            universe = sorted({c.ticker for c in composites})

            bt_config = BacktestConfig(
                start_date=start.date(),
                end_date=end.date(),
                universe=universe,
                rebalance_frequency=RebalanceFrequency(rebalance),
                num_quantiles=quantiles,
                short_quantile=None if long_only else quantiles,
            )

            result = run_backtest(config=bt_config, signals=composites)

            calc = MetricsCalculator()
            metrics = calc.compute_all(result)

            if output_format == "json":
                _output_backtest_json(result, metrics)
            else:
                _output_backtest_table(result, metrics, bt_config)
        finally:
            await store.close()

    _run_async(_run())


def _output_backtest_table(result, metrics: dict, bt_config) -> None:
    """Render backtest results as Rich tables."""
    console.print()
    console.print(
        f"[bold]Backtest Results: {bt_config.start_date} \u2192 {bt_config.end_date}[/bold]"
    )
    console.print(
        f"  Universe: {len(bt_config.universe)} tickers  |  "
        f"Rebalance: {bt_config.rebalance_frequency}  |  "
        f"Quantiles: {bt_config.num_quantiles}"
    )
    console.print()

    table = Table(title="Performance Summary")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Total Return", f"{result.total_return:.1%}")
    table.add_row("Annualized Return", f"{result.annualized_return:.1%}")
    table.add_row("Sharpe Ratio", f"{result.sharpe_ratio:.2f}")
    table.add_row("Max Drawdown", f"{result.max_drawdown:.1%}")
    table.add_row("Turnover", f"{result.turnover:.1%}")

    if "sortino_ratio" in metrics:
        table.add_row("Sortino Ratio", f"{metrics['sortino_ratio']:.2f}")

    if result.information_ratio is not None:
        table.add_row("Information Ratio", f"{result.information_ratio:.2f}")

    if result.factor_exposures:
        table.add_section()
        for factor, loading in result.factor_exposures.items():
            table.add_row(f"  {factor}", f"{loading:.3f}")

    console.print(table)

    if "ic_mean" in metrics and metrics["ic_mean"] is not None:
        console.print(
            f"\nRank IC: {metrics['ic_mean']:.3f} "
            f"(ICIR: {metrics.get('ic_ir', 0):.2f})"
        )


def _output_backtest_json(result, metrics: dict) -> None:
    """Write backtest results as JSON to stdout."""
    output = {
        "result": result.model_dump(mode="json"),
        "metrics": {k: v for k, v in metrics.items() if v is not None},
    }
    click.echo(json.dumps(output, indent=2, default=str))


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--host", type=str, default="0.0.0.0", help="Bind address.")
@click.option("--port", "-p", type=int, default=8000, help="Port number.")
@click.option("--reload", is_flag=True, default=False, help="Auto-reload on code changes.")
@click.pass_context
def serve(ctx: click.Context, host: str, port: int, reload: bool) -> None:
    """Start the REST API server."""
    try:
        import uvicorn
    except ImportError:
        console.print(
            "[red]uvicorn not installed. Install with: "
            "pip install edgar-sentinel[api][/red]"
        )
        raise SystemExit(1)

    console.print(f"Starting edgar-sentinel API on [bold]{host}:{port}[/bold]")
    console.print(f"API docs: http://{host}:{port}/docs")

    uvicorn.run(
        "edgar_sentinel.api.app:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
    )


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


@cli.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show system status and data coverage."""
    async def _run():
        config = _load_config(ctx)
        store = await _create_store_async(config)
        try:
            stats = await _gather_stats(store)

            table = Table(title="EDGAR Sentinel Status")
            table.add_column("Metric", style="bold")
            table.add_column("Value", justify="right")

            table.add_row("Storage backend", config.storage.backend.value)
            table.add_row("Database path", config.storage.sqlite_path)
            table.add_section()
            table.add_row("Total filings", str(stats["total_filings"]))
            table.add_row("Unique tickers", str(stats["unique_tickers"]))
            table.add_row(
                "Date range",
                f"{stats['earliest_filing']} \u2192 {stats['latest_filing']}"
                if stats["total_filings"] > 0
                else "N/A",
            )
            table.add_section()
            table.add_row("Composite signals", str(stats["composite_signals"]))
            table.add_row("Latest signal date", stats.get("latest_signal_date", "N/A"))

            console.print(table)
        finally:
            await store.close()

    _run_async(_run())


async def _gather_stats(store) -> dict:
    """Gather basic statistics from storage."""
    filings = await store.list_filings()
    composites = await store.get_composites()

    tickers = {f.ticker for f in filings if f.ticker}
    filing_dates = [f.filed_date for f in filings]
    signal_dates = [c.signal_date for c in composites]

    return {
        "total_filings": len(filings),
        "unique_tickers": len(tickers),
        "earliest_filing": str(min(filing_dates)) if filing_dates else "N/A",
        "latest_filing": str(max(filing_dates)) if filing_dates else "N/A",
        "composite_signals": len(composites),
        "latest_signal_date": str(max(signal_dates)) if signal_dates else "N/A",
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
