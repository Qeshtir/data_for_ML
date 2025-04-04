import dagster as dg
from dagster import file_relative_path
from src.enrich_data import main as enricher
from src.scraper_final import scrape_it
from src.cleaner import clean

ORIGINAL_DF = file_relative_path(__file__, "../artifacts/cian_dataset.dill")
ENRICHED_DF = file_relative_path(__file__, "../artifacts/cian_dataset_enriched.dill")
CLEANED_DF = file_relative_path(__file__, "../artifacts/cian_dataset_cleaned.dill")


@dg.asset
def scrape_asset():
    scrape_it(ORIGINAL_DF)


@dg.asset(deps=[scrape_asset])
def enrich_asset():
    enricher(ORIGINAL_DF, ENRICHED_DF)


@dg.asset(deps=[enrich_asset])
def clean_asset():
    clean(ENRICHED_DF, CLEANED_DF)


daily_refresh_job = dg.define_asset_job(
    "daily_refresh", selection=["scrape_asset", "enrich_asset", "clean_asset"]
)

daily_schedule = dg.ScheduleDefinition(
    job=daily_refresh_job,
    cron_schedule="0 0 * * *",  # Runs at midnight daily
    execution_timezone="Europe/Moscow",
)

defs = dg.Definitions(
    assets=[scrape_asset, enrich_asset, clean_asset],
    jobs=[daily_refresh_job],
    schedules=[daily_schedule],
)