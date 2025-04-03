import dagster as dg
from dagster import file_relative_path
from src.enrich_data import main as enricher
from src.scraper_final import scrape_it

OUTPUT_PATH = file_relative_path(__file__, "../artifacts/cian_dataset.dill")
LOAD_PATH = file_relative_path(__file__, "../artifacts/cian_dataset_enriched.dill")


@dg.asset
def scrape_asset():
    scrape_it(OUTPUT_PATH)


@dg.asset(deps=[scrape_asset])
def enrich_asset():
    enricher(OUTPUT_PATH, LOAD_PATH)


daily_refresh_job = dg.define_asset_job(
    "daily_refresh", selection=["scrape_asset", "enrich_asset"]
)

daily_schedule = dg.ScheduleDefinition(
    job=daily_refresh_job,
    cron_schedule="0 0 * * *",  # Runs at midnight daily
    execution_timezone="Europe/Moscow",
)

defs = dg.Definitions(
    assets=[scrape_asset, enrich_asset],
    jobs=[daily_refresh_job],
    schedules=[daily_schedule],
)