import dagster as dg
from dagster import file_relative_path
from src.enrich_data import main as enricher
from src.scraper_final import scrape_it
from src.cleaner import clean
from src.text_and_img_processor import text_and_img_processor as processor

ORIGINAL_DF = file_relative_path(__file__, "../artifacts/cian_dataset_with_images.dill")
ENRICHED_DF = file_relative_path(__file__, "../artifacts/cian_dataset_enriched.dill")
CLEANED_DF = file_relative_path(__file__, "../artifacts/cian_dataset_cleaned.dill")
PROCESSED_DF = file_relative_path(__file__, "../artifacts/multimodal_df.dill")
NAVEC_PATH = file_relative_path(__file__,  '../artifacts/navec_news_v1_1B_250K_300d_100q.tar')
NER_PATH = file_relative_path(__file__, '../artifacts/slovnet_ner_news_v1.tar')


@dg.asset
def scrape_asset():
    scrape_it(ORIGINAL_DF)


@dg.asset(deps=[scrape_asset])
def enrich_asset():
    enricher(ORIGINAL_DF, ENRICHED_DF)


@dg.asset(deps=[enrich_asset])
def clean_asset():
    clean(ENRICHED_DF, CLEANED_DF)


@dg.asset(deps=[scrape_asset])
def ranking_asset():
    processor(ORIGINAL_DF, PROCESSED_DF, NAVEC_PATH, NER_PATH)


daily_refresh_job = dg.define_asset_job(
    "daily_refresh", selection=["scrape_asset", "enrich_asset", "clean_asset", "ranking_asset"]
)

daily_schedule = dg.ScheduleDefinition(
    job=daily_refresh_job,
    cron_schedule="0 0 * * *",  # Runs at midnight daily
    execution_timezone="Europe/Moscow",
)

defs = dg.Definitions(
    assets=[scrape_asset, enrich_asset, clean_asset, ranking_asset],
    jobs=[daily_refresh_job],
    schedules=[daily_schedule],
)