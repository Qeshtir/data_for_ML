import asyncio
import aiohttp
import os
import pandas as pd
import random
import dill
import copy

from bs4 import BeautifulSoup

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0",
]

BASE_HEADERS = {
            "User-Agent": random.choice(USER_AGENTS),
            "Referer": "https://www.cian.ru/",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }


BASE_URL = "https://www.cian.ru/cat.php?deal_type=sale&engine_version=2&object_type[0]=2&offer_type=flat&p="
BASE_URL_ENDING = "&region=-1"

OUTPUT_PATH = "../artifacts/cian_dataset.dill"

# Фикс 429-й от циана
CONCURRENT_REQUESTS = 5

async def fetch_page(session, url, max_retries=3):
    attempt = 0
    local_headers = copy.copy(BASE_HEADERS)
    while attempt < max_retries:
        try:
            async with session.get(url, headers=local_headers) as response:
                if response.status == 429:
                    # Если получен статус 429, задержка и повторный запрос
                    print(f"Получен 429 для {url}. Задержка и повторный запрос.")
                    await asyncio.sleep(10)
                    attempt += 1
                    continue
                elif response.status == 403:
                    # Если получен статус 403, смена User-Agent и повторный запрос
                    print(f"Получен 403 для {url}. Меняем User-Agent и повторяем запрос.")
                    local_headers["User-Agent"] = random.choice(USER_AGENTS)
                    await asyncio.sleep(3)
                    attempt += 1
                    continue
                else:
                    text = await response.text()
                    # Задержка для защиты от перегрузки сервера
                    await asyncio.sleep(1)
                    return text
        except Exception as e:
            print(f"Ошибка запроса {url}: {e}")
            await asyncio.sleep(3)
            attempt += 1
    print(f"Не удалось получить данные по {url} после {max_retries} попыток.")
    return ""


def scrape(html, flats):
    if not html:
        return
    soup = BeautifulSoup(html, 'html.parser')
    for i, item in enumerate(soup.find_all('article')):
        flats_list = []

        # Наименование объявления. Пример:
        # <span data-mark="OfferTitle"><span class="">2-комн. квартира, 86,1 м², 4/13 этаж</span></span>
        span_heading = item.find('span', attrs={'data-mark': 'OfferTitle'})
        if span_heading:
            inner_heading_span = span_heading.find('span', )
            if inner_heading_span:
                flats_list.append(inner_heading_span.text)
            else:
                flats_list.append("No_content")
                #continue
        else:
            flats_list.append("No_content")
            #continue

        # Наименование объявления, вторая строка. Пример:
        span_subheading = item.find('span', attrs={'data-mark': 'OfferSubtitle'})
        if span_subheading:
            flats_list.append(span_subheading.text)
        else:
            flats_list.append("No_content")

        # Наименование ЖК. Пример:
        # <a class="_93444fe79c--jk--dIktL" href="https://zhk-frunzenskaya-naberezhnaya-i.cian.ru/" target="_blank">ЖК «Клубный город-парк «Фрунзенская набережная»</a>
        a_jk = item.find('a', class_='_93444fe79c--jk--dIktL')
        if a_jk:
            flats_list.append(a_jk.text)
        else:
            flats_list.append("No_content")

        # Срок сдачи. Пример:
        # <span data-mark="Deadline">сдача ГК: 3 кв. 2027 года</span>
        deadline = item.find('span', attrs={'data-mark': 'Deadline'})
        if deadline:
            flats_list.append(deadline.text)
        else:
            flats_list.append("No_content")

        # Адрес объекта. Пример:
        # <a data-name="GeoLabel" class="_93444fe79c--link--NQlVc" href="/cat.php?deal_type=sale&amp;engine_version=2&amp;object_type%5B0%5D=2&amp;offer_type=flat&amp;region=1" target="_blank">Москва</a> их там несколько
        geolables = item.find_all('a', attrs={'data-name': 'GeoLabel'})
        if geolables:
            g_m = []
            for geo in geolables:
                g_m.append(geo.text)
            g_ = ", ".join(g_m)
            flats_list.append(g_)
        else:
            flats_list.append("No_content")

        # Цена в рублях.
        span = item.find('span', attrs={'data-mark': 'MainPrice'})
        if span:
            inner_span = span.find('span', )
            if inner_span:
                flats_list.append(inner_span.text)
            else:
                flats_list.append("No_content")
        else:
            flats_list.append("No_content")

        # Цена за м**2. Пример:
        # <p data-mark="PriceInfo">2 711 614&nbsp;₽/м²</p>
        meter_price = item.find('p', attrs={'data-mark': 'PriceInfo'})
        if meter_price:
            flats_list.append(meter_price.text)
        else:
            flats_list.append("No_content")
        text_class = "_93444fe79c--color_text-primary-default--vSRPB _93444fe79c--lineHeight_20px--fX7_V _93444fe79c--fontWeight_normal--JEG_c _93444fe79c--fontSize_14px--reQMB _93444fe79c--display_block--KYb25 _93444fe79c--text--b2YS3 _93444fe79c--text_letterSpacing__normal--yhcXb"


        # p tag для описания
        desc = item.find('p', class_=text_class)
        if desc:
            flats_list.append(desc.text)
        else:
            flats_list.append("No_content")

        # To_Do парсить наименование застройщика. Пример:
        # <a href="https://cian.ru/zastroishchik-sminex-586/" target="_blank" rel="noopener" class="_93444fe79c--link--wbne1"><span style="letter-spacing:-0.2px" class="_93444fe79c--color_current_color--KRvSV _93444fe79c--color_current_color--MqB6f _93444fe79c--lineHeight_6u--cedXD _93444fe79c--fontWeight_bold--BbhnX _93444fe79c--fontSize_16px--QNYmt _93444fe79c--display_block--KYb25 _93444fe79c--text--b2YS3">Sminex</span></a>

        flats.loc[len(flats)] = flats_list


async def fetch_with_semaphore(session, url, semaphore):
    async with semaphore:
        return await fetch_page(session, url)


async def async_htmls(p):
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    async with aiohttp.ClientSession() as session:
        tasks = [
            fetch_with_semaphore(session, BASE_URL + str(i) + BASE_URL_ENDING, semaphore)
            for i in range(1, p)
        ]
        htmls = await asyncio.gather(*tasks, return_exceptions=True)
    return htmls


def load_existing_data(OUTPUT_PATH):
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "rb") as f:
            try:
                return dill.load(f)
            except Exception as e:
                print(f"Ошибка загрузки существующих данных: {e}")
    return pd.DataFrame(columns=["object_title", "object_subtitle", "jk_name", "deadline", "geo", "price", "meter_price", "desc"])


def save_new_data(new_df, OUTPUT_PATH):
    old_df = load_existing_data(OUTPUT_PATH)
    combined = pd.concat([old_df, new_df], ignore_index=True)
    combined.drop_duplicates(inplace=True)
    after = len(combined)
    with open(OUTPUT_PATH, "wb") as f:
        dill.dump(combined, f)
    print(f"Сохранено {after} объектов (новых добавлено: {after - len(old_df)})")


def scrape_it(OUTPUT_PATH):
    # костыль под винду
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    flats = pd.DataFrame(
        columns=["object_title", "object_subtitle", "jk_name", "deadline", "geo", "price", "meter_price", "desc"])

    p = 100

    htmls = asyncio.run(async_htmls(p))
    for document in htmls:
        scrape(document, flats)

    save_new_data(flats, OUTPUT_PATH)

    loaded_flats = load_existing_data(OUTPUT_PATH)

    print(loaded_flats.head(10))
    print(f"Saved {len(loaded_flats)} objects")
