import pandas as pd
import datetime as dt
import numpy as np
import dotenv
import os
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
dotenv.load_dotenv()

POSTGRES_ADDRESS=os.getenv("POSTGRES_ADDRESS")
POSTGRES_USER=os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD=os.getenv("POSTGRES_PASSWORD")
POSTGRES_DATABASE=os.getenv("POSTGRES_DATABASE")

data = pd.read_csv("data/raw/waiting_list_corpus_raw/Rene Lagos - SELECT_ID_CORTA_FOLIO_INGRESO_GES_RUTPACIENTE_ESPECIALIDAD_FECHA_201810301333.csv")
data["INGRESO"] = pd.to_datetime(data.INGRESO,errors="coerce")
data["FECHANACIMIENTO"] = pd.to_datetime(data.FECHANACIMIENTO,format="%d/%m/%y")
future = data['FECHANACIMIENTO'] > '2018-01-01'
data.loc[future, 'FECHANACIMIENTO'] -= dt.timedelta(days=365.25*100)
data["GES"] = np.where(data["GES"] == "SI", True,np.where(data["GES"] == "NO", False, np.nan))
chunks = np.array_split(data, 100)
for i,chunk in enumerate(chunks):
    chunk.to_sql("ges",f'postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_ADDRESS}/{POSTGRES_DATABASE}',if_exists="append",index=False)
    logger.info(f'{i} %')