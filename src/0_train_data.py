from requests import get
from io import BytesIO
import pandas as pd
import os

def baixa_nomes():

    if not os.path.exists("raw_data"):
        os.makedirs("raw_data")

    res = get("https://data.brasil.io/dataset/genero-nomes/nomes.csv.gz").content
    nomes = pd.read_csv(BytesIO(res), compression = "gzip")

    nomes["prop_female"] = (nomes.frequency_female / nomes.frequency_total).fillna(0)
    nomes = nomes[["first_name", "prop_female"]]
    nomes.to_csv("raw_data/nomes.csv", index = False)


if __name__ == "__main__":
    baixa_nomes()