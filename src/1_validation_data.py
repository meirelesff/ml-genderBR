from unidecode import unidecode
from zipfile import ZipFile
from requests import get
import pandas as pd
import shutil
import os

def baixa_validacao_tse(url, tmp="tmp", chunk_size=256):

    if not os.path.exists(tmp):
        os.makedirs(tmp)

    # Download and extract zip file
    res = get(url, stream=True)
    with open(tmp + "/tse.zip", "wb") as fd:
        for chunk in res.iter_content(chunk_size=chunk_size):
            fd.write(chunk)

    dados = ZipFile(tmp + "/tse.zip")
    dados.extractall(path=tmp)

    # Read, select, and filter data
    nomes_tse = pd.read_csv(tmp + "/consulta_cand_2020_BRASIL.csv", sep=";",\
        encoding="latin1")
    nomes_tse = nomes_tse[["SG_UF", "SG_UF_NASCIMENTO", "NR_IDADE_DATA_POSSE",\
        "DS_CARGO", "DS_COR_RACA", "SQ_CANDIDATO", "NM_CANDIDATO", "DS_GENERO"]]
    nomes_tse.columns = nomes_tse.columns.str.lower()

    # Cleaning, strip accents
    nomes_tse = nomes_tse.query("sg_uf_nascimento != 'ZZ'") # Brazilian-born only
    nomes_tse.loc[:, "nm_candidato"] = nomes_tse.nm_candidato\
        .apply(lambda x: unidecode(x)\
            .lower()\
                .split(" ")[0])

    nomes_tse.to_csv("raw_data/tse_validacao.csv", index = False)

    # Clean tmp folder
    shutil.rmtree(tmp)


if __name__ == "__main__":
    url = "https://cdn.tse.jus.br/estatistica/sead/odsele/consulta_cand/consulta_cand_2020.zip"
    baixa_validacao_tse(url)