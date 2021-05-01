.PHONY: dados discursos modelos
data_files = src/train_data.py src/validation_data.py
model_files = src/model.py

models/best_xgb.pkl: src/model.py raw_data/nomes.csv src/pipelines/__init__.py
	time python src/model.py

raw_data/deps.csv: src/case_data.py
	python src/case_data.py

raw_data/tse_validacao.csv: src/validation_data.py
	python src/validation_data.py

raw_data/nomes.csv: src/train_data.py
	python src/train_data.py

modelos:
	for x in $(model_files); do \
        python $$x ; \
    done

dados:
	for x in $(data_files); do \
        python $$x ; \
    done

discursos:
	time python src/case_data.py

