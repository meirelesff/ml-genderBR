.PHONY: dados discursos
data_files = src/0_train_data.py src/1_validation_data.py

models/best_xgb.pkl: src/3_model.py raw_data/nomes.csv src/pipelines/__init__.py \
					settings.toml
	time python src/3_model.py

raw_data/deps.csv: src/2_case_data.py
	python src/2_case_data.py

raw_data/tse_validacao.csv: src/1_validation_data.py
	python src/1_validation_data.py

raw_data/nomes.csv: src/0_train_data.py
	python src/0_train_data.py

dados:
	for x in $(data_files); do \
        python $$x ; \
    done

discursos:
	time python src/2_case_data.py

