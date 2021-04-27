raw_data/tse_validacao.csv: src/1_validation_data.py
	python src/1_validation_data.py

raw_data/nomes.csv: src/0_train_data.py
	python src/0_train_data.py

