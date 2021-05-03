.PHONY: dados discursos modelos lstm clear
data_files = src/train_data.py src/validation_data.py
model_files = src/model.py

models/best_lstm_256units.h5: src/lstm_model.py
	time python src/lstm_model.py

models/best_xgb.pkl: src/model.py raw_data/nomes.csv src/pipelines/__init__.py
	time python src/model.py

raw_data/deps.csv: src/deputies_speech_data.py
	python src/deputies_speech_data.py

raw_data/tse_validacao.csv: src/validation_data.py
	python src/validation_data.py

raw_data/nomes.csv: src/train_data.py
	python src/train_data.py

lstm:
	python src/lstm_model.py

modelos:
	for x in $(model_files); do \
        python $$x ; \
    done

dados:
	for x in $(data_files); do \
        python $$x ; \
    done

discursos:
	time python src/deputies_speech_data.py

clear:
	rm -r raw_data results src/models 

