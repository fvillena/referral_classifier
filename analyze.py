from src.data.text_data import CorpusGenerator

generator = CorpusGenerator('data/raw/waiting_list_corpus_raw/','Rene Lagos - SELECT_ID_CORTA_FOLIO_INGRESO_GES_RUTPACIENTE_ESPECIALIDAD_FECHA_201810301333.csv')
generator.load_files()
generator.preprocess()
generator.process()
generator.write_corpus('data/interim/corpus.txt')