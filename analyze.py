from src.data.text_data import CorpusGenerator
from src.data.ges_data import GesDatasetGenerator

corpus_generator = CorpusGenerator('data/raw/waiting_list_corpus_raw/','Rene Lagos - SELECT_ID_CORTA_FOLIO_INGRESO_GES_RUTPACIENTE_ESPECIALIDAD_FECHA_201810301333.csv')
corpus_generator.load_files()
corpus_generator.preprocess()
corpus_generator.process()
corpus_generator.write_corpus('data/interim/corpus.txt')

ges_generator = GesDatasetGenerator('data/raw/waiting_list_corpus_raw/Rene Lagos - SELECT_ID_CORTA_FOLIO_INGRESO_GES_RUTPACIENTE_ESPECIALIDAD_FECHA_201810301333.csv')
ges_generator.preprocess()
ges_generator.split()
ges_generator.write_dataset('data/processed/','data/interim/')