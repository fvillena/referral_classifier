from src.data.text_data import CorpusGenerator
from src.data.ges_data import GesDatasetGenerator
from src.data.urg_data import UrgDatasetGenerator
from src.features.text_embedding import TextVectorizer
from src.models.ges_classifier import GesModelTrainer
from src.visualization.grid_search_viz import GridSearchVisualizer
from src.visualization.cross_val_viz import CrossValVisualizer
from src.visualization.roc_curve import RocCurve
from src.models.validation import StatisticalAnalysis
from src.models.validation import Performance
import os

corpus_generator = CorpusGenerator('data/raw/waiting_list_corpus_raw/','Rene Lagos - SELECT_ID_CORTA_FOLIO_INGRESO_GES_RUTPACIENTE_ESPECIALIDAD_FECHA_201810301333.csv')
corpus_generator.load_files()
corpus_generator.preprocess()
corpus_generator.process()
corpus_generator.write_corpus('data/processed/corpus.txt')

ges_generator = GesDatasetGenerator('data/raw/waiting_list_corpus_raw/Rene Lagos - SELECT_ID_CORTA_FOLIO_INGRESO_GES_RUTPACIENTE_ESPECIALIDAD_FECHA_201810301333.csv')
ges_generator.preprocess()
ges_generator.split()
ges_generator.write_dataset('data/processed/','data/interim/')

os.system('bash src/models/compute_embeddings.sh')

vectorizer = TextVectorizer('models/embeddings.vec','data/interim/train_text.txt','data/interim/test_text.txt')
vectorizer.vectorize_text()
vectorizer.write_data('ges','data/processed/')

trainer = GesModelTrainer('data/processed/train_text.npy','data/processed/train_age.txt','data/processed/train_labels.txt')
trainer.grid_search()
trainer.generate_report('reports/')

grid_search_viz = GridSearchVisualizer('reports/grid_search/')
grid_search_viz.plot('reports/figures/grid_search.pdf')

trainer.train_models('reports/grid_search/',n_jobs=-1)
trainer.generate_report('reports/')

cross_val_viz = CrossValVisualizer('reports/cross_val/')
cross_val_viz.plot('reports/figures/cross_val.pdf')

statistical_analyzer = StatisticalAnalysis('reports/cross_val/')
statistical_analyzer.analyze()
statistical_analyzer.generate_report('reports/statistical_analysis.json')

trainer.train_best_model('data/processed/test_text.npy','data/processed/test_age.txt','data/processed/test_labels.txt','reports/best_model_results.txt')

best_model_performance = Performance('reports/best_model_results.txt')
best_model_performance.analyze()
best_model_performance.generate_report('reports/best_model_performance.json')

rock = RocCurve('reports/best_model_performance.json')
rock.plot('reports/figures/roc_curve.pdf')

# URG

urg_generator = UrgDatasetGenerator('data/raw/waiting_list_corpus_raw/IQ_CONSOLIDADO_14102018_DEPURACION_UGD_revDJara_sinRUT.csv')
urg_generator.preprocess()
urg_generator.split()
urg_generator.write_dataset('data/processed/','data/interim/')

urg_vectorizer = TextVectorizer('models/embeddings.vec','data/interim/urg_train_text.txt','data/interim/urg_test_text.txt')
urg_vectorizer.vectorize_text()
urg_vectorizer.write_data('urg','data/processed/')

