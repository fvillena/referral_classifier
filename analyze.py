from src.data.text_data import CorpusGenerator
from src.data.ges_data import GesDatasetGenerator
from src.features.text_embedding import TextVectorizer
from src.models.ges_classifier import GesModelTrainer
from src.visualization.grid_search_viz import GridSearchVisualizer
from src.visualization.cross_val_viz import CrossValVisualizer
from src.models.validation import StatisticalAnalysis
import os

# corpus_generator = CorpusGenerator('data/raw/waiting_list_corpus_raw/','Rene Lagos - SELECT_ID_CORTA_FOLIO_INGRESO_GES_RUTPACIENTE_ESPECIALIDAD_FECHA_201810301333.csv')
# corpus_generator.load_files()
# corpus_generator.preprocess()
# corpus_generator.process()
# corpus_generator.write_corpus('data/processed/corpus.txt')

# ges_generator = GesDatasetGenerator('data/raw/waiting_list_corpus_raw/Rene Lagos - SELECT_ID_CORTA_FOLIO_INGRESO_GES_RUTPACIENTE_ESPECIALIDAD_FECHA_201810301333.csv')
# ges_generator.preprocess()
# ges_generator.split()
# ges_generator.write_dataset('data/processed/','data/interim/')

# os.system('bash src/models/compute_embeddings.sh')

# vectorizer = TextVectorizer('models/embeddings.vec','data/interim/train_text.txt','data/interim/test_text.txt')
# vectorizer.vectorize_text()
# vectorizer.write_data('data/processed/')

trainer = GesModelTrainer('data/processed/train_text.npy','data/processed/train_age.txt','data/processed/train_labels.txt')
# trainer.grid_search()
# trainer.generate_report('reports/')

# grid_search_viz = GridSearchVisualizer('reports/grid_search/')
# grid_search_viz.plot('reports/figures/grid_search.pdf')

# trainer.train_models('reports/grid_search/',n_jobs=-1)
# trainer.generate_report('reports/')

# cross_val_viz = CrossValVisualizer('reports/cross_val/')
# cross_val_viz.plot('reports/figures/cross_val.pdf')

# statistical_analyzer = StatisticalAnalysis('reports/cross_val/')
# statistical_analyzer.analyze()
# statistical_analyzer.generate_report('reports/statistical_analysis.json')

trainer.train_best_model('data/processed/test_text.npy','data/processed/test_age.txt','data/processed/test_labels.txt','results/best_model_results.txt')