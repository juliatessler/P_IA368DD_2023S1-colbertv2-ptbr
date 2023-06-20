from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer

if __name__=='__main__':
    with Run().context(RunConfig(nranks=1, experiment="mmarco")):

        config = ColBERTConfig(
            nbits=2,
            root="/home/manoel/Documents/Doutorado/P_IA368DD_2023S1/Projeto Final/data/indexes",
        )
        indexer = Indexer(checkpoint="/home/manoel/Documents/Doutorado/P_IA368DD_2023S1/Projeto Final/ColBERT/experiments/msmarco/none/2023-06/07/15.01.44/checkpoints/colbert-70000", config=config)
        indexer.index(name="mmarco.nbits=2", collection="/home/manoel/Documents/Doutorado/P_IA368DD_2023S1/Projeto Final/data/collection.tsv", overwrite="resume")
