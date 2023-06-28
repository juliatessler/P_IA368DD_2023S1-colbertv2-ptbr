from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer

if __name__=='__main__':
    with Run().context(RunConfig(nranks=1, experiment="mmarco")):

        config = ColBERTConfig(
            nbits=2,
            root="./data/indexes",
        )
        indexer = Indexer(checkpoint="/path/to/colbert/checkpoint", config=config)
        indexer.index(name="mmarco.nbits=2", collection="./data/collection.tsv", overwrite="resume")
