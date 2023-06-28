from colbert.data import Queries
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Searcher

if __name__=='__main__':
    with Run().context(RunConfig(nranks=1, experiment="mmarco")):

        config = ColBERTConfig(
            root="./ColBERT/experiments/mmarco/indexes/",
        )
        searcher = Searcher(index="msmarco.nbits=2", config=config)
        queries = Queries("./data/portuguese_queries.dev.small.tsv")
        ranking = searcher.search_all(queries, k=100)
        ranking.save("msmarco.nbits=2.ranking.tsv")
