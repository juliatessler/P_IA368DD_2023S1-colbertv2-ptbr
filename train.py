from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Trainer

if __name__=='__main__':
    with Run().context(RunConfig(nranks=1, experiment="mmarco")):

        config = ColBERTConfig(
            bsize=14,
            root="experiments/"
        )
        trainer = Trainer(
            triples="../data/triples.train.small.tsv",
            queries="../data/queries.tsv",
            collection="../data/collection.tsv",
            config=config,
        )

        checkpoint_path = trainer.train(checkpoint="neuralmind/bert-base-portuguese-cased")

        print(f"Saved checkpoint to {checkpoint_path}...")
